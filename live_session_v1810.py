"""V18.10 intraday provenance: never mistake a completed daily bar for a live quote.

Only a same-session, timestamped, recent, official matched trade can be used
for live entry recommendations. Other rows are visible as diagnostics only.
This module deliberately has no network requests and does not alter Alpha.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any

import pandas as pd

TAIPEI = dt.timezone(dt.timedelta(hours=8))
MAX_EXECUTION_TRADE_AGE_SECONDS = 180
MAX_FUTURE_CLOCK_SKEW_SECONDS = 90


def _ts(value: Any) -> dt.datetime | None:
    if value is None or not str(value).strip():
        return None
    try:
        stamp = pd.to_datetime(value, errors='coerce')
        if pd.isna(stamp):
            return None
        stamp = stamp.to_pydatetime() if hasattr(stamp, 'to_pydatetime') else stamp
        return stamp.replace(tzinfo=TAIPEI) if stamp.tzinfo is None else stamp.astimezone(TAIPEI)
    except (TypeError, ValueError, OverflowError):
        return None


def quote_freshness(quote: dict, now: dt.datetime, *, official_verified: bool,
                    max_age_seconds: int = MAX_EXECUTION_TRADE_AGE_SECONDS) -> dict:
    """Individual quote gate; NEVER infer quote time from network-fetch time.

    Return `can_execute` only for a verified official matched price, a complete
    session timestamp and an acceptable latency. A previous session price may
    still be shown with a clear non-live label, never silently promoted.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=TAIPEI)
    else:
        now = now.astimezone(TAIPEI)
    value = quote.get('close')
    try:
        price = float(value)
    except (TypeError, ValueError):
        price = float('nan')
    if not math.isfinite(price) or price <= 0:
        # A missing MIS `z` does not prove that the stock never traded today.
        # `v` may show cumulative volume even when the live matched-price field
        # is absent. Separate this condition from a genuine no-trade quote.
        try:
            traded_volume = float(quote.get('total_volume'))
        except (TypeError, ValueError):
            traded_volume = float('nan')
        stamp_raw = str(quote.get('date') or '').strip()
        stamp = _ts(stamp_raw)
        active_today = stamp is not None and stamp.astimezone(TAIPEI).date() == now.date()
        has_volume = bool(active_today and math.isfinite(traded_volume) and traded_volume > 0)
        return {'can_execute': False,
                'reason': 'MATCHED_PRICE_NOT_PUBLISHED' if has_volume else 'NO_MATCHED_PRICE',
                'label': '今日有成交，最新成交價未回傳' if has_volume else '最新成交價待取得',
                'price': None, 'age_seconds': None, 'quote_time': '',
                'previous_close': quote.get('previous_close')}
    # A remembered actual price is useful evidence, but ceases to be the
    # latest executable print if later cumulative volume has risen and the
    # official source omitted the corresponding new matched price.
    if bool(quote.get('volume_increased_after_last_trade')) or bool(quote.get('volume_uncertain_after_last_trade')):
        return {'can_execute': False, 'reason': 'NEWER_TRADE_PRICE_UNKNOWN',
                'label': '後續有成交或成交量待核，新成交價缺漏', 'price': price,
                'age_seconds': None, 'quote_time': str(quote.get('date') or ''),
                'previous_close': quote.get('previous_close')}
    stamp_raw = str(quote.get('date') or '').strip()
    # A date without an explicit HH:MM:SS is insufficient for a live decision.
    if ' ' not in stamp_raw and 'T' not in stamp_raw:
        return {'can_execute': False, 'reason': 'NO_TRADE_TIME', 'label': '缺成交時間', 'price': price,
                'age_seconds': None, 'quote_time': stamp_raw}
    stamp = _ts(stamp_raw)
    if stamp is None:
        return {'can_execute': False, 'reason': 'BAD_TRADE_TIME', 'label': '時間無法核對', 'price': price,
                'age_seconds': None, 'quote_time': stamp_raw}
    time_label = stamp.strftime('%m/%d %H:%M:%S')
    if stamp.date() != now.date():
        return {'can_execute': False, 'reason': 'OLD_SESSION', 'label': '非今日成交', 'price': price,
                'age_seconds': None, 'quote_time': time_label}
    age = (now - stamp).total_seconds()
    if age < -MAX_FUTURE_CLOCK_SKEW_SECONDS:
        return {'can_execute': False, 'reason': 'FUTURE_TIMESTAMP', 'label': '時間異常', 'price': price,
                'age_seconds': int(age), 'quote_time': time_label}
    if age > max_age_seconds:
        return {'can_execute': False, 'reason': 'OLD_TRADE', 'label': '成交價過舊', 'price': price,
                'age_seconds': int(age), 'quote_time': time_label}
    if not official_verified:
        return {'can_execute': False, 'reason': 'NO_OFFICIAL_VERIFICATION', 'label': '官方未核價',
                'price': price, 'age_seconds': max(0, int(age)), 'quote_time': time_label}
    return {'can_execute': True, 'reason': 'VERIFIED_CURRENT_TRADE', 'label': '官方成交已核價',
            'price': price, 'age_seconds': max(0, int(age)), 'quote_time': time_label}


def evaluate_quote_rows(df: pd.DataFrame, now: dt.datetime, *, official_verified: bool) -> dict[str, dict]:
    """For render: map a ticker stock_id to verified/diagnostic quote metadata."""
    if df is None or df.empty or 'stock_id' not in df.columns:
        return {}
    out = {}
    for _, item in df.iterrows():
        stock_id = str(item.get('stock_id') or '')
        if stock_id in {'001', '101'} or not stock_id:
            continue
        q = quote_freshness(item.to_dict(), now, official_verified=official_verified)
        q['previous_close'] = item.get('previous_close')
        try:
            chg = float(item.get('change_rate'))
            q['change_pct'] = chg if math.isfinite(chg) else None
        except (TypeError, ValueError):
            q['change_pct'] = None
        out[stock_id] = q
    return out
