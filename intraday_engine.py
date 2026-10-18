"""Intraday strength overlay for Taiwan Alpha Radar V16.6 Dynamic Intraday Freeze.

The completed-day model decides *what is structurally attractive*. This module
adds a one-minute full-market discovery pass, then re-orders a bounded detail
pool using current-session strength. It never replaces long-horizon evidence
with a partial intraday candle.

Realtime provider chain:
1) FinMind realtime snapshot when a token is configured;
2) TWSE/TPEx Market Information System (MIS) 5-second public quote page as a
   no-token fallback for the user's own research. Public redistribution of TWSE
   realtime data can require a separate information-use agreement.
"""
from __future__ import annotations

import datetime
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from policy_engine import evaluate_entry_timing, position_guidance
from strategy_config import (
    HORIZONS as CONFIG_HORIZONS,
    INTRADAY_BLEND_SCHEDULE,
    INTRADAY_FACTOR_RANGES,
    INTRADAY_FACTOR_WEIGHTS,
    INTRADAY_SPREAD_FREE_PCT,
    INTRADAY_SPREAD_PENALTY_PER_PCT,
    INTRADAY_STATE,
    LIVE_SCAN,
    PANIC_REVERSAL_POLICY,
)

REALTIME_URL = "https://api.finmindtrade.com/api/v4/taiwan_stock_tick_snapshot"
MIS_URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
MIS_HOME = "https://mis.twse.com.tw/stock/index.jsp"

# MIS's `z` is a matched price for *this response*, not a guaranteed always-on
# last-sale price. A live Streamlit rerun must not erase a prior, independently
# timestamped matched price merely because `z` temporarily becomes `-`.
# Memory is process-local, bounded and day-scoped. It is never a replacement
# for either a true trade timestamp or the 180-second execution freshness gate.
_OFFICIAL_TRADE_LOCK = threading.RLock()
_OFFICIAL_TRADE_MEMORY: dict[tuple[str, str], dict] = {}


def _official_trade_timestamp(row: dict) -> datetime.datetime | None:
    value = str(row.get("date") or "").strip()
    if " " not in value:
        return None
    try:
        stamp = datetime.datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
    return stamp.astimezone(datetime.timezone(datetime.timedelta(hours=8)))


def _remember_official_trade(row: dict, now: datetime.datetime) -> None:
    """Retain only actual matched `z` with an explicit same-session time."""
    stamp = _official_trade_timestamp(row)
    price = _finite(row.get("close"), np.nan)
    if stamp is None or stamp.date() != now.date() or not (np.isfinite(price) and price > 0):
        return
    # Explicitly reject large future drift; a misleading timestamp can never
    # become a cache-backed executable trade.
    if (now - stamp).total_seconds() < -90:
        return
    key = (str(row.get("market") or ""), str(row.get("stock_id") or ""))
    if not key[0] or not key[1] or key[1] in {"001", "101"}:
        return
    with _OFFICIAL_TRADE_LOCK:
        prior = _OFFICIAL_TRADE_MEMORY.get(key)
        prior_stamp = _official_trade_timestamp(prior) if prior else None
        if prior_stamp is None or stamp >= prior_stamp:
            _OFFICIAL_TRADE_MEMORY[key] = dict(row, price_origin="本次官方成交")
        # Memory remains tiny: purge another date, irrespective of stock count.
        for old_key, old in list(_OFFICIAL_TRADE_MEMORY.items()):
            old_stamp = _official_trade_timestamp(old)
            if old_stamp is None or old_stamp.date() != now.date():
                del _OFFICIAL_TRADE_MEMORY[old_key]


def _recalled_official_trade(stock_id: str, market: str, now: datetime.datetime) -> dict | None:
    key = (market, stock_id)
    with _OFFICIAL_TRADE_LOCK:
        item = _OFFICIAL_TRADE_MEMORY.get(key)
        stamp = _official_trade_timestamp(item) if item else None
        if stamp is None or stamp.date() != now.date():
            _OFFICIAL_TRADE_MEMORY.pop(key, None)
            return None
        if (now - stamp).total_seconds() < -90:
            return None
        return dict(item, price_origin="本交易日先前核實成交")


def _taipei_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))


def _stock_id(ticker: str) -> str:
    return str(ticker or "").split(".")[0].strip()


def market_is_open(now: datetime.datetime | None = None) -> bool:
    now = now or _taipei_now()
    if now.weekday() >= 5:
        return False
    t = now.timetz().replace(tzinfo=None)
    return datetime.time(9, 0) <= t <= datetime.time(13, 30)


def _finite(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _clip01(v: float, lo: float, hi: float) -> float:
    if not np.isfinite(v):
        return 0.5
    if hi <= lo:
        return 0.5
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def intraday_blend_weight(horizon: str, now: datetime.datetime | None = None) -> float:
    """Return a time-matured overlay weight for the current session.

    The daily model dominates immediately after the open.  As more of the
    session is observed, the realtime overlay grows toward its configured cap.
    """
    horizon = horizon if horizon in CONFIG_HORIZONS else "short"
    now = now or _taipei_now()
    minutes = (now.hour * 60 + now.minute + now.second / 60.0) - 9 * 60
    minutes = float(np.clip(minutes, 0.0, 270.0))
    schedule = INTRADAY_BLEND_SCHEDULE[horizon]
    weight = float(schedule[0][1])
    for minute_mark, configured_weight in schedule:
        if minutes >= float(minute_mark):
            weight = float(configured_weight)
        else:
            break
    return weight


@dataclass
class RealtimeStatus:
    available: bool
    reason: str = ""
    quote_date: str = ""
    source: str = "FinMind realtime snapshot"
    market_open: bool = False
    fetched_at: str = ""
    verified: bool = False
    verification: str = ""
    requested_count: int = 0
    executable_count: int = 0
    diagnostic_counts: dict = field(default_factory=dict)
    quote_diagnostics: list = field(default_factory=list)
    auxiliary_quotes: list = field(default_factory=list)
    request_diagnostics: dict = field(default_factory=dict)


class FinMindRealtimeClient:
    """Realtime client with a no-token official MIS fallback.

    The class name is kept for backward compatibility with app.py.  When a
    FinMind token exists it is tried first.  If the token is absent, lacks the
    realtime entitlement, or the response is stale/unavailable, the client
    falls back to TWSE/TPEx MIS quotes.
    """
    def __init__(self, token: str | None = None):
        self.token = (token if token is not None else os.getenv("FINMIND_TOKEN", "")).strip()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 AlphaRadar/16.6",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://mis.twse.com.tw/stock/index.jsp",
        })
        # Never send a FinMind bearer token to the TWSE MIS website.
        # Authorization is attached only to the FinMind endpoint request.
        self._mis_bootstrapped = False

    @property
    def configured(self) -> bool:
        # MIS does not require a user token, so the realtime layer is configured
        # even when FINMIND_TOKEN is absent.
        return True

    def _request_ids(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        params: list[tuple[str, str]] = []
        for stock_id in ids:
            params.append(("data_id", stock_id))
        if self.token:
            params.append(("token", self.token))
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        r = self.session.get(REALTIME_URL, params=params, timeout=10, headers=headers)
        r.raise_for_status()
        js = r.json()
        if not isinstance(js, dict):
            return []
        status = js.get("status")
        if status not in (None, 200, "200"):
            msg = str(js.get("msg") or js.get("message") or "FINMIND_API_ERROR")
            raise RuntimeError(msg)
        data = js.get("data", [])
        return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []

    @staticmethod
    def _num(v, default=np.nan):
        try:
            if v in (None, "", "-"):
                return default
            x = float(str(v).replace(",", ""))
            return x if np.isfinite(x) else default
        except Exception:
            return default

    @classmethod
    def _first_book_value(cls, v):
        if v in (None, "", "-"):
            return np.nan
        for part in str(v).split("_"):
            x = cls._num(part)
            if np.isfinite(x):
                return x
        return np.nan

    def _bootstrap_mis(self):
        if self._mis_bootstrapped:
            return
        try:
            self.session.get(MIS_HOME, timeout=3)
        except Exception:
            pass
        self._mis_bootstrapped = True

    @staticmethod
    def _mis_channel(ticker: str) -> str:
        ticker = str(ticker or "").strip().upper()
        stock_id = _stock_id(ticker)
        prefix = "otc" if ticker.endswith(".TWO") else "tse"
        return f"{prefix}_{stock_id}.tw"

    def _mis_convert(self, raw: dict) -> dict | None:
        if not isinstance(raw, dict):
            return None
        ch = str(raw.get("ch") or "")
        code = str(raw.get("c") or "").strip()
        if "t00" in ch:
            stock_id = "001"
        elif "o00" in ch:
            stock_id = "101"
        else:
            stock_id = code
        if not stock_id:
            return None

        prev = self._num(raw.get("y"))
        # TWSE MIS documents `z` as the latest matched price and `y` as the
        # previous close.  Unknown/undocumented fields (for example `pz`) and
        # book midpoints must not be promoted to an actual trade price. If `z`
        # is absent, the row is retained for metadata but cannot drive an
        # execution decision.
        close = self._num(raw.get("z"))

        # Missing MIS matched-price `z` means *unknown*, not an unchanged price.
        # In particular an active stock can report cumulative volume while the
        # public quote endpoint temporarily returns `z="-"`. Never manufacture
        # a flat +0.00% reading or promote order-book / previous prices to trades.
        change_rate = ((close / prev) - 1.0) * 100.0 if np.isfinite(close) and np.isfinite(prev) and prev > 0 else None
        volume_lots = self._num(raw.get("v"), 0.0)
        # MIS cumulative volume is published in round lots for equities.  The
        # historical cache uses shares, so normalize to shares for comparisons.
        total_volume = volume_lots * 1000.0 if stock_id not in {"001", "101"} else volume_lots
        amount = close * total_volume if np.isfinite(close) and total_volume > 0 and stock_id not in {"001", "101"} else np.nan

        d = str(raw.get("d") or "")
        t = str(raw.get("t") or "")
        date_text = ""
        if len(d) == 8 and d.isdigit():
            date_text = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            if t and t != "-":
                date_text += f" {t}"

        return {
            "stock_id": stock_id,
            "date": date_text,
            "quote_channel": ch,
            "market": "TPEx" if str(raw.get("ex") or "").lower() == "otc" or ch.startswith("otc_") else "TWSE",
            "close": None if not np.isfinite(close) else float(close),
            "previous_close": None if not np.isfinite(prev) else float(prev),
            "limit_up": self._num(raw.get("u")),
            "limit_down": self._num(raw.get("w")),
            "open": self._num(raw.get("o")),
            "high": self._num(raw.get("h")),
            "low": self._num(raw.get("l")),
            "average_price": np.nan,  # MIS does not expose VWAP in this payload.
            "change_rate": round(float(change_rate), 4) if change_rate is not None else None,
            "total_volume": float(total_volume),
            "volume_reported": raw.get("v") not in (None, "", "-"),
            "mis_quote_has_trade": bool(np.isfinite(close) and close > 0),
            "total_amount": None if not np.isfinite(amount) else float(amount),
            "buy_price": self._first_book_value(raw.get("b")),
            "sell_price": self._first_book_value(raw.get("a")),
            "buy_volume": self._first_book_value(raw.get("g")),
            "sell_volume": self._first_book_value(raw.get("f")),
            "source": "TWSE/TPEx MIS",
            "price_origin": "本次官方成交" if np.isfinite(close) and close > 0 else "官方本次未回傳成交價",
            # Safe diagnostic facts; `tlong` is not substituted for trade time.
            "matched_price_field": "數字" if np.isfinite(close) and close > 0 else "缺值",
            "raw_trade_clock": t,
            "raw_update_epoch_ms": str(raw.get("tlong") or "")[:16],

        }

    def _mis_snapshots(self, tickers: list[str], batch_size: int = 45) -> tuple[pd.DataFrame, RealtimeStatus]:
        """Timestamp-safe MIS quote collector with bounded recovery.

        Distinguishes: current returned trade, observed earlier trade,
        today's volume but missing trade price, channel not returned, and
        endpoint failure. Only real `z` with original per-symbol trade time
        may become an actionable quote. A missing `z` never becomes zero or
        yesterday's close, best bid/ask, `pz`, or chart high/low.
        """
        now = _taipei_now()
        self._bootstrap_mis()
        # Bounded network budget avoids freezing Streamlit's phone UI when the
        # external MIS service is slow, inaccessible or rate-limited.
        deadline = time.monotonic() + 14.0
        tickers = list(dict.fromkeys(str(t).upper() for t in tickers if _stock_id(t)))
        expected_market = {_stock_id(t): ('TPEx' if t.endswith('.TWO') else 'TWSE')
                           for t in tickers}
        requested_ids = set(expected_market)
        requested_n = len(requested_ids)
        channels = list(dict.fromkeys([self._mis_channel(t) for t in tickers] +
                                      ['tse_t00.tw', 'otc_o00.tw']))
        rows_by_id: dict[str, dict] = {}
        # A late HTTP reply can contain an earlier price and smaller volume.
        # Keep the highest cumulative volume seen across ALL reply samples.
        highest_observed_volume: dict[str, float] = {}
        errors: list[str] = []
        query_succeeded = False
        requests_made = 0
        response_rows = 0
        response_dates: list[str] = []
        response_server_times: list[str] = []

        def fetch(channels_to_fetch: list[str], *, chunk: int | None = None) -> None:
            nonlocal query_succeeded, requests_made, response_rows
            slice_size = max(1, int(chunk if chunk is not None else batch_size))
            for i in range(0, len(channels_to_fetch), slice_size):
                batch = channels_to_fetch[i:i + slice_size]
                if time.monotonic() >= deadline:
                    errors.append('MIS_FETCH_BUDGET_EXHAUSTED')
                    break
                # This endpoint refreshes on its own cadence. Do not fire
                # many near-identical requests in the same server window;
                # it is more useful to sample again 20s later.
                if requests_made >= 4 and requested_n <= 20:
                    break
                params = {
                    'ex_ch': '|'.join(batch), 'json': '1', 'delay': '0',
                    '_': str(int(_taipei_now().timestamp() * 1000)),
                }
                requests_made += 1
                try:
                    remaining = max(0.5, deadline - time.monotonic())
                    r = self.session.get(MIS_URL, params=params, timeout=(2.5, min(5.0, remaining)))
                    r.raise_for_status()
                    js = r.json()
                    if not isinstance(js, dict) or str(js.get('rtcode', '0000')) not in {'0000', '0'}:
                        errors.append('MIS_BAD_RESPONSE_OR_RTCODE')
                        continue
                    raw_array = js.get('msgArray')
                    if not isinstance(raw_array, list):
                        errors.append('MIS_MSGARRAY_MISSING')
                        continue
                    query_succeeded = True
                    response_rows += len(raw_array)
                    qtime = js.get('queryTime') or {}
                    if isinstance(qtime, dict):
                        response_dates.append(str(qtime.get('sysDate') or '')[:8])
                        response_server_times.append(str(qtime.get('sysTime') or '')[:12])
                    for raw in raw_array:
                        item = self._mis_convert(raw)
                        if not item:
                            continue
                        stock_id = str(item['stock_id'])
                        if stock_id not in requested_ids and stock_id not in {'001', '101'}:
                            continue
                        # Never accept an OTC response under a TWSE stock code
                        # or vice versa, even in an inconsistent batch reply.
                        if stock_id in requested_ids and item.get('market') != expected_market[stock_id]:
                            errors.append('MIS_MARKET_CHANNEL_MISMATCH')
                            continue
                        if stock_id in requested_ids and item.get('volume_reported'):
                            sampled_volume = _finite(item.get('total_volume'), np.nan)
                            if np.isfinite(sampled_volume):
                                highest_observed_volume[stock_id] = max(
                                    highest_observed_volume.get(stock_id, 0.0), sampled_volume)
                        prior = rows_by_id.get(stock_id)
                        current_ts = _official_trade_timestamp(item)
                        prior_ts = _official_trade_timestamp(prior) if prior else None
                        priced = bool(_finite(item.get('close'), np.nan) > 0)
                        prior_priced = bool(prior and _finite(prior.get('close'), np.nan) > 0)
                        # Retry may return an older row. Preserve the most
                        # recent, timestamped matched price, not the last HTTP
                        # response, and retain metadata separately.
                        if prior_priced and not priced:
                            prior['latest_reported_volume'] = item.get('total_volume')
                            prior['latest_missing_trade_price'] = True
                            prior['matched_price_field'] = '本輪缺值，保留先前成交'
                            old_volume = _finite(prior.get('total_volume'), np.nan)
                            new_volume = _finite(item.get('total_volume'), np.nan)
                            if np.isfinite(old_volume) and np.isfinite(new_volume) and new_volume > old_volume:
                                prior['volume_increased_after_last_trade'] = True
                            if not item.get('volume_reported'):
                                prior['volume_uncertain_after_last_trade'] = True
                            continue
                        if prior_priced and priced and prior_ts and (current_ts is None or current_ts < prior_ts):
                            continue
                        rows_by_id[stock_id] = item
                        if priced and stock_id in requested_ids:
                            _remember_official_trade(item, _taipei_now())
                except requests.HTTPError as exc:
                    code = getattr(exc.response, 'status_code', '')
                    errors.append(f'MIS_HTTP_{code or "ERROR"}')
                except requests.Timeout:
                    errors.append('MIS_TIMEOUT')
                except requests.ConnectionError:
                    errors.append('MIS_NETWORK_UNREACHABLE')
                except (ValueError, requests.RequestException):
                    errors.append('MIS_INVALID_JSON_OR_TRANSPORT')
                except Exception as exc:
                    errors.append(f'MIS_{type(exc).__name__}')

        fetch(channels)
        # Bounded fallback: first retry every missing requested stock in small
        # groups; then rotate two singles so later ranks are not starved.
        retry_channels: list[str] = []
        individual_retries = 0
        if query_succeeded and 0 < requested_n <= 20:
            need_retry = [t for t in tickers if _stock_id(t) not in rows_by_id or
                          not (_finite(rows_by_id[_stock_id(t)].get('close'), np.nan) > 0)]
            retry_channels = list(dict.fromkeys(self._mis_channel(t) for t in need_retry))[:16]
            if retry_channels:
                fetch(retry_channels, chunk=4)
            still_missing = [t for t in need_retry if _stock_id(t) not in rows_by_id or
                             not (_finite(rows_by_id[_stock_id(t)].get('close'), np.nan) > 0)]
            if still_missing:
                offset = int(now.timestamp() // 60) % len(still_missing)
                rotated = still_missing[offset:] + still_missing[:offset]
                for ticker in rotated[:2]:
                    fetch([self._mis_channel(ticker)], chunk=1)
                    individual_retries += 1

        # Prefer actual same-day matched prices observed by this server during
        # earlier 60-second fragment polls over a later MIS `z='-'` placeholder.
        restored = 0
        for stock_id, market in expected_market.items():
            memorized = _recalled_official_trade(stock_id, market, _taipei_now())
            if not memorized:
                continue
            old = rows_by_id.get(stock_id)
            if old is not None and _finite(old.get('close'), np.nan) > 0:
                old_ts = _official_trade_timestamp(old)
                mem_ts = _official_trade_timestamp(memorized)
                if old_ts is not None and mem_ts is not None and old_ts >= mem_ts:
                    continue
            if old:
                memorized['latest_reported_volume'] = old.get('total_volume')
                memorized['latest_missing_trade_price'] = not (_finite(old.get('close'), np.nan) > 0)
                if memorized['latest_missing_trade_price']:
                    memorized['matched_price_field'] = '本輪缺值，保留先前成交'
                remembered_volume = _finite(memorized.get('total_volume'), np.nan)
                reported_volume = _finite(old.get('total_volume'), np.nan)
                if np.isfinite(reported_volume) and np.isfinite(remembered_volume) and reported_volume > remembered_volume:
                    memorized['volume_increased_after_last_trade'] = True
                if not old.get('volume_reported'):
                    memorized['volume_uncertain_after_last_trade'] = True
            if not old:
                memorized['matched_price_field'] = '本輪未回傳，保留先前成交'
            rows_by_id[stock_id] = memorized
            restored += 1

        # Responses can arrive out of order across retries: a stale price
        # must never regain BUY eligibility after a newer z-less, higher-
        # volume sample was observed earlier in this same polling cycle.
        for stock_id, item in rows_by_id.items():
            if stock_id not in requested_ids or not (_finite(item.get('close'), np.nan) > 0):
                continue
            highest = highest_observed_volume.get(stock_id)
            priced_volume = _finite(item.get('total_volume'), np.nan)
            if highest is not None and np.isfinite(priced_volume) and highest > priced_volume:
                item['volume_increased_after_last_trade'] = True
                item['latest_reported_volume'] = highest

        checked_at = _taipei_now()
        date_text = checked_at.date().isoformat()
        df = pd.DataFrame(rows_by_id.values()) if rows_by_id else pd.DataFrame()
        quote_dates = pd.to_datetime(df.get('date'), errors='coerce') if not df.empty else pd.Series(dtype='datetime64[ns]')
        quote_date = str(quote_dates.max().date()) if len(quote_dates) and quote_dates.notna().any() else ''
        diagnostics = [dict(v) for k, v in rows_by_id.items() if k in requested_ids] if requested_n <= 40 else []
        received = sum(1 for k in requested_ids if k in rows_by_id)
        priced = sum(1 for k in requested_ids if k in rows_by_id and
                     _finite(rows_by_id[k].get('close'), np.nan) > 0)
        no_price = received - priced
        missing = requested_n - received
        missing_matched = sum(1 for k in requested_ids if k in rows_by_id
                              and not (_finite(rows_by_id[k].get('close'), np.nan) > 0)
                              and _finite(rows_by_id[k].get('total_volume'), 0.0) > 0
                              and str(rows_by_id[k].get('date') or '')[:10] == date_text)
        counts = {'requested': requested_n, 'received': received,
                  'priced': priced, 'missing': missing,
                  'no_trade_price': no_price, 'traded_without_last': missing_matched,
                  'other_without_last': max(0, no_price - missing_matched),
                  'retry_group_count': len(retry_channels),
                  'retry_individual_count': individual_retries,
                  'restored_prior_trade': restored,
                  'newer_volume_without_price': sum(1 for r in diagnostics
                          if r.get('volume_increased_after_last_trade') or r.get('volume_uncertain_after_last_trade')),
                  'http_requests': requests_made,
                  'raw_rows_seen': response_rows, 'http_errors': len(errors),
                  'stale': 0, 'fresh': 0}
        request_audit = {'attempts': requests_made, 'rows_seen': response_rows,
                         'errors': errors[:10], 'source_dates': sorted(set(response_dates)),
                         'server_times': response_server_times[-3:],
                         'last_collected_at': checked_at.isoformat(timespec='seconds')}

        def result(available: bool, why: str, *, verified: bool = False,
                   frame: pd.DataFrame | None = None, fresh_n: int = 0):
            counts['fresh'] = int(fresh_n)
            return (frame if frame is not None else pd.DataFrame(), RealtimeStatus(
                available, why, quote_date=quote_date, source='TWSE/TPEx MIS',
                market_open=market_is_open(checked_at), fetched_at=checked_at.isoformat(timespec='seconds'),
                verified=verified, verification='official_mis' if verified else 'official_mis_unavailable',
                requested_count=requested_n, executable_count=int(fresh_n),
                diagnostic_counts=dict(counts), quote_diagnostics=diagnostics,
                request_diagnostics=dict(request_audit)))

        if not query_succeeded:
            return result(False, errors[0] if errors else 'MIS_ENDPOINT_UNAVAILABLE')
        if df.empty:
            return result(False, 'MIS_RETURNED_NO_STOCK_ROWS')
        if 'stock_id' not in df.columns:
            return result(False, 'MIS_MISSING_STOCK_ID')

        ids = df['stock_id'].astype(str)
        prices = pd.to_numeric(df.get('close'), errors='coerce')
        matched = ids.isin(requested_ids) & prices.notna() & np.isfinite(prices) & (prices > 0)
        executable = matched.copy()
        # If the exchange reports *newer cumulative volume* but omits its
        # matching price, a previous price is no longer the latest trade,
        # regardless of the 180s age test. Keep read-only evidence, not BUY.
        advanced = df.get('volume_increased_after_last_trade')
        if advanced is not None:
            executable &= ~advanced.fillna(False).astype(bool)
        uncertain = df.get('volume_uncertain_after_last_trade')
        if uncertain is not None:
            executable &= ~uncertain.fillna(False).astype(bool)
        if market_is_open(checked_at):
            times = pd.to_datetime(df.get('date'), errors='coerce')
            clock = pd.Timestamp(checked_at.replace(tzinfo=None))
            age = (clock - times).dt.total_seconds()
            fresh = times.notna() & (times.dt.date == checked_at.date()) & (age <= 180.0) & (age >= -90.0)
            executable &= fresh
            counts['stale'] = int((matched & ~fresh).sum())
        executable_n = len(set(ids[executable].tolist()))
        index_rows = ids.isin({'001', '101'})
        exec_frame = df[executable | index_rows].copy()
        if requested_n and executable_n == 0:
            reason = ('MIS_RECENT_PRICE_NOT_LATEST' if priced and (advanced is not None or uncertain is not None) and market_is_open(checked_at) and bool(matched.any()) else
                      'MIS_TRADES_TOO_OLD' if priced and market_is_open(checked_at) else
                      'MIS_MATCHED_PRICE_MISSING' if no_price else
                      'MIS_NO_STOCK_ROWS' if missing else 'MIS_NO_EXECUTABLE_PRICE')
            return result(False, reason)
        reason = '' if executable_n >= requested_n else f'MIS_PARTIAL_{executable_n}_OF_{requested_n}'
        return result(True, reason, verified=True, frame=exec_frame, fresh_n=executable_n)

    def _finmind_snapshots(self, tickers: list[str], batch_size: int = 45) -> tuple[pd.DataFrame, RealtimeStatus]:
        if not self.token:
            return pd.DataFrame(), RealtimeStatus(False, "FINMIND_TOKEN_NOT_CONFIGURED", source="FinMind realtime snapshot", market_open=market_is_open())
        ids = list(dict.fromkeys([_stock_id(t) for t in tickers if _stock_id(t)]))
        ids_with_index = ids + ["001", "101"]
        rows: list[dict] = []
        try:
            for i in range(0, len(ids_with_index), batch_size):
                rows.extend(self._request_ids(ids_with_index[i:i + batch_size]))
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", "")
            return pd.DataFrame(), RealtimeStatus(False, f"HTTP_{status or 'ERROR'}", source="FinMind realtime snapshot", market_open=market_is_open())
        except Exception as exc:
            return pd.DataFrame(), RealtimeStatus(False, f"{type(exc).__name__}", source="FinMind realtime snapshot", market_open=market_is_open())
        if not rows:
            return pd.DataFrame(), RealtimeStatus(False, "NO_REALTIME_ROWS", source="FinMind realtime snapshot", market_open=market_is_open())
        df = pd.DataFrame(rows)
        if "stock_id" not in df.columns:
            return pd.DataFrame(), RealtimeStatus(False, "MISSING_STOCK_ID", source="FinMind realtime snapshot", market_open=market_is_open())
        df["stock_id"] = df["stock_id"].astype(str)
        ds = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.Series(dtype="datetime64[ns]")
        quote_date = str(ds.max().date()) if len(ds) and ds.notna().any() else ""
        now = _taipei_now()
        if market_is_open(now) and quote_date and quote_date != now.date().isoformat():
            return pd.DataFrame(), RealtimeStatus(False, f"STALE_REALTIME_{quote_date}", quote_date=quote_date, source="FinMind realtime snapshot", market_open=True, fetched_at=now.isoformat(timespec="seconds"))
        return df.drop_duplicates("stock_id", keep="last"), RealtimeStatus(
            True, quote_date=quote_date, source="FinMind realtime snapshot", market_open=market_is_open(now),
            fetched_at=now.isoformat(timespec="seconds"), verified=False, verification="provider_only"
        )

    @staticmethod
    def _fresh_enough_for_execution(df: pd.DataFrame, now: datetime.datetime | None = None, max_age_seconds: int = 180) -> bool:
        """Require a real timestamp, not merely today's date, before an unverified quote can be considered fresh.

        This does not make the provider authoritative; it is only a sanity check for diagnostics.
        The execution path still prefers official MIS whenever it is reachable.
        """
        if df is None or df.empty or "date" not in df.columns:
            return False
        now = now or _taipei_now()
        ds = pd.to_datetime(df["date"], errors="coerce")
        ds = ds.dropna()
        if ds.empty:
            return False
        ts = ds.max()
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize(datetime.timezone(datetime.timedelta(hours=8)))
        else:
            ts = ts.tz_convert(datetime.timezone(datetime.timedelta(hours=8)))
        age = (pd.Timestamp(now) - ts).total_seconds()
        return -30 <= age <= max_age_seconds

    def verified_snapshots(self, tickers: Iterable[str], batch_size: int = 45) -> tuple[pd.DataFrame, RealtimeStatus]:
        """Safe official execution layer, plus separately labelled observations.

        A licensed FinMind snapshot may help the user *see* an indicative price,
        but is never silently promoted to an official matched execution tick.
        """
        tickers = list(dict.fromkeys(str(t) for t in tickers if _stock_id(t)))
        mis_df, mis_status = self._mis_snapshots(tickers, batch_size=batch_size)
        official_ids = (set(mis_df['stock_id'].astype(str)) - {'001', '101'}
                        if mis_df is not None and not mis_df.empty and 'stock_id' in mis_df.columns else set())
        outstanding = [t for t in tickers if _stock_id(t) not in official_ids]
        fm_df, fm_status = pd.DataFrame(), None
        if self.token and outstanding:
            fm_df, fm_status = self._finmind_snapshots(outstanding, batch_size=batch_size)
        auxiliary = []
        if fm_status and fm_status.available and fm_df is not None and not fm_df.empty:
            for item in fm_df.to_dict('records'):
                ident = str(item.get('stock_id') or '')
                if ident in {_stock_id(t) for t in outstanding}:
                    # Keep only UI diagnostic fields and let the display decide
                    # if a provider timestamp is usable as an observation.
                    auxiliary.append({key: item.get(key) for key in
                                      ('stock_id', 'date', 'close', 'previous_close', 'change_rate', 'total_volume')})
        mis_status.auxiliary_quotes = auxiliary
        if mis_status.available and mis_df is not None and not mis_df.empty:
            return mis_df, mis_status

        # Even a timestamped provider quote is observation-only. Do not return
        # it as the dataframe to the buy/rerank engine.
        mis_status.verified = False
        mis_status.verification = 'official_not_available'
        mis_status.available = False
        mis_status.executable_count = 0
        if auxiliary and not mis_status.reason:
            mis_status.reason = 'MIS_UNAVAILABLE_PROVIDER_OBSERVATION_ONLY'
        return pd.DataFrame(), mis_status

    def snapshots(self, tickers: Iterable[str], batch_size: int = 45) -> tuple[pd.DataFrame, RealtimeStatus]:
        tickers = list(dict.fromkeys([str(t) for t in tickers if _stock_id(t)]))
        # Prefer the richer FinMind snapshot if entitlement exists, but never let
        # a missing token/permission silently disable intraday ranking.
        if self.token:
            df, status = self._finmind_snapshots(tickers, batch_size=batch_size)
            if status.available and not df.empty:
                return df, status
        return self._mis_snapshots(tickers)

def _quote_features(row: pd.Series, market_rate_pp: float) -> dict:
    """Score current-session strength only.

    Important Architecture Freeze rule: a large one-day move or near-limit-up
    print is *not* penalized here.  Those are execution/entry risks handled by
    policy_engine.evaluate_entry_timing().
    """
    close = _finite(row.get("close"))
    open_ = _finite(row.get("open"))
    high = _finite(row.get("high"))
    low = _finite(row.get("low"))
    avg = _finite(row.get("average_price"))
    change_pp = _finite(row.get("change_rate"), 0.0)
    vol_ratio = _finite(row.get("volume_ratio"), np.nan)
    amount = _finite(row.get("total_amount"), _finite(row.get("amount"), np.nan))
    buy_p = _finite(row.get("buy_price"))
    sell_p = _finite(row.get("sell_price"))
    buy_v = _finite(row.get("buy_volume"))
    sell_v = _finite(row.get("sell_volume"))

    relative_pp = change_pp - market_rate_pp
    vwap_gap_pp = ((close / avg) - 1.0) * 100.0 if np.isfinite(close) and np.isfinite(avg) and avg > 0 else np.nan
    prior = _finite(row.get("previous_close"), np.nan)
    gap_open_pct = ((open_ / prior) - 1.0) * 100.0 if np.isfinite(prior) and prior > 0 and np.isfinite(open_) else np.nan
    open_move_pp = ((close / open_) - 1.0) * 100.0 if np.isfinite(close) and np.isfinite(open_) and open_ > 0 else np.nan
    day_pos = (close - low) / (high - low) if all(np.isfinite(x) for x in [close, high, low]) and high > low else 0.5
    rebound_from_low_pp = ((close / low) - 1.0) * 100.0 if np.isfinite(close) and np.isfinite(low) and low > 0 else np.nan
    drawdown_from_high_pp = ((close / high) - 1.0) * 100.0 if np.isfinite(close) and np.isfinite(high) and high > 0 else np.nan
    mid = (buy_p + sell_p) / 2.0 if np.isfinite(buy_p) and np.isfinite(sell_p) and buy_p + sell_p > 0 else np.nan
    spread_pp = ((sell_p - buy_p) / mid) * 100.0 if np.isfinite(mid) and mid > 0 else np.nan
    bid_pressure = buy_v / (buy_v + sell_v) if np.isfinite(buy_v) and np.isfinite(sell_v) and buy_v + sell_v > 0 else 0.5

    ranges = INTRADAY_FACTOR_RANGES
    weights = INTRADAY_FACTOR_WEIGHTS
    rel_s = _clip01(relative_pp, *ranges["relative_market"]) * 100.0
    vwap_s = _clip01(vwap_gap_pp, *ranges["vwap"]) * 100.0
    vol_s = _clip01(vol_ratio, *ranges["volume"]) * 100.0 if np.isfinite(vol_ratio) else 50.0
    pos_s = _clip01(day_pos, *ranges["day_position"]) * 100.0
    open_s = _clip01(open_move_pp, *ranges["open_move"]) * 100.0
    bid_s = _clip01(bid_pressure, *ranges["bid_pressure"]) * 100.0
    turn_lo, turn_hi = ranges["turnover_twd"]
    if np.isfinite(amount) and amount > 0:
        turn_s = _clip01(np.log10(amount), np.log10(turn_lo), np.log10(turn_hi)) * 100.0
    else:
        turn_s = 45.0

    raw = (
        weights["relative_market"] * rel_s
        + weights["vwap"] * vwap_s
        + weights["volume"] * vol_s
        + weights["day_position"] * pos_s
        + weights["open_move"] * open_s
        + weights["bid_pressure"] * bid_s
        + weights["turnover"] * turn_s
    )
    # Wide spread is a tradability/data-quality issue, not a chase penalty.
    spread_penalty = max(0.0, (spread_pp - INTRADAY_SPREAD_FREE_PCT) * INTRADAY_SPREAD_PENALTY_PER_PCT) if np.isfinite(spread_pp) else 0.0
    score = float(np.clip(raw - spread_penalty, 0.0, 100.0))

    cfg = INTRADAY_STATE
    if (relative_pp <= cfg["relative_weak"]) or (
        np.isfinite(vwap_gap_pp) and vwap_gap_pp < cfg["vwap_weak"] and day_pos < cfg["day_position_weak"]
    ):
        state = "盤中轉弱"
    elif score >= cfg["strong_score"] and relative_pp >= cfg["strong_relative"] and (
        not np.isfinite(vol_ratio) or vol_ratio >= cfg["strong_volume"]
    ) and (not np.isfinite(vwap_gap_pp) or vwap_gap_pp >= 0):
        state = "盤中強勢"
    elif score >= cfg["stable_score"] and (not np.isfinite(vwap_gap_pp) or vwap_gap_pp >= cfg["stable_vwap"]):
        state = "走勢穩定"
    elif np.isfinite(vol_ratio) and vol_ratio >= cfg["volume_watch"] and score >= cfg["volume_watch_score"]:
        state = "量能升溫"
    else:
        state = "等待確認"

    reasons: list[str] = []
    if relative_pp >= cfg["reason_relative"]:
        reasons.append(f"比大盤強 {relative_pp:.1f} 個百分點")
    elif relative_pp <= -cfg["reason_relative"]:
        reasons.append(f"比大盤弱 {abs(relative_pp):.1f} 個百分點")
    if np.isfinite(vol_ratio) and vol_ratio >= cfg["reason_volume"]:
        reasons.append(f"量能約平常 {vol_ratio:.1f} 倍")
    if np.isfinite(vwap_gap_pp):
        if vwap_gap_pp >= cfg["reason_vwap_up"]:
            reasons.append("守在盤中均價上方")
        elif vwap_gap_pp <= cfg["reason_vwap_down"]:
            reasons.append("跌到盤中均價下方")
    if not reasons:
        reasons.append("盤中訊號尚未形成明顯優勢")

    return {
        "intraday_score": round(score, 1),
        "state": state,
        "change_rate_pct": round(change_pp, 2),
        "market_change_rate_pct": round(float(market_rate_pp), 2),
        "relative_market_pct_pt": round(float(relative_pp), 2),
        "vwap_gap_pct": None if not np.isfinite(vwap_gap_pp) else round(float(vwap_gap_pp), 2),
        "volume_ratio": None if not np.isfinite(vol_ratio) else round(float(vol_ratio), 2),
        "day_position": round(float(day_pos), 3),
        "rebound_from_low_pct": None if not np.isfinite(rebound_from_low_pp) else round(float(rebound_from_low_pp), 2),
        "drawdown_from_high_pct": None if not np.isfinite(drawdown_from_high_pp) else round(float(drawdown_from_high_pp), 2),
        "spread_pct": None if not np.isfinite(spread_pp) else round(float(spread_pp), 3),
        "bid_pressure": round(float(bid_pressure), 3),
        "price": None if not np.isfinite(close) else float(close),
        "previous_close": None if not np.isfinite(_finite(row.get("previous_close"))) else float(_finite(row.get("previous_close"))),
        "limit_up": None if not np.isfinite(_finite(row.get("limit_up"))) else float(_finite(row.get("limit_up"))),
        "limit_down": None if not np.isfinite(_finite(row.get("limit_down"))) else float(_finite(row.get("limit_down"))),
        "quote_source": str(row.get("source") or ""),
        "average_price": None if not np.isfinite(avg) else float(avg),
        "open": None if not np.isfinite(open_) else float(open_),
        "open_gap_pct": None if not np.isfinite(gap_open_pct) else round(float(gap_open_pct),2),
        "open_to_now_pct": None if not np.isfinite(open_move_pp) else round(float(open_move_pp),2),
        "high": None if not np.isfinite(high) else float(high),
        "low": None if not np.isfinite(low) else float(low),
        "total_amount": None if not np.isfinite(amount) else float(amount),
        "spread_penalty": round(float(spread_penalty), 1),
        "reasons": reasons[:2],
    }


def rerank_snapshot(
    snap: dict,
    horizon: str,
    realtime_df: pd.DataFrame,
    top_n: int = 10,
    now: datetime.datetime | None = None,
) -> list[dict]:
    """Re-order the already-vetted daily candidate pool using realtime strength."""
    if not snap or realtime_df is None or realtime_df.empty:
        return []
    horizon = horizon if horizon in CONFIG_HORIZONS else "short"
    now = now or _taipei_now()
    rows = {str(r["stock_id"]): r for _, r in realtime_df.iterrows()}
    twse = rows.get("001")
    otc = rows.get("101")
    out: list[dict] = []
    regime = ((snap.get("market") or {}).get("regime") or "UNKNOWN")
    w = intraday_blend_weight(horizon, now)

    for stock in snap.get("stocks", []):
        ticker = str(stock.get("ticker", ""))
        q = rows.get(_stock_id(ticker))
        if q is None:
            continue
        if market_is_open(now) and "date" in q:
            qdt = pd.to_datetime(q.get("date"), errors="coerce")
            if pd.notna(qdt) and str(qdt.date()) != now.date().isoformat():
                continue
        index_row = otc if ticker.endswith(".TWO") else twse
        market_rate = _finite(index_row.get("change_rate"), 0.0) if index_row is not None else 0.0
        q = q.copy()

        # Estimate full-session pace from cumulative volume.  This preserves the
        # complete candidate set while avoiding repeated historical recomputes.
        if not np.isfinite(_finite(q.get("volume_ratio"))):
            current_volume = _finite(q.get("total_volume"), np.nan)
            avg_volume = _finite(stock.get("avg_volume_20d"), np.nan)
            mins = max(0.0, (now.hour * 60 + now.minute + now.second / 60.0) - 9 * 60)
            session_fraction = float(np.clip(mins / 270.0, 0.10, 1.0))
            if np.isfinite(current_volume) and np.isfinite(avg_volume) and avg_volume > 0:
                q["volume_ratio"] = float(np.clip((current_volume / avg_volume) / session_fraction, 0.05, 6.0))

        intraday = _quote_features(q, market_rate)
        intraday["minutes_since_open"] = round(max(0.0,(now.hour*60+now.minute+now.second/60)-9*60),2)
        # A row can legitimately have today's date but no executable last/book
        # price. Never fall back to the completed-day close and call it live.
        if not np.isfinite(_finite(intraday.get("price"))):
            continue
        plan = stock.get("horizons", {}).get(horizon, {}).get("plan") or {}
        timing = evaluate_entry_timing(
            intraday.get("price"),
            plan,
            horizon,
            change_pct=intraday.get("change_rate_pct"),
            market_change_pct=intraday.get("market_change_rate_pct"),
            volume_ratio=intraday.get("volume_ratio"),
            relative_market_pct_pt=intraday.get("relative_market_pct_pt"),
            day_position=intraday.get("day_position"),
        )
        intraday["entry_timing"] = timing
        intraday["position_guidance"] = position_guidance(regime, horizon, timing)

        daily_score = _finite(stock.get("horizons", {}).get(horizon, {}).get("ranking_score"), np.nan)
        if not np.isfinite(daily_score):
            continue
        final_score = float(np.clip(daily_score * (1.0 - w) + intraday["intraday_score"] * w, 0.0, 100.0))
        item = dict(stock)
        item["intraday"] = intraday
        item["daily_ranking_score"] = round(float(daily_score), 1)
        item["live_ranking_score"] = round(final_score, 1)
        item["intraday_blend_weight"] = round(float(w), 3)
        out.append(item)

    out.sort(key=lambda x: x.get("live_ranking_score", -1), reverse=True)
    return out[: max(1, int(top_n))]


def candidate_tickers(
    snap: dict,
    per_horizon: int | None = None,
    max_total: int | None = None,
) -> list[str]:
    """Union of completed-day front ranks used as the stable daily core."""
    if not snap:
        return []
    per_horizon = int(per_horizon or LIVE_SCAN.get("per_horizon", 80))
    max_total = int(max_total or LIVE_SCAN.get("daily_core_max", LIVE_SCAN.get("max_total", 220)))
    stocks = [x for x in snap.get("stocks", []) if isinstance(x, dict)]
    chosen: list[str] = []
    for h in ["short", "mid", "long"]:
        ranked = sorted(
            stocks,
            key=lambda x: x.get("horizons", {}).get(h, {}).get("ranking_score", -1),
            reverse=True,
        )
        chosen.extend([str(x.get("ticker")) for x in ranked[:per_horizon] if x.get("ticker")])
    return list(dict.fromkeys(chosen))[:max_total]


def full_market_tickers(snap: dict, max_total: int | None = None) -> list[str]:
    """Return the liquid full-market quote universe for the one-minute scan.

    New snapshots persist ``intraday_universe`` from the completed-day liquidity
    pass. Older snapshots fall back to the fully evaluated stock list, so a
    deploy never fails merely because a saved snapshot predates V16.6.
    """
    if not snap:
        return []
    limit = int(max_total or LIVE_SCAN.get("full_market_quote_max", 1800))
    source = snap.get("intraday_universe") or snap.get("stocks") or []
    tickers = [str(x.get("ticker")) for x in source if isinstance(x, dict) and x.get("ticker")]
    tickers = list(dict.fromkeys(tickers))
    return tickers[:limit] if limit > 0 else tickers


def _intraday_meta(snap: dict) -> dict[str, dict]:
    rows = snap.get("intraday_universe") or []
    if not rows:
        rows = snap.get("stocks") or []
    return {str(x.get("ticker")): x for x in rows if isinstance(x, dict) and x.get("ticker")}


def _quote_rows(realtime_df: pd.DataFrame) -> dict[str, pd.Series]:
    if realtime_df is None or realtime_df.empty or "stock_id" not in realtime_df.columns:
        return {}
    return {str(r["stock_id"]): r for _, r in realtime_df.iterrows()}


def _estimated_volume_ratio(q: pd.Series, avg_volume: float | None, now: datetime.datetime) -> float | None:
    existing = _finite(q.get("volume_ratio"), np.nan)
    if np.isfinite(existing):
        return float(existing)
    current_volume = _finite(q.get("total_volume"), np.nan)
    avg = _finite(avg_volume, np.nan)
    if not np.isfinite(current_volume) or not np.isfinite(avg) or avg <= 0:
        return None
    mins = max(0.0, (now.hour * 60 + now.minute + now.second / 60.0) - 9 * 60)
    session_fraction = float(np.clip(mins / 270.0, 0.10, 1.0))
    return float(np.clip((current_volume / avg) / session_fraction, 0.05, 6.0))


def quick_rank_market(
    snap: dict,
    realtime_df: pd.DataFrame,
    now: datetime.datetime | None = None,
    top_n: int | None = None,
    global_context: dict | None = None,
) -> list[dict]:
    """Cheap full-market discovery pass using only current quotes + cached metadata.

    This stage never creates a buy recommendation. It only identifies names that
    deserve the heavier execution pass. The expensive historical model is not
    recomputed every minute.
    """
    if not snap or realtime_df is None or realtime_df.empty:
        return []
    now = now or _taipei_now()
    top_n = int(top_n or LIVE_SCAN.get("discovery_top", 140))
    rows = _quote_rows(realtime_df)
    twse = rows.get("001")
    otc = rows.get("101")
    meta = _intraday_meta(snap)
    out: list[dict] = []
    for ticker, m in meta.items():
        q = rows.get(_stock_id(ticker))
        if q is None:
            continue
        index_row = otc if ticker.endswith(".TWO") else twse
        market_rate = _finite(index_row.get("change_rate"), 0.0) if index_row is not None else 0.0
        q = q.copy()
        vr = _estimated_volume_ratio(q, m.get("avg_volume_20d"), now)
        if vr is not None:
            q["volume_ratio"] = vr
        feat = _quote_features(q, market_rate)
        if not np.isfinite(_finite(feat.get("price"))):
            continue
        pre = _finite(m.get("pre_score"), 0.0)
        # Live tape must dominate ordinary discovery. A small completed-day prior
        # keeps obviously weak/noisy prints from monopolizing the mover list.
        pre_norm = float(np.clip((pre + 25.0) / 125.0, 0.0, 1.0) * 100.0)
        momentum_discovery = 0.90 * float(feat.get("intraday_score") or 0.0) + 0.10 * pre_norm

        # V16.8: during a broad selloff, a stock can be interesting precisely
        # because it was sold down and then absorbs the pressure. This discovery
        # score only promotes the name into the heavier evaluation pool; it does
        # NOT itself create a buy signal.
        pcfg = PANIC_REVERSAL_POLICY
        shock = _finite((global_context or {}).get("shock_score"), np.nan)
        market_selloff = float(market_rate) <= float(pcfg["taiwan_selloff_pct"])
        panic_active = (np.isfinite(shock) and shock >= float(pcfg["min_global_shock"])) or market_selloff
        panic_discovery = -1.0
        chg = _finite(feat.get("change_rate_pct"), np.nan)
        rel = _finite(feat.get("relative_market_pct_pt"), np.nan)
        rebound = _finite(feat.get("rebound_from_low_pct"), np.nan)
        dpos = _finite(feat.get("day_position"), np.nan)
        if panic_active and np.isfinite(chg) and float(pcfg["min_change_pct"]) <= chg <= float(pcfg["max_change_pct"]):
            rel_s = _clip01(rel, 0.0, 3.0) * 100.0 if np.isfinite(rel) else 0.0
            rebound_s = _clip01(rebound, 0.4, 3.0) * 100.0 if np.isfinite(rebound) else 0.0
            pos_s = _clip01(dpos, 0.25, 0.72) * 100.0 if np.isfinite(dpos) else 0.0
            panic_discovery = 0.40 * rel_s + 0.30 * rebound_s + 0.15 * pos_s + 0.15 * pre_norm

        discovery = max(momentum_discovery, panic_discovery)
        mode = "PANIC_RESILIENCE" if panic_discovery > momentum_discovery else "MOMENTUM"
        out.append({
            "ticker": ticker,
            "name": str(m.get("name") or ticker),
            "industry": str(m.get("industry") or ""),
            "discovery_score": round(float(discovery), 1),
            "discovery_mode": mode,
            "panic_discovery_score": None if panic_discovery < 0 else round(float(panic_discovery), 1),
            "intraday": feat,
        })
    out.sort(key=lambda x: x.get("discovery_score", -1), reverse=True)
    return out[:max(1, top_n)]


def dynamic_candidate_pool(
    snap: dict,
    realtime_df: pd.DataFrame,
    now: datetime.datetime | None = None,
    global_context: dict | None = None,
) -> dict:
    """Build the one-minute detail pool from daily core + new live leaders.

    The daily core is always preserved. Discovery names are appended only if the
    completed-day model already evaluated them; names outside that model are
    surfaced as discovery-only observations rather than being promoted to a buy
    recommendation without historical evidence.
    """
    if not snap:
        return {"detail_tickers": [], "discovery": [], "stats": {}}
    now = now or _taipei_now()
    core = candidate_tickers(
        snap,
        per_horizon=int(LIVE_SCAN.get("per_horizon", 80)),
        max_total=int(LIVE_SCAN.get("daily_core_max", 220)),
    )
    discovery = quick_rank_market(snap, realtime_df, now=now, top_n=int(LIVE_SCAN.get("discovery_top", 140)), global_context=global_context)
    evaluated = {str(x.get("ticker")) for x in (snap.get("stocks") or []) if isinstance(x, dict) and x.get("ticker")}
    detail_max = int(LIVE_SCAN.get("detail_max", 300))
    detail = list(core)
    novel_evaluated: list[str] = []
    discovery_only: list[str] = []
    for row in discovery:
        ticker = str(row.get("ticker") or "")
        if not ticker:
            continue
        if ticker not in evaluated:
            discovery_only.append(ticker)
            continue
        if ticker in detail:
            continue
        if len(detail) >= detail_max:
            break
        detail.append(ticker)
        novel_evaluated.append(ticker)
    quote_map = _quote_rows(realtime_df)
    quoted_stocks = sum(
        1 for t in full_market_tickers(snap)
        if _stock_id(t) in quote_map and np.isfinite(_finite(quote_map[_stock_id(t)].get("close")))
    )
    stats = {
        "full_universe_n": len(full_market_tickers(snap)),
        "quoted_n": quoted_stocks,
        "daily_core_n": len(core),
        "discovery_n": len(discovery),
        "panic_discovery_n": sum(1 for x in discovery if str(x.get("discovery_mode")) == "PANIC_RESILIENCE"),
        "novel_detail_n": len(novel_evaluated),
        "discovery_only_n": len(discovery_only),
        "detail_n": len(detail),
    }
    return {
        "detail_tickers": detail,
        "daily_core_tickers": core,
        "novel_detail_tickers": novel_evaluated,
        "discovery_only_tickers": discovery_only,
        "discovery": discovery,
        "stats": stats,
    }


def filter_realtime_df(realtime_df: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    """Filter a full-market quote frame to a detail pool while keeping indices."""
    if realtime_df is None or realtime_df.empty or "stock_id" not in realtime_df.columns:
        return pd.DataFrame()
    ids = {_stock_id(t) for t in tickers if _stock_id(t)} | {"001", "101"}
    return realtime_df[realtime_df["stock_id"].astype(str).isin(ids)].copy()

