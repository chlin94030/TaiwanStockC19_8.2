"""V18.18 independent stock-doctor quote overlay (read-only unless verified).

The completed-session research stock is never mutated. On each connected UI
fragment tick the app may ask for a newly verified MIS print, then use this
pure adapter to recompute the buying/holding evaluation from that print.
Vendor minute bars are solely informational, NEVER an executable quote.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import re
from urllib.parse import quote

import pandas as pd
import requests

from intraday_engine import rerank_snapshot
from live_session_v1810 import quote_freshness, TAIPEI
from trading_calendar import INTRADAY, session_reference_policy

YAHOO_CHART = 'https://query1.finance.yahoo.com/v8/finance/chart/'


def _num(value):
    try:
        n = float(value)
        return n if math.isfinite(n) and n > 0 else None
    except (ValueError, TypeError):
        return None


def fetch_yahoo_minute_reference(ticker: str, now: dt.datetime | None = None) -> dict:
    """Read-only Yahoo minute-bar observation, with source/time/price checks.

    A chart bar may be delayed; never treat a chart meta price, daily close,
    previousClose, bid or ask as a matched live execution trade.
    """
    now = now or dt.datetime.now(TAIPEI)
    now = now.replace(tzinfo=TAIPEI) if now.tzinfo is None else now.astimezone(TAIPEI)
    code = str(ticker or '').strip().upper()
    if not re.fullmatch(r'[A-Z0-9]{3,7}\.(?:TW|TWO)', code):
        return {'unavailable_reason': 'INVALID_TICKER'}
    if os.getenv('ALPHA_RADAR_DISABLE_YAHOO_INTRADAY', '0') == '1':
        return {'unavailable_reason': 'VENDOR_DISABLED'}
    try:
        res = requests.get(
            YAHOO_CHART + quote(code, safe='.'),
            params={'interval': '1m', 'range': '1d', 'includePrePost': 'false'},
            timeout=(2.0, 3.0),
            headers={'User-Agent': 'Mozilla/5.0 (compatible; AlphaRadar/18.19)', 'Accept': 'application/json'},
        )
        res.raise_for_status()
        payload = (res.json().get('chart') or {}).get('result') or []
        if not payload:
            return {'unavailable_reason': 'EMPTY_VENDOR_RESULT'}
        item = payload[0]
        meta = item.get('meta') or {}
        # Provider chart can unexpectedly redirect/map to another symbol.
        received = str(meta.get('symbol') or '').strip().upper()
        if received and received != code:
            return {'unavailable_reason': 'VENDOR_SYMBOL_MISMATCH'}
        stamps = item.get('timestamp') or []
        q = ((item.get('indicators') or {}).get('quote') or [{}])[0]
        closes = q.get('close') or []
        prev = _num(meta.get('chartPreviousClose'))
        for epoch, close in reversed(list(zip(stamps, closes))):
            value = _num(close)
            if value is None:
                continue
            stamp = dt.datetime.fromtimestamp(float(epoch), tz=dt.timezone.utc).astimezone(TAIPEI)
            delta = (now - stamp).total_seconds()
            if stamp.date() != now.date() or delta < -90 or delta > 6 * 3600:
                continue
            return {
                'ticker': code, 'price': value,
                'timestamp': stamp.isoformat(timespec='seconds'),
                'source': 'Yahoo Finance 1分鐘K（第三方、可能延遲）',
                'previous_close': prev,
                'age_seconds': max(0, int(delta)),
                'reference_only': True,
            }
        return {'unavailable_reason': 'NO_VALID_TODAY_MINUTE_BAR'}
    except requests.HTTPError as exc:
        status = getattr(exc.response, 'status_code', None)
        return {'unavailable_reason': f'HTTP_{status}' if status else 'HTTP_ERROR'}
    except requests.Timeout:
        return {'unavailable_reason': 'VENDOR_TIMEOUT'}
    except requests.ConnectionError:
        return {'unavailable_reason': 'VENDOR_CONNECTION_ERROR'}
    except (TypeError, ValueError, OverflowError, KeyError, IndexError):
        return {'unavailable_reason': 'VENDOR_FORMAT_INVALID'}
    except requests.RequestException:
        return {'unavailable_reason': 'VENDOR_REQUEST_ERROR'}


def finmind_auxiliary_reference(status, ticker: str, now: dt.datetime | None = None) -> dict:
    """Use an already-fetched FinMind quote only as a timestamped observation.

    This does not issue another network request. Some provider snapshots contain
    a date-only record; those cannot be called today's minute quote.
    """
    now = now or dt.datetime.now(TAIPEI)
    now = now.replace(tzinfo=TAIPEI) if now.tzinfo is None else now.astimezone(TAIPEI)
    requested = str(ticker or '').split('.')[0]
    expected_market = 'TPEx' if str(ticker).upper().endswith('.TWO') else 'TWSE'
    for item in (getattr(status, 'auxiliary_quotes', None) or []):
        if not isinstance(item, dict) or str(item.get('stock_id') or '') != requested:
            continue
        supplied_market = str(item.get('market') or '')
        if supplied_market and supplied_market != expected_market:
            continue
        stamp_raw = str(item.get('date') or '')
        if not stamp_raw or ('T' not in stamp_raw and ' ' not in stamp_raw):
            continue
        try:
            stamp = dt.datetime.fromisoformat(stamp_raw.replace('Z','+00:00'))
            stamp = stamp.replace(tzinfo=TAIPEI) if stamp.tzinfo is None else stamp.astimezone(TAIPEI)
            diff = (now - stamp).total_seconds()
            price = _num(item.get('close'))
            if price is None or stamp.date() != now.date() or not (-90 <= diff <= 6*3600):
                continue
            return {'ticker':str(ticker).upper(), 'price':price, 'timestamp':stamp.isoformat(timespec='seconds'),
                    'source':'FinMind 行情快照（第三方、僅供觀察）',
                    'age_seconds':max(0,int(diff)), 'reference_only':True}
        except (ValueError,TypeError,OverflowError):
            continue
    return {}


_REASON_ZH = {
    'MIS_ENDPOINT_UNAVAILABLE': '官方行情連線無法使用',
    'MIS_MATCHED_PRICE_MISSING': '官方未回傳最新成交價',
    'MIS_NO_EXECUTABLE_PRICE': '目前無可核實即時成交價',
    'MIS_TRADES_TOO_OLD': '已取得的成交價超過180秒',
    'MIS_RECENT_PRICE_NOT_LATEST': '成交量有增加，但較新的成交價缺漏',
    'MIS_NO_STOCK_ROWS': '官方未回傳這檔股票',
    'MIS_RETURNED_NO_STOCK_ROWS': '本次查詢沒有股票報價',
    'NO_EXECUTABLE_PRICE_FOR_TICKER': '這檔股票沒有符合時效的成交價',
}


def _matching_row(frame: pd.DataFrame | None, stock_id: str, market: str) -> dict | None:
    if frame is None or frame.empty or 'stock_id' not in frame.columns:
        return None
    for row in frame.to_dict('records'):
        if str(row.get('stock_id') or '') != stock_id:
            continue
        market_row = str(row.get('market') or '')
        if market_row and market_row != market:
            continue
        return row
    return None


def resolve_doctor_live(
    stock: dict, snap: dict, horizon: str, now: dt.datetime,
    frame: pd.DataFrame | None, status, *, official_eod_approved: bool,
    vendor_reference: dict | None = None,
) -> dict:
    """Overlay exactly one verified ticker, keeping vendor/missing prices read-only.

    Status.verified is batch-level; also require this very ticker's official
    row, market channel, date, trade time and 180-second freshness to match.
    """
    now = now.replace(tzinfo=TAIPEI) if now.tzinfo is None else now.astimezone(TAIPEI)
    ticker = str(stock.get('ticker') or '').strip().upper()
    stock_id = ticker.split('.')[0]
    market = 'TPEx' if ticker.endswith('.TWO') else 'TWSE'
    phase = session_reference_policy(now)['phase']
    status_reason = str(getattr(status, 'reason', '') or 'MIS_ENDPOINT_UNAVAILABLE')
    checking_time = str(getattr(status, 'fetched_at', '') or '')
    verified_source = bool(status and getattr(status, 'available', False) and getattr(status, 'verified', False))
    result = {
        'stock': stock, 'phase': phase, 'official_quote_verified': False,
        'verified_for_buy': False, 'reference_only': True, 'price': None,
        'quote_time': '', 'price_source': '', 'price_label': '今日成交價未取得',
        'checked_at': checking_time, 'reason': _REASON_ZH.get(status_reason, status_reason),
        'age_seconds': None, 'change_pct': None, 'quote_observation': None,
        'diagnostic_counts': dict(getattr(status, 'diagnostic_counts', None) or {}),
        'request_diagnostics': dict(getattr(status, 'request_diagnostics', None) or {}),
    }
    if phase != INTRADAY:
        result['reason'] = '目前非台股盤中時段，使用最近完整交易日基準'
        return result

    row = _matching_row(frame, stock_id, market)
    if row is None:
        for diag in (getattr(status, 'quote_diagnostics', None) or []):
            if str(diag.get('stock_id') or '') == stock_id and (not diag.get('market') or diag.get('market') == market):
                row = dict(diag)
                break
    quote_check = quote_freshness(row, now, official_verified=verified_source) if row else None
    if row and quote_check:
        price = _num(quote_check.get('price'))
        stamp = str(row.get('date') or '')
        # Only display this as TODAY's observed official print when the real
        # trade date is known. Last-session quotes must not be Today's price.
        if price is not None and stamp[:10] == now.date().isoformat() and (' ' in stamp or 'T' in stamp):
            result.update({
                'price': price, 'quote_time': str(quote_check.get('quote_time') or stamp),
                'price_source': '證交所 MIS 成交（官方）',
                'price_label': ('官方即時成交價' if quote_check.get('can_execute') else '今日較早官方成交價（非即時）'),
                'age_seconds': quote_check.get('age_seconds'), 'reference_only': True,
                'change_pct': row.get('change_rate'),
            })
        result['quote_observation'] = quote_check
        result['reason'] = str(quote_check.get('label') or result['reason'])

    if row and quote_check and quote_check.get('can_execute') and verified_source:
        result['official_quote_verified'] = True
        result['price_label'] = '官方已核實成交價'
        if official_eod_approved:
            # A verified quote may move the entry model ONLY if the underlying
            # daily model is itself approved. Use original rank/forecast.
            mini = {'stocks': [stock], 'market': (snap.get('market') or {}),
                    'snapshot_id': str(snap.get('snapshot_id') or 'doctor')}
            safe_row = pd.DataFrame([row])
            live = rerank_snapshot(mini, horizon, safe_row, top_n=1, now=now)
            if live and str(live[0].get('ticker') or '').upper() == ticker and live[0].get('intraday'):
                result['stock'] = live[0]
                result['verified_for_buy'] = True
                result['reference_only'] = False
                result['reason'] = '個股核價及日線來源均通過，可以進行盤中模型檢核'
            else:
                result['reason'] = '官方成交價已核實，但盤中因子尚未成功重算；仍不發布買點'
        else:
            result['reason'] = '已取得官方即時成交價；但日線來源待核，暫不發布買賣訊號'
        return result

    vendor = vendor_reference or {}
    if not result['official_quote_verified'] and vendor:
        try:
            vendor_stamp = dt.datetime.fromisoformat(str(vendor.get('timestamp') or ''))
            if vendor_stamp.tzinfo is None:
                vendor_stamp = vendor_stamp.replace(tzinfo=TAIPEI)
            vendor_stamp = vendor_stamp.astimezone(TAIPEI)
            vprice = _num(vendor.get('price'))
            official_stamp = None
            # A MIS timestamp with z='-' is not an actual matched-price time.
            # It must NOT suppress a correctly timestamped vendor minute bar.
            if row and _num(row.get('close')) is not None and row.get('date'):
                try:
                    official_stamp = dt.datetime.fromisoformat(str(row['date']))
                    if official_stamp.tzinfo is None:
                        official_stamp = official_stamp.replace(tzinfo=TAIPEI)
                    official_stamp = official_stamp.astimezone(TAIPEI)
                except (TypeError, ValueError, OverflowError):
                    pass
            more_recent = official_stamp is None or vendor_stamp > official_stamp
            if (vprice is not None and more_recent and vendor_stamp.date() == now.date()
                    and -90 <= (now - vendor_stamp).total_seconds() <= 6*3600):
                result.update({
                    'price': vprice,
                    'quote_time': vendor_stamp.strftime('%m/%d %H:%M'),
                    'price_source': str(vendor.get('source') or '第三方分鐘行情（可能延遲）'),
                    'price_label': '本日分鐘K參考價（非即時）',
                    'age_seconds': max(0, int((now - vendor_stamp).total_seconds())),
                    'reference_only': True,
                })
                result['reason'] = '官方成交未通過核價，第三方分鐘K僅供觀察，不得視為即時買點'
        except (TypeError, ValueError, OverflowError):
            pass
    return result
