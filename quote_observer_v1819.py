"""V18.19 stock quote observability: **observation is not execution**.

An official individually verified trade <=180 seconds may reach the entry
engine (subject to valid EOD). An independently timestamped older MIS trade or
vendor minute-bar may be displayed, NEVER sent to the buy/holding engine.

No previous close, bid/ask, undocumented MIS field or provider metadata price is
promoted to an observed transaction. Market/ticker/date gates apply separately.
"""
from __future__ import annotations

import datetime as dt
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Callable, Mapping

import pandas as pd

from doctor_live_v1818 import fetch_yahoo_minute_reference
from live_session_v1810 import TAIPEI, quote_freshness
from intraday_engine import market_is_open

_VENDOR_LOCK = threading.RLock()
_VENDOR_CACHE: dict[str, dict] = {}
_VENDOR_PENDING: dict[str, Future] = {}
_VENDOR_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix='AlphaVendorReference')
_VENDOR_GENERATION = 0


def _now_tpe(now: dt.datetime | None) -> dt.datetime:
    now = now or dt.datetime.now(TAIPEI)
    return now.replace(tzinfo=TAIPEI) if now.tzinfo is None else now.astimezone(TAIPEI)


def _number(value) -> float | None:
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _positive(value) -> float | None:
    f = _number(value)
    return f if f is not None and f > 0 else None


def _parse_datetime(value) -> dt.datetime | None:
    if not value or (' ' not in str(value) and 'T' not in str(value)):
        return None
    try:
        s = dt.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return s.replace(tzinfo=TAIPEI) if s.tzinfo is None else s.astimezone(TAIPEI)
    except (ValueError, TypeError, OverflowError):
        return None


def stock_market(ticker: str) -> str:
    return 'TPEx' if str(ticker).strip().upper().endswith('.TWO') else 'TWSE'


def _age(now: dt.datetime, stamp: dt.datetime) -> int:
    return max(0, int((now - stamp).total_seconds()))


def _human_age(seconds: int | None) -> str:
    if seconds is None:
        return '時間未核'
    if seconds < 60:
        return f'{seconds}秒前'
    if seconds < 3600:
        return f'{seconds//60}分鐘前'
    return f'{seconds//3600}小時{(seconds%3600)//60}分前'


def _market_row(rows, ticker: str) -> dict | None:
    """Match BOTH ticker and exchange; do not trust a mixed batch's stock id."""
    ticker = str(ticker or '').strip().upper()
    stock_id = ticker.split('.')[0]
    expected = stock_market(ticker)
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict('records') if not rows.empty else []
    for row in (rows or []):
        if not isinstance(row, dict):
            continue
        if str(row.get('stock_id') or '') == stock_id and str(row.get('market') or '') == expected:
            return dict(row)
    return None


def _vendor_item(ticker: str, vendor: Mapping | None, now: dt.datetime) -> dict | None:
    vendor = vendor or {}
    price = _positive(vendor.get('price'))
    stamp = _parse_datetime(vendor.get('timestamp'))
    if price is None or stamp is None or stamp.date() != now.date():
        return None
    age = (now - stamp).total_seconds()
    if age < -90 or age > 6 * 3600:
        return None
    symbol = str(vendor.get('ticker') or '').strip().upper()
    if symbol and symbol != str(ticker).strip().upper():
        return None
    prior = _positive(vendor.get('previous_close'))
    change = ((price / prior) - 1) * 100 if prior else None
    return {'price': price, 'stamp': stamp, 'age_seconds': max(0, int(age)),
            'source': str(vendor.get('source') or '第三方分鐘K'), 'previous_close': prior,
            'change_pct': change, 'kind': 'VENDOR'}


def _official_item(ticker: str, row: dict | None, now: dt.datetime, status) -> tuple[dict | None, dict | None]:
    if not row:
        return None, None
    official_status = bool(getattr(status, 'available', False) and getattr(status, 'verified', False))
    check = quote_freshness(row, now, official_verified=official_status)
    stamp = _parse_datetime(row.get('date'))
    value = _positive(row.get('close'))
    candidate = None
    if stamp is not None and stamp.date() == now.date() and value is not None:
        age_seconds = (now - stamp).total_seconds()
        if -90 <= age_seconds <= 6 * 3600:
            candidate = {'price': value, 'stamp': stamp, 'age_seconds': max(0,int(age_seconds)),
                         'source': '證交所 MIS', 'previous_close': _positive(row.get('previous_close')),
                         'change_pct': _number(row.get('change_rate')), 'kind': 'OFFICIAL',
                         'can_execute': bool(check.get('can_execute'))}
    return candidate, check


def compose_quote(ticker: str, *, now: dt.datetime, status=None,
                  executable_rows=None, diagnostic_rows=None, vendor: dict | None = None,
                  daily_reference: float | None = None, daily_date: str = '',
                  eod_approved: bool = False) -> dict:
    """Return one source-traceable, date-safe price tile and independent BUY gate.

    The `can_execute` bit additionally requires the quote to be in the
    *executable* MIS result set, not only in diagnostic rows.
    """
    now = _now_tpe(now)
    live_row = _market_row(executable_rows, ticker)
    diag = _market_row(diagnostic_rows, ticker)
    row = live_row or diag
    official, check = _official_item(ticker, row, now, status)
    vendor_item = _vendor_item(ticker, vendor, now)
    official_live = bool(live_row and official and check and check.get('can_execute')
                         and market_is_open(now))
    selected = None
    if official_live:
        selected = official
    elif vendor_item and (official is None or vendor_item['stamp'] > official['stamp']):
        selected = vendor_item
    elif official:
        selected = official
    elif vendor_item:
        selected = vendor_item

    ref = _positive(daily_reference)
    ref_dt = str(daily_date or '')
    status_reason = str(getattr(status, 'reason', '') or '')
    if selected:
        price = selected['price']
        stamp = selected['stamp']
        prev = selected.get('previous_close')
        # A same-day MIS `y` is valid as last close if provider did not supply
        # chartPreviousClose, but is never itself treated as the current price.
        if prev is None and diag and str(diag.get('date') or '')[:10] == now.date().isoformat():
            prev = _positive(diag.get('previous_close'))
        if prev is None and ref is not None and ref_dt and ref_dt < now.date().isoformat():
            prev = ref
        pct = ((price / prev) - 1) * 100 if prev else None
        freshness = _human_age(selected['age_seconds'])
        if official_live:
            quality, badge = ('VERIFIED', '官方核實｜可檢核買點' if eod_approved else '官方核實｜日線未核，停止買點')
        elif selected['kind'] == 'VENDOR':
            quality, badge = 'VENDOR_OBSERVATION', '第三方分鐘參考｜不可作買點'
        else:
            quality, badge = 'OLD_OFFICIAL', '今日較早官方成交｜不可作買點'
        why = ('' if official_live else
               str(check.get('label') or '') if selected['kind'] == 'OFFICIAL' and check else
               '官方新成交價缺漏；顯示當日第三方分鐘K')
        return {'ticker': ticker, 'price': price, 'change_pct': pct, 'change_reference':
                ('官方昨收' if selected['kind']=='OFFICIAL' or (diag and str(diag.get('date') or '')[:10] == now.date().isoformat() and _positive(diag.get('previous_close'))) else
                 '第三方昨收／前日模型價' if prev else '無昨收'),
                'quote_time': stamp.strftime('%m/%d %H:%M:%S'), 'timestamp': stamp.isoformat(timespec='seconds'),
                'age_seconds': selected['age_seconds'], 'age_label': freshness,
                'source': selected['source'], 'quality': quality, 'badge': badge,
                'can_execute': official_live, 'can_buy': bool(official_live and eod_approved),
                'reason': why, 'previous_close': prev, 'official_reason': check.get('reason') if check else status_reason,
                'day': now.date().isoformat(), 'daily_date': ref_dt, 'daily_reference': ref,
                'total_volume': _number(diag.get('total_volume')) if diag else None,
                'volume_reported': bool(diag.get('volume_reported')) if diag else False}
    else:
        if diag and check:
            why = str(check.get('label') or '官方報價無成交價')
        elif status_reason:
            why = '官方未回傳本檔價格' if str(status_reason).startswith('MIS_PARTIAL') else status_reason
        else:
            why = '官方與第三方來源目前均無當日報價'
        return {'ticker': ticker, 'price': None, 'change_pct': None, 'change_reference': '無',
                'quote_time': '', 'timestamp': '', 'age_seconds': None, 'age_label': '未取得',
                'source': '當日成交價格缺漏', 'quality': 'NO_PRICE',
                'badge': '當日價格未取得', 'can_execute': False, 'can_buy': False,
                'reason': why, 'previous_close': None, 'official_reason': check.get('reason') if check else status_reason,
                'day': now.date().isoformat(), 'daily_date': ref_dt, 'daily_reference': ref,
                'total_volume': _number(diag.get('total_volume')) if diag else None,
                'volume_reported': bool(diag.get('volume_reported')) if diag else False}


def _newest_valid_vendor(ticker: str, current: dict | None, previous: dict | None,
                         now: dt.datetime) -> dict | None:
    """Never move a visible same-day bar backward when a feed responds out of order."""
    candidate = _vendor_item(ticker, current, now)
    prior = _vendor_item(ticker, previous, now)
    if candidate and (not prior or candidate['stamp'] >= prior['stamp']):
        return dict(current or {})
    if prior:
        return dict(previous or {})
    return None


def fetch_vendor_batch(tickers, *, now: dt.datetime | None = None,
                       fetcher: Callable | None = None, max_workers: int = 4,
                       max_tickers: int = 12, cache_ttl_seconds: int = 75) -> dict[str, dict]:
    """Bounded parallel minute-bar fallback. Cache includes negative results.

    Memory remains read-only observation. On a vendor outage, a previously
    fetched same-day minute bar may be retained, with its ORIGINAL timestamp.
    """
    now = _now_tpe(now)
    fetcher = fetcher or fetch_yahoo_minute_reference
    unique = list(dict.fromkeys(str(t or '').strip().upper() for t in tickers))[:max_tickers]
    today = now.date().isoformat()
    now_monotonic = time.monotonic()
    results: dict[str, dict] = {}
    to_fetch: list[str] = []
    with _VENDOR_LOCK:
        for ticker in unique:
            cached = _VENDOR_CACHE.get(ticker)
            if cached and cached.get('day') == today:
                observation = _vendor_item(ticker, cached.get('last_good'), now)
                if observation:
                    results[ticker] = dict(cached['last_good'])
                if now_monotonic - float(cached.get('checked_monotonic') or 0) < cache_ttl_seconds:
                    if ticker not in results and cached.get('error'):
                        results[ticker] = {'unavailable_reason': cached['error']}
                    continue
            to_fetch.append(ticker)
    if to_fetch:
        def request(ticker: str):
            try:
                value = fetcher(ticker, now=now)
                return value if isinstance(value, dict) else {}
            except Exception as exc:
                return {'unavailable_reason': type(exc).__name__}
        # Each request itself has a short network timeout. Maximum parallelism
        # avoids serially blocking the page for Top-10 equities.
        with ThreadPoolExecutor(max_workers=max(1,min(int(max_workers), 5))) as pool:
            futures = {pool.submit(request, ticker): ticker for ticker in to_fetch}
            for future in as_completed(futures):
                ticker = futures[future]
                result = future.result()
                valid = _vendor_item(ticker, result, now)
                with _VENDOR_LOCK:
                    prior = _VENDOR_CACHE.get(ticker) or {}
                    latest_good = _newest_valid_vendor(
                        ticker, result, prior.get('last_good') if prior.get('day') == today else None, now)
                    _VENDOR_CACHE[ticker] = {
                        'day': today, 'checked_monotonic': time.monotonic(),
                        'last_good': latest_good,
                        'error': (str(result.get('unavailable_reason') or 'NO_MINUTE_BAR')
                                  if not valid else ('VENDOR_OUT_OF_ORDER' if latest_good != result else '')),
                    }
                    fallback = _vendor_item(ticker, latest_good, now)
                    if fallback:
                        results[ticker] = dict(latest_good)
                    else:
                        results[ticker] = {'unavailable_reason': _VENDOR_CACHE[ticker]['error']}
    return results


def quote_health(tiles: list[dict]) -> dict:
    n = len(tiles)
    return {'requested': n,
            'verified_quote': sum(x.get('quality') == 'VERIFIED' for x in tiles),
            'old_official': sum(x.get('quality') == 'OLD_OFFICIAL' for x in tiles),
            'vendor_observation': sum(x.get('quality') == 'VENDOR_OBSERVATION' for x in tiles),
            'no_today_price': sum(x.get('quality') == 'NO_PRICE' for x in tiles),
            'displayable_price': sum(x.get('price') is not None for x in tiles),
            'buy_eligible_source': sum(bool(x.get('can_buy')) for x in tiles)}


def clear_vendor_cache() -> None:
    """Manual retry: allows the next check to bypass a cached provider failure."""
    global _VENDOR_GENERATION
    with _VENDOR_LOCK:
        _VENDOR_GENERATION += 1
        for future in _VENDOR_PENDING.values():
            future.cancel()
        _VENDOR_PENDING.clear()
        _VENDOR_CACHE.clear()



def fetch_vendor_batch_async(tickers, *, now: dt.datetime | None = None,
                            fetcher: Callable | None = None, cache_ttl_seconds: int = 75,
                            max_tickers: int = 12) -> dict[str, dict]:
    """Schedule optional reference quotes without blocking Streamlit rendering.

    The next fragment tick sees completed values. Pending jobs and vendor errors
    are visible as labels, never mistaken for actual market transactions.
    """
    now = _now_tpe(now)
    day = now.date().isoformat()
    fn = fetcher or fetch_yahoo_minute_reference
    result = {}
    for ticker in list(dict.fromkeys(str(t or '').strip().upper() for t in tickers))[:max_tickers]:
        with _VENDOR_LOCK:
            prev = _VENDOR_CACHE.get(ticker)
            if prev and prev.get('day') == day:
                good = prev.get('last_good')
                if _vendor_item(ticker, good, now):
                    result[ticker] = dict(good)
                if (time.monotonic() - float(prev.get('checked_monotonic') or 0) < cache_ttl_seconds):
                    if ticker not in result:
                        result[ticker] = {'unavailable_reason': str(prev.get('error') or 'NO_MINUTE_BAR')}
                    continue
            if ticker in _VENDOR_PENDING:
                if ticker not in result:
                    result[ticker]={'unavailable_reason':'BACKUP_FETCHING'}
                continue
            generation = _VENDOR_GENERATION
            def worker(sym=ticker):
                try:
                    obs = fn(sym, now=now)
                    return obs if isinstance(obs, dict) else {}
                except Exception as exc:
                    return {'unavailable_reason': type(exc).__name__}
            future=_VENDOR_POOL.submit(worker)
            _VENDOR_PENDING[ticker]=future
            def finalize(fut, sym=ticker, gen=generation):
                try:
                    obs=fut.result()
                except Exception as exc:
                    obs={'unavailable_reason':type(exc).__name__}
                with _VENDOR_LOCK:
                    if gen!=_VENDOR_GENERATION or _VENDOR_PENDING.get(sym) is not fut:
                        return
                    previous=_VENDOR_CACHE.get(sym) or {}
                    last_good=_newest_valid_vendor(
                        sym, obs, previous.get('last_good') if previous.get('day') == day else None, now)
                    fetched=_vendor_item(sym,obs,now)
                    _VENDOR_CACHE[sym]={
                        'day':day,'checked_monotonic':time.monotonic(),
                        'last_good':last_good,
                        'error': (str(obs.get('unavailable_reason') or 'NO_MINUTE_BAR')
                                  if not fetched else ('VENDOR_OUT_OF_ORDER' if last_good != obs else '')),
                    }
                    _VENDOR_PENDING.pop(sym,None)
            future.add_done_callback(finalize)
            if ticker not in result:
                result[ticker]={'unavailable_reason':'BACKUP_FETCHING'}
    return result
