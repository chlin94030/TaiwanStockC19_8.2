"""Non-blocking current-day observations; no automatic buy/order signals."""
from __future__ import annotations
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import threading
import time
from v19_config import TZ
from v19_data import safe_symbol

_LOCK=threading.RLock()
_POOL=ThreadPoolExecutor(max_workers=2,thread_name_prefix='V19Quote')
_CACHE={}
_PENDING={}


def _fetch(ticker):
    from intraday_engine import FinMindRealtimeClient
    from quote_observer_v1819 import compose_quote,fetch_vendor_batch
    now=datetime.now(TZ)
    try:
        client=FinMindRealtimeClient();rows,status=client.verified_snapshots([ticker])
        vendor=fetch_vendor_batch([ticker],now=now,max_tickers=1).get(ticker)
        quote=compose_quote(ticker,now=datetime.now(TZ),status=status,executable_rows=rows,
                            diagnostic_rows=getattr(status,'quote_diagnostics',[]),vendor=vendor,
                            daily_reference=None,daily_date='',eod_approved=False)
        # A verified quote is not proof that an experimental daily model may trade.
        quote['can_buy']=False;quote['is_buy_signal']=False
        return quote
    except Exception as e:
        return {'ticker':ticker,'price':None,'reason':type(e).__name__+': '+str(e),
                'timestamp':'','can_buy':False,'is_buy_signal':False,'day':now.date().isoformat()}


def observation(ticker,fetcher=None,now=None,ttl=20):
    ticker=safe_symbol(ticker);now=now or datetime.now(TZ);day=now.date().isoformat();fn=fetcher or _fetch
    with _LOCK:
        old=_CACHE.get(ticker,{})
        if old.get('day')!=day:old={}
        pending=_PENDING.get(ticker)
        if pending and pending.done():
            try:quote=pending.result()
            except Exception as e:quote={'ticker':ticker,'price':None,'reason':type(e).__name__}
            _PENDING.pop(ticker,None)
            # Each retained observation keeps its ORIGINAL timestamp/day.
            quote_day=str(quote.get('day') or str(quote.get('timestamp',''))[:10])
            if quote_day and quote_day!=day:
                quote={'ticker':ticker,'price':None,'reason':'Previous-session quote rejected','day':day}
            if quote.get('price') is None and (old.get('quote') or {}).get('price') is not None:
                reason=quote.get('reason','Provider did not return a newer price')
                quote=dict(old['quote'],refresh_error=reason)
            seen_black=bool(old.get('seen_black'))
            change=quote.get('change_pct')
            if isinstance(change,(int,float)) and change<0:seen_black=True
            quote['observed_black_to_red']=bool(seen_black and isinstance(change,(int,float)) and change>0)
            quote['can_buy']=False;quote['is_buy_signal']=False
            old={'day':day,'at':time.monotonic(),'quote':quote,'seen_black':seen_black}
            _CACHE[ticker]=old
        quote=dict(old.get('quote') or {'ticker':ticker,'price':None,'reason':'FETCHING','day':day,'can_buy':False})
        if ticker not in _PENDING and time.monotonic()-float(old.get('at',0))>=ttl:
            _PENDING[ticker]=_POOL.submit(fn,ticker)
        quote['pending']=ticker in _PENDING
        # Recompute freshness when displaying a retained same-day trade.
        stamp=quote.get('timestamp')
        if stamp:
            try:
                t=datetime.fromisoformat(stamp.replace('Z','+00:00'))
                if t.tzinfo is None:t=t.replace(tzinfo=TZ)
                age=max(0,int((now-t).total_seconds()));quote['age_seconds']=age
                quote['age_label']=f'{age}s' if age<60 else f'{age//60}min'
                if age>180:quote['can_execute']=False
            except (ValueError,TypeError):quote['can_execute']=False
        return quote
