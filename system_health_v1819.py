"""Non-blocking V18.19 daily-field contract; never guesses missing evidence.

This is a monitor, not an additional model. Existing Alpha/DQ/MC scores and
ranking are never altered by it. Diagnose breadth by observing the SAME fields
rendered in the mobile recommendations, not different substitute data.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any


def _valid_number(v: Any, positive: bool=False) -> bool:
    try:
        n=float(v)
        return math.isfinite(n) and (n>0 if positive else True)
    except (ValueError, TypeError):
        return False


def inspect_daily_fields(snapshot: dict | None, *, limit: int=10) -> dict:
    snapshot=snapshot or {}
    date=str(snapshot.get('price_date') or '')
    stocks=list(snapshot.get('stocks') or [])
    stock_rows=[]
    checks={name:{'total':0,'available':0,'invalid':0,'missing':0,'samples':[]} for name in
            ('股票代碼','股票名稱','日線收盤參考價','個股日線日期','短線Alpha','中線Alpha','長線Alpha',
             'DQ證據品質','MC模型信心','歷史買點分數','歷史風控價區','資料來源日')}
    problems=[]
    def mark(key:str, code:str, available:bool, invalid:bool=False):
        entry=checks[key];entry['total']+=1
        if available:entry['available']+=1
        elif invalid:entry['invalid']+=1
        else:entry['missing']+=1
        if not available and len(entry['samples'])<5:entry['samples'].append(code)
    for stock in stocks[:max(0,int(limit))]:
        code=str(stock.get('ticker') or '未記錄')
        sd=str(stock.get('price_date') or '')
        horizons=stock.get('horizons') or {}
        row={'ticker':code,'name':str(stock.get('name') or ''),'price_date':sd}
        mark('股票代碼',code,bool(code and code!='未記錄' and '.' in code))
        mark('股票名稱',code,bool(str(stock.get('name') or '').strip()))
        mark('日線收盤參考價',code,_valid_number(stock.get('price'),positive=True))
        mark('個股日線日期',code,bool(date and sd==date), invalid=bool(sd and date and sd != date))
        mark('資料來源日',code,bool(date), invalid=bool(date and sd and sd>date))
        for h,label in (('short','短線Alpha'),('mid','中線Alpha'),('long','長線Alpha')):
            block=horizons.get(h) or {}
            rank=block.get('ranking_score')
            mark(label,code,_valid_number(rank) and 0<=float(rank)<=100,invalid=_valid_number(rank))
        short=horizons.get('short') or {}
        rank_v18=short.get('rank_v18') or {}
        dq=rank_v18.get('dq_score'); mc=rank_v18.get('mc_score')
        mark('DQ證據品質',code,_valid_number(dq) and 0<=float(dq)<=100,invalid=_valid_number(dq))
        mark('MC模型信心',code,_valid_number(mc) and 0<=float(mc)<=100,invalid=_valid_number(mc))
        timing=short.get('entry_timing') or {}
        mark('歷史買點分數',code,_valid_number(timing.get('score')) or
             _valid_number((stock.get('next_session') or {}).get('entry_score')))
        plan=short.get('plan') or {}
        zl=plan.get('zone_low');zh=plan.get('zone_high')
        have_zone=_valid_number(zl,positive=True) and _valid_number(zh,positive=True)
        mark('歷史風控價區',code,bool(have_zone and float(zl)<=float(zh)),invalid=bool(have_zone))
        row.update({'alpha':(short or {}).get('ranking_score'), 'dq':dq,'mc':mc,
                    'historical_entry':timing.get('score'), 'price':stock.get('price'),
                    'zone_low':zl,'zone_high':zh})
        stock_rows.append(row)
    if not stocks:problems.append('沒有股票快照；需要建立完整日線基準')
    for name,counts in checks.items():
        if counts['invalid']:
            problems.append(f"{name}存在格式或來源日期異常 {counts['invalid']} 檔")
    return {'model_date':date or '未建立','sample_count':len(stock_rows),
            'fields':checks,'issues':problems,'stock_samples':stock_rows,
            'meaning':'每日欄位快照檢查，並非盤中來源可達性或選股報酬保證'}


def inspect_news_window(items:list[dict], now:dt.datetime,max_age_hours:int=72)->dict:
    if now.tzinfo is None: raise ValueError('now must be timezone-aware')
    result={'total':len(items),'within_window':0,'expired':0,'missing_published_at':0,'future_invalid':0}
    for item in items:
        try:
            stamp=dt.datetime.fromisoformat(str(item.get('published_iso') or '').replace('Z','+00:00'))
            if stamp.tzinfo is None: raise ValueError('timezone is unknown')
        except (ValueError,TypeError,OverflowError):
            result['missing_published_at']+=1;continue
        age=(now-stamp).total_seconds()/3600
        if age<-0.5:result['future_invalid']+=1
        elif age<=max_age_hours:result['within_window']+=1
        else:result['expired']+=1
    return result


def verified_recent_news_items(items: list[dict], now: dt.datetime, max_age_hours: int=72) -> list[dict]:
    """Strict 72-hour filter for *visible* headlines, timezone required.

    An RSS fetch may have its original timezone; do not silently infer Taipei
    time when the upstream did not specify one. Reject >30-minute future dates.
    """
    if now.tzinfo is None:
        raise ValueError('now must have a time zone')
    recent=[]
    for item in items:
        try:
            stamp=dt.datetime.fromisoformat(str(item.get('published_iso') or '').replace('Z','+00:00'))
            if stamp.tzinfo is None:
                continue
            hours=(now-stamp).total_seconds()/3600.0
            if -0.5 <= hours <= max_age_hours:
                recent.append(item)
        except (ValueError,TypeError,OverflowError,AttributeError):
            continue
    return recent
