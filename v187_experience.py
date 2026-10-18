"""V18.7 transparent watch/readiness layer.  Never issues unverified buy signals.

This module is intentionally pure: no network calls, no arbitrary scores from
missing inputs, and no mutations of Alpha/DQ/MC or the frozen base model.
"""
from __future__ import annotations
from collections import Counter
import math
from typing import Any
from industry_profile import fine_industry

# Sourced, observable US securities.  These are watchlists, not official SOX
# constituents, ETF holdings, or recommendations.  Missing values remain blank.
US_WATCH_GROUPS: dict[str, list[tuple[str, str]]] = {
    'AI運算與晶片': [('tsm_adr','台積電ADR'),('nvda','輝達'),('amd','超微'),('avgo','博通'),('mrvl','邁威爾'),('qcom','高通')],
    '記憶體與晶圓設備': [('mu','美光'),('asml','艾司摩爾'),('amat','應用材料'),('lrcx','科林研發'),('klac','科磊')],
    'AI伺服器／網通／電力': [('dell','戴爾'),('vrt','維諦'),('anet','Arista'),('smci','美超微')],
}
US_WATCH_SYMBOLS = {
    'avgo':'AVGO','mrvl':'MRVL','qcom':'QCOM','mu':'MU','asml':'ASML',
    'amat':'AMAT','lrcx':'LRCX','klac':'KLAC','dell':'DELL',
    'vrt':'VRT','anet':'ANET','smci':'SMCI',
}

TW_WATCH_GROUPS: dict[str, tuple[str, ...]] = {
    '先進半導體／封測': ('2330','2303','3711','3131','3583','6239','2454','3661','3443','2408'),
    'AI伺服器與組裝': ('2382','3231','6669','2317','2376','4938','2324','2356'),
    '散熱／電源／機構': ('2308','3017','3324','3653','2421','2059','3533'),
    '高速PCB／CCL': ('2383','6274','2368','3037','8046','3189'),
    '網通／光學': ('2345','6285','3008','2379','6531'),
    '重電／電網': ('1519','1503','1513','1609','1504'),
}

LIGHT_THRESHOLDS = {'alpha': (75, 60), 'dq': (75, 55), 'mc': (75, 55)}
LIGHT_EXPLANATIONS = {
    'alpha': ('值得優先研究', '可觀察', '吸引力較低', '尚無模型分數'),
    'dq': ('資料較充分', '部分證據缺漏', '重要資料不足', '資料未核實'),
    'mc': ('證據較穩健', '仍需驗證', '信心偏低', '缺乏實證'),
}

def finite(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool) or str(value).strip() in ('','—','None'):
            return None
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def signal_light(value: Any, kind: str) -> dict:
    """Color reflects this *one* metric only; alpha green is NOT a buy signal."""
    val = finite(value)
    good, watch = LIGHT_THRESHOLDS[kind]
    labels = LIGHT_EXPLANATIONS[kind]
    if val is None:
        return {'color': 'gray', 'icon': '⚪', 'label': labels[3], 'score': None}
    idx = 0 if val >= good else 1 if val >= watch else 2
    return {'color': ('green','amber','red')[idx], 'icon': ('🟢','🟡','🔴')[idx],
            'label': labels[idx], 'score': val}


def _research(stock: dict, horizon: str) -> dict:
    return ((stock.get('horizons') or {}).get(horizon) or {})


def entry_readiness(stock: dict, horizon: str, next_session: dict | None = None) -> dict:
    """Non-execution pre-open screening; hard restrictions cannot be overridden.

    READY_SETUP means only that historical EOD criteria currently align.
    Never means an actual buy: the next session needs a fresh verified official
    quote, price/volume/stop and risk budget again.
    """
    from recommendation_engine import _components
    c = _components(stock, horizon, live=False)
    nxt = next_session or {}
    z = c.get('zone') or {}
    p = finite(c.get('price'))
    entry = finite(c.get('entry'))
    sel = finite(c.get('selection'))
    expected = finite(c.get('expected'))
    rr = finite(c.get('reward_risk'))
    positive = finite(c.get('positive'))
    confidence = finite(c.get('confidence'))
    samples = c.get('sample_n') or 0
    chase = str((c.get('timing') or {}).get('chase_risk') or '')
    timing_action = str((c.get('timing') or {}).get('action') or '')
    position = finite((c.get('position') or {}).get('percent'))
    profile_penalty = finite((stock.get('v18_profile') or {}).get('risk_penalty_for_entry')) or 0
    # Next-session risk also includes fresh, verified macro pressure.  Never
    # mark an EOD setup ready if its execution-only safety layer rejects it.
    penalty = max(profile_penalty, finite(nxt.get('v18_entry_penalty')) or 0)
    macro_penalty = finite(nxt.get('v184_macro_entry_penalty')) or 0
    # Historical-sample evidence is a hard prerequisite for readiness.
    from strategy_config import BUY_PRIORITY_POLICY as policy
    from strategy_config import ONE_WEEK_ENTRY_POLICY
    evidence = (confidence is not None and confidence >= policy['min_forecast_confidence']
                and samples >= policy['min_sample_n'])
    price_ok = (p is not None and p > 0 and bool(z.get('available')))
    hard_no = chase in ('高','極高') or ('不追' in timing_action) or ('暫不' in timing_action) or (position is not None and position <= 0)
    from short_trade_tiers import classify_short_trade
    short_tier = (classify_short_trade(expected, rr, c.get('downside')) if horizon == 'short' else None)
    # Distinguish watch-only >=1.5%/1.2 from formal actionable >=3%/1.8.
    # Only a verified *next trading day* quote can subsequently produce BUY.
    tier_ok = bool(short_tier and short_tier['can_advance_to_buy_gate']) if horizon == 'short' else True
    opportunity = (tier_ok
                   and expected is not None and expected >= (ONE_WEEK_ENTRY_POLICY['min_short_expected_net_return'] if horizon=='short' else policy['min_expected_return'])
                   and rr is not None and rr >= (ONE_WEEK_ENTRY_POLICY['min_short_reward_risk'] if horizon=='short' else policy['min_reward_risk'])
                   and (positive is None or positive >= policy['min_positive_rate'])
                   and sel is not None and sel >= policy['min_selection'])
    trigger = finite(z.get('trigger'))
    low, high = finite(z.get('low')), finite(z.get('high'))
    gap_pct = ((trigger / p - 1)*100) if trigger and p and p>0 else None
    target_gap = ((high / p -1)*100) if high and p and p>0 else None
    if not price_ok:
        code, explanation = 'DATA', '價格或買入區資料不足，暫無條件判斷'
    elif not evidence:
        code, explanation = 'EVIDENCE', '歷史案例樣本或可靠度不足，先補證據'
    elif hard_no or penalty >= 4:
        code, explanation = ('OVERHEAT', '隔夜市場風險偏高，等盤中確認或降低部位' if macro_penalty >= 3
                             else '現價追價或風險預算不合格，暫不買進')
    elif not opportunity:
        code, explanation = ('EDGE', short_tier['reason'] if short_tier and not tier_ok
                             else '保守預期報酬、風險比或選股條件未通過')
    elif z.get('needs_trigger'):
        near = (gap_pct is not None and 0 <= gap_pct <= 3.0 and entry is not None and entry >= 58)
        code = 'TRIGGER' if near else 'TIMING'
        explanation = (f'距突破確認價約 {gap_pct:.1f}%，下一交易日仍需盤中價量確認'
                       if gap_pct is not None and gap_pct >= 0 else '等待突破確認價')
    elif bool(z.get('current_in_zone')) and entry is not None and entry >= policy['min_entry']:
        code, explanation = 'READY_SETUP', '收盤價位於策略區間且證據通過；下一交易日仍須核對官方即時價'
    elif bool(z.get('current_above_zone')):
        code, explanation = 'PULLBACK', f'現價高於參考區，距區間下緣約 {abs(target_gap):.1f}%' if target_gap is not None else '等回到合理買入區'
    elif bool(z.get('current_below_zone')):
        code, explanation = 'BELOW', '價格低於參考區，等待結構回穩與再次確認'
    elif entry is None or entry < policy['min_entry']:
        code, explanation = 'TIMING', '價格雖接近，但進場強度尚未達門檻'
    else:
        code, explanation = 'TIMING', '下一交易日等待量價確認'
    labels = {'READY_SETUP':'盤前條件就緒','TRIGGER':'接近突破','PULLBACK':'等合理價',
              'BELOW':'等結構回穩','TIMING':'待量價確認','OVERHEAT':'不追高',
              'EVIDENCE':'證據不足','EDGE':'報酬風險未過','DATA':'資料待核'}
    priority = {'READY_SETUP':0,'TRIGGER':1,'TIMING':2,'PULLBACK':3,'BELOW':4,
                'OVERHEAT':5,'EDGE':6,'EVIDENCE':7,'DATA':8}
    color = {'READY_SETUP':'green','TRIGGER':'amber','TIMING':'amber',
             'PULLBACK':'amber','BELOW':'amber','OVERHEAT':'red','EDGE':'red',
             'EVIDENCE':'gray','DATA':'gray'}[code]
    return {'code':code,'label':labels[code], 'color':color,
            'priority':priority[code], 'reason':explanation,
            'trigger_gap_pct':round(gap_pct,1) if gap_pct is not None else None,
            'price':p, 'reference_zone': {'low':low,'high':high,'trigger':trigger},
            'is_live_buy':False, 'short_tier':short_tier}


def prioritize_next_session(snap: dict | None, horizon: str='short', n: int=8, *,
                            pool_n: int | None=None, global_context: dict | None=None) -> list[dict]:
    """Look beyond original top-10 to surface actionable setups near the top.

    All shortlisted names must *still* pass _selection_quality. Selection
    ranking scores are unchanged. Every item is research-only until MIS quote.
    """
    from radar_service import _selection_quality
    from recommendation_engine import evaluate_next_session
    stocks = [s for s in ((snap or {}).get('stocks') or []) if isinstance(s,dict)]
    stocks.sort(key=lambda s: finite(_research(s,horizon).get('ranking_score')) or -1, reverse=True)
    # A readiness search must not silently exclude lower-ranked, legitimate
    # candidates: evaluate every eligible stock, then cache the result by EOD id.
    limit = pool_n if pool_n is not None else len(stocks)
    eligible = [s for s in stocks if _selection_quality(s,horizon)[0]][:limit]
    out=[]
    for s in eligible:
        e = evaluate_next_session(s,horizon,global_context=global_context)
        readiness = entry_readiness(s,horizon,e)
        obj = dict(s)
        e['readiness'] = readiness
        obj['next_session'] = e
        out.append(obj)
    out.sort(key=lambda s:(int(s['next_session']['readiness']['priority']),
                           -float(s['next_session']['score']),
                           -(finite(_research(s,horizon).get('ranking_score')) or 0)))
    return out[:max(0,int(n))]


def readiness_counts(stocks: list[dict], horizon: str='short') -> Counter:
    result=Counter()
    for s in stocks:
        sblock = s.get('next_session') or {}
        ready = sblock.get('readiness') or entry_readiness(s,horizon,sblock)
        result[str(ready.get('code'))] += 1
    return result


def taiwan_watch_snapshot(snap:dict|None) -> dict[str,list[dict]]:
    """Observed current-model Taiwan stocks, NOT a re-ranked recommendation."""
    stocks = {str(s.get('ticker') or '').split('.')[0]:s for s in ((snap or {}).get('stocks') or []) if isinstance(s,dict)}
    by_group={}
    for group, codes in TW_WATCH_GROUPS.items():
        rows=[]
        for code in codes:
            s=stocks.get(code)
            if not s:
                continue
            rows.append({'ticker':str(s.get('ticker') or code),'name':s.get('name') or code,
                         'category':fine_industry(s.get('ticker') or code,s.get('name'),s.get('industry')),
                         'price':finite(s.get('price')), 'date':str(s.get('price_date') or ''),
                         'score_short':finite(_research(s,'short').get('ranking_score'))})
        by_group[group]=rows
    return by_group
