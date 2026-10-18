"""Five-session execution guards and exit discipline (research-only).

Open-to-now decline, red-to-black and gap extension are intraday-only.
No candle-close future information is used to make a decision at 09:15.
A price inside yesterday's zone NEVER overrides these live gates.
Thresholds are provisional; a verified minute-bar PIT replay is required.
"""
from __future__ import annotations

import math
from typing import Any

from strategy_config import ONE_WEEK_ENTRY_POLICY, HOLDING_SESSIONS


def _n(v: Any) -> float | None:
    try:
        value = float(v)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def open_to_now_guard(live: dict | None, reference_close: float | None = None) -> dict:
    """Evaluate an ALREADY provenance-verified intraday quote.

    Reject missing open, yesterday-close, high/low, clock time for a short buy.
    Never substitute signal EOD price for a missing current trade price.
    """
    policy = ONE_WEEK_ENTRY_POLICY
    info = live or {}
    p = _n(info.get('price'))
    op = _n(info.get('open'))
    prior = _n(info.get('previous_close')) or _n(reference_close)
    hi = _n(info.get('high'))
    lo = _n(info.get('low'))
    minutes = _n(info.get('minutes_since_open'))
    vwap_gap = _n(info.get('vwap_gap_pct'))
    if (p is None or op is None or prior is None or hi is None or lo is None or minutes is None
            or min(p,op,prior,hi,lo) <= 0 or hi < max(op,p,lo) or lo > min(op,p,hi)):
        return _result(False, 'INSUFFICIENT_OPEN_DATA', '\u7f3a\u5c11\u958b\u76e4\u50f9\u6216\u76e4\u4e2d\u6642\u9593\uff0c\u4e0d\u5224\u65b7\u53ef\u8cb7')
    gap = (op / prior - 1.0) * 100.0
    from_open = (p / op - 1.0) * 100.0
    from_high = (p / hi - 1.0) * 100.0
    from_prior = (p / prior - 1.0) * 100.0
    location = (p - lo) / (hi - lo) if hi > lo else 0.5
    diagnostics = {
        'open_gap_pct':round(gap,2), 'from_open_pct':round(from_open,2),
        'from_high_pct':round(from_high,2), 'from_prior_pct':round(from_prior,2),
        'day_position':round(location,3), 'vwap_gap_pct':vwap_gap,
        'minutes_since_open':round(minutes,1),
    }
    if minutes < float(policy['observe_open_minutes']):
        return _result(False,'OBSERVE_OPEN','\u958b\u76e4\u5f8c\u524d15\u5206\u9418\u5148\u89c0\u5bdf\uff0c\u4e0d\u7acb\u5373\u8ffd\u50f9',diagnostics)
    if gap >= float(policy['extreme_open_gap_pct']) and from_prior > 0.0:
        return _result(False,'GAP_CHASE','\u958b\u76e4\u5927\u5e45\u8df3\u7a7a\uff0c\u4e00\u9031\u7b56\u7565\u4e0d\u8ffd\u9ad8',diagnostics)
    if op > prior and from_prior <= float(policy['red_to_black_change_pct']):
        return _result(False,'RED_TO_BLACK','\u958b\u9ad8\u5f8c\u7531\u7d05\u7ffb\u9ed1\uff0c\u66ab\u4e0d\u8ffd\u9032',diagnostics)
    if gap >= float(policy['min_gap_for_fade_pct']) and from_open <= float(policy['open_fade_below_open_pct']):
        return _result(False,'OPEN_FADE','\u958b\u9ad8\u5f8c\u8dcc\u7834\u958b\u76e4\u50f9\uff0c\u5148\u7b49\u91cd\u65b0\u7ad9\u7a69',diagnostics)
    if (p < op and from_high <= -float(policy['high_to_now_reversal_pct'])
            and location < float(policy['min_day_position_when_fading'])):
        return _result(False,'HIGH_REJECTION','\u9ad8\u6a94\u56de\u843d\u4e14\u63a5\u8fd1\u76e4\u4e2d\u4f4e\u9ede\uff0c\u907f\u514d\u63a5\u5200',diagnostics)
    if (gap >= float(policy['min_gap_for_fade_pct']) and p < op and vwap_gap is not None
            and vwap_gap <= float(policy['vwap_weak_pct'])):
        return _result(False,'VWAP_WEAK','\u8dcc\u7834\u958b\u76e4\u50f9\u4e14\u843d\u65bc\u76e4\u4e2d\u5747\u50f9\u4e0b\u65b9',diagnostics)
    return _result(True,'PASS','\u958b\u76e4\u56de\u5410\u6aa2\u67e5\u901a\u904e\uff1b\u4ecd\u9700\u901a\u904e\u50f9\u683c\u3001\u91cf\u80fd\u8207\u98a8\u96aa\u5831\u916c',diagnostics)


def _result(ok: bool, code: str, reason: str, details: dict | None = None) -> dict:
    return {'pass':bool(ok), 'code':code,'reason':reason,**(details or {})}


def one_week_exit_review(*, entry: float | None, price: float | None, atr: float | None,
                         held_sessions: int | None, peak: float | None = None,
                         original_stop: float | None = None) -> dict:
    """Risk/time management for an EXISTING position; never simulates fills."""
    cost, current, vol = _n(entry), _n(price), _n(atr)
    if cost is None or cost <= 0 or current is None or current <= 0 or vol is None or vol <= 0:
        return {'state':'DATA', 'action':'\u8cc7\u6599\u4e0d\u8db3\uff0c\u4e0d\u81ea\u52d5\u8a55\u65b7\u51fa\u5834', 'max_hold_sessions':5}
    stop = _n(original_stop)
    if stop is None:
        stop = cost - 1.5 * vol
    reached = _n(peak)
    if reached is not None and reached >= cost + ONE_WEEK_ENTRY_POLICY['profit_protection_activation_atr'] * vol:
        stop = max(stop,cost,reached-ONE_WEEK_ENTRY_POLICY['profit_protection_trail_atr'] * vol)
    age = None if held_sessions is None else max(0,int(held_sessions))
    if current <= stop:
        code, action = 'RISK_EXIT','\u89f8\u53ca\u505c\u640d\u6216\u79fb\u52d5\u4fdd\u8b77\u4f4d\uff0c\u512a\u5148\u6aa2\u8996\u51fa\u5834'
    elif current >= cost + ONE_WEEK_ENTRY_POLICY['profit_target_review_atr'] * vol:
        code, action = 'TAKE_PROFIT_REVIEW','\u5df2\u9054\u7b2c5\u65e5\u524d\u7684ATR\u7372\u5229\u76ee\u6a19\uff0c\u8a55\u4f30\u5206\u6279\u7372\u5229\u4e86\u7d50'
    elif age is not None and age >= HOLDING_SESSIONS['short']:
        code, action = 'TIME_EXIT','\u9054\u7b2c5\u500b\u4ea4\u6613\u65e5\uff0c\u4e00\u9031\u7b56\u7565\u9032\u5165\u51fa\u5834\u6aa2\u8996'
    else:
        code, action = 'HOLD_WITH_DEADLINE','\u4ecd\u5728\u4e00\u9031\u671f\u9650\u5167\uff0c\u6301\u7e8c\u76e3\u6e2c\u9632\u5b88\u50f9'
    return {'state':code,'action':action,'protective_stop':round(stop,2),'max_hold_sessions':HOLDING_SESSIONS['short'],
            'held_sessions':age,'net_unadjusted_return_pct':round((current/cost-1)*100,2)}
