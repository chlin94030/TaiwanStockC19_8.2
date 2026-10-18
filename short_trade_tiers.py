"""Evidence-based 5-session *entry* tiers, separate from canonical Alpha ranking.

Returns are model-estimated conservative NET fractions (0.03 means +3%,
not 3 percentage points of certain realized performance). The downside is a
risk estimate; all three numbers must come from the same candidate and quote.
The tiers are exploratory thresholds, NOT empirically calibrated guarantees.
"""
from __future__ import annotations

from math import isfinite

from strategy_config import SHORT_TRADE_TIER_POLICY


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def classify_short_trade(expected_net_return, reward_risk, downside=None) -> dict:
    """Classify price-relative 5-day evidence without changing Alpha selection.

    UNKNOWN : missing or non-finite model evidence; cannot qualify.
    BELOW   : under baseline research screen (1.5% / 1.2).
    WATCH   : baseline research screen, but NOT an executable priority entry.
    PRIORITY: >=3.0% AND >=1.8; can proceed to all other independent gates.
    HIGH    : >=5.0% AND >=2.0; *not* automatic buy.
    """
    expected = _number(expected_net_return)
    rr = _number(reward_risk)
    risk = _number(downside)
    policy = SHORT_TRADE_TIER_POLICY
    valid = expected is not None and rr is not None and rr >= 0
    tiers = (('HIGH', '高潛力優選', 'green', policy['high']),
             ('PRIORITY', '正式短線候選', 'green', policy['priority']),
             ('WATCH', '一般研究觀察', 'amber', policy['watch']))
    tier, label, color = 'UNKNOWN', '報酬證據待確認', 'gray'
    if valid:
        tier, label, color = 'BELOW', '未達研究門檻', 'red'
        for key, display, tint, requirements in tiers:
            if expected >= requirements['min_net_return'] and rr >= requirements['min_reward_risk']:
                tier, label, color = key, display, tint
                break
    can_advance = tier in {'PRIORITY', 'HIGH'}
    target = policy['priority']
    if not valid:
        reason = '五日保守淨報酬或估計下檔風險資料不足，不可推定有買入機會'
    elif can_advance:
        reason = ('可進入買點檢核，仍須通過官方即時價、量價、追高及部位風控'
                  if tier == 'PRIORITY' else '潛在報酬較高，仍需通過全部盤中買點與風控')
    else:
        reason = (f'正式短線候選需五日保守淨報酬至少 {target["min_net_return"]:.0%}，'
                  f'報酬／風險比至少 {target["min_reward_risk"]:.1f}；目前僅列研究觀察')
    return {
        'tier': tier, 'label': label, 'color': color,
        'can_advance_to_buy_gate': can_advance, 'model_return_available': valid,
        'expected_net_return': expected, 'expected_net_pct': round(expected * 100, 2) if expected is not None else None,
        'reward_risk': rr, 'downside': risk,
        'downside_pct': round(risk * 100, 2) if risk is not None else None,
        'reason': reason,
        'research_min_net_return': policy['watch']['min_net_return'],
        'research_min_reward_risk': policy['watch']['min_reward_risk'],
        'priority_min_net_return': policy['priority']['min_net_return'],
        'priority_min_reward_risk': policy['priority']['min_reward_risk'],
        'high_min_net_return': policy['high']['min_net_return'],
        'high_min_reward_risk': policy['high']['min_reward_risk'],
    }
