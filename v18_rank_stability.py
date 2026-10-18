"""Cross-sectional, deterministic *contemporaneous* ranking sensitivity.

Ranks are recomputed under small feature/blending perturbations. This is not
out-of-sample success and does not make predictions appear independently proven.
"""
from __future__ import annotations

import math
from typing import Any
from v18_confidence_engine import evaluate_recommendation_eligibility


def apply_rank_stability(stocks: list[dict], top_k: int = 20) -> dict:
    stats = {}
    for h in ('short', 'mid', 'long'):
        candidates = [s for s in stocks if isinstance(((s.get('horizons') or {}).get(h) or {}).get('rank_v18'), dict)]
        if len(candidates) < 5:
            stats[h] = {'measured': False, 'reason': '同時點全市場樣本不足'}
            continue
        codes = [s['ticker'] for s in candidates]
        rank = {s['ticker']: ((s['horizons'][h])['rank_v18']) for s in candidates}
        base_scores = {c: float(rank[c]['score']) for c in codes}
        scenarios = []
        for alpha in (.80, 1.20):
            scenarios.append({c: float(rank[c]['score_v17']) +
                                   (base_scores[c] - float(rank[c]['score_v17'])) * alpha
                              for c in codes})
        for sign in (-1., 1.):
            scenarios.append({
                s['ticker']: base_scores[s['ticker']] + sign * .08 *
                   float(rank[s['ticker']].get('overlay_weight') or 0) *
                   (float((s.get('v18_profile') or {}).get('dimensions', {}).get('momentum', 50)) -
                    float((s.get('v18_profile') or {}).get('dimensions', {}).get('quality', 50)))
                for s in candidates
            })
        # Re-sort *all* currently evaluated liquid stocks under each variation.
        base_order = sorted(codes, key=lambda c: (-base_scores[c], str(c)))
        baseline = {c: idx for idx, c in enumerate(base_order)}
        variant_orders = [sorted(codes, key=lambda c: (-sc[c], str(c))) for sc in scenarios]
        variant_ranks = [{c: idx for idx, c in enumerate(order)} for order in variant_orders]
        k = min(int(top_k), len(codes))
        overlap = [len(set(base_order[:k]) & set(order[:k])) / k for order in variant_orders]
        for stock in candidates:
            code = stock['ticker']; b = stock['horizons'][h]['rank_v18']
            deviations = [abs(v[code] - baseline[code]) for v in variant_ranks]
            # Normalize by universe size so 10 places in 2000 stocks isn't
            # treated like 10 places among 20. Emphasize fragile top ranks.
            delta_scaled = sum(deviations) / len(deviations) / max(1.0, math.sqrt(len(codes)))
            robustness = round(max(0., 100. - 24. * math.sqrt(delta_scaled)), 1)
            c = b.get('confidence') or {}
            dq = float(b.get('dq_score') or 0.)
            hist = c.get('historical_score')
            mc = round(min(100., .50 * dq + .30 * float(hist) + .20 * robustness), 1) if hist is not None else None
            b['mc_score'] = mc
            b['signal_stability_type'] = 'cross_sectional_rank_sensitivity'
            b['rank_stability_score'] = robustness
            b['rank_stability_mean_displacement'] = round(sum(deviations) / len(deviations), 2)
            c['stability_score'] = robustness
            c['mc_score'] = mc
            b['eligibility'] = evaluate_recommendation_eligibility(
                b['score'], dq, mc, c.get('independent_categories_count', 0), h,
                historical_oos_verified=False,
                pit_violations=sum(1 for x in ((stock.get('v18_profile') or {}).get('factors') or {}).values()
                                   if x.get('status') == 'LOOKAHEAD'),
            )
        stats[h] = {'measured': True, 'universe': len(codes), 'top_k': k,
                    'min_top_k_overlap_pct': round(100 * min(overlap), 1),
                    'mean_top_k_overlap_pct': round(100 * sum(overlap) / len(overlap), 1),
                    'scenarios': len(scenarios), 'note': '同期參數擾動穩定度，不是樣本外準確率'}
    return stats
