"""Alpha Radar V18 integrated (PIT-gated) DQ/MC/evidence-aware blending.

Selection, evidence quality, historical analog confidence and entry risk are
separate quantities. No missing-evidence optimism, no default H=70/S=80.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
import math
from typing import Any

from strategy_config import V18_DIMENSION_WEIGHTS, V18_OVERLAY_BLEND


class FactorStatus(str, Enum):
    VERIFIED = 'VERIFIED'
    OBSERVED_ONLY = 'OBSERVED_ONLY'
    PENDING = 'PENDING'
    PARTIAL = 'PARTIAL'
    SOURCE_ERROR = 'SOURCE_ERROR'
    MISSING = 'MISSING'
    STALE = 'STALE'
    NOT_APPLICABLE = 'NOT_APPLICABLE'
    INVALID = 'INVALID'
    LOOKAHEAD = 'LOOKAHEAD'
    UNVERIFIED_TIME = 'UNVERIFIED_TIME'


GROUP_MEMBERS = {
    'quality': {'revenue': .50, 'margin': .30, 'cash': .20},
    'chip': {'trust': .30, 'institutions': .30, 'holders': .25, 'margin_clean': .15},
    'momentum': {'rs': .30, 'squeeze': .20, 'volume': .20, 'breakout': .30},
    'sector': {'peers': .55, 'breadth': .45},
}
FACTOR_GROUP = {name: group for group, sub in GROUP_MEMBERS.items() for name in sub}
FAMILY_BY_GROUP = {'quality': 'company', 'chip': 'ownership',
                   'momentum': 'price', 'sector': 'price'}
STATUS_BASE_Q = {
    FactorStatus.VERIFIED: 1.00,
    FactorStatus.OBSERVED_ONLY: .75,
    FactorStatus.PENDING: .70,
    FactorStatus.PARTIAL: .55,
}
RECOMMENDATION_GATES = {
    'short': {'min_dq': 60.0, 'min_mc': 65.0, 'min_families': 2},
    'mid': {'min_dq': 65.0, 'min_mc': 70.0, 'min_families': 3},
    'long': {'min_dq': 70.0, 'min_mc': 75.0, 'min_families': 3},
}


@dataclass(frozen=True)
class FactorEvidence:
    factor_id: str
    status: FactorStatus
    score: float | None
    reliability: float = 0.0
    category: str = ''
    source: str = ''
    period_end: str | None = None
    published_at: str | None = None
    first_seen_at: str | None = None
    note: str = ''
    pit_validated: bool = False
    valid_until: str | None = None
    decision_at: str | None = None


def _num(x: Any) -> float | None:
    try:
        value = float(x)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _clip(x, a=0, b=100):
    return max(a, min(b, float(x)))


def weights_for(horizon: str) -> dict[str, float]:
    """Exactly 13 *positive* factors, 100%; macro/heat excluded."""
    if horizon not in V18_DIMENSION_WEIGHTS:
        raise ValueError('unknown horizon: ' + str(horizon))
    dimension = V18_DIMENSION_WEIGHTS[horizon]
    result = {name: dimension[group] * sub_weight
              for group, member_weights in GROUP_MEMBERS.items()
              for name, sub_weight in member_weights.items()}
    if len(result) != 13 or abs(sum(result.values()) - 1) > 1e-10:
        raise AssertionError('invalid factor weights')
    return result


def _validated(e: FactorEvidence) -> tuple[bool, float]:
    try:
        status = FactorStatus(e.status)
    except ValueError:
        return False, 0.0
    score = _num(e.score)
    reliability = _num(e.reliability)
    if score is None or not (0 <= score <= 100) or reliability is None:
        return False, 0.0
    if status not in STATUS_BASE_Q:
        return False, 0.0
    # A typed evidence object cannot grant itself PIT validity merely by
    # writing status='VERIFIED'. Upstream official EOD/sector and PIT ledger
    # have to provide an explicit validated evidence-chain marker.
    if not e.pit_validated:
        return False, 0.0
    if not e.period_end:
        return False, 0.0
    if status in {FactorStatus.PENDING, FactorStatus.OBSERVED_ONLY} and not e.first_seen_at:
        return False, 0.0
    if status == FactorStatus.PENDING:
        # A previous period is usable only until a known decision timestamp.
        # The evidence must truly predate the decision and still be timely.
        if not e.valid_until or not e.decision_at:
            return False, 0.0
        from pit_data_manager import _dt
        decision = _dt(e.decision_at)
        expires = _dt(e.valid_until)
        first_seen = _dt(e.first_seen_at)
        if not all((decision, expires, first_seen)) or first_seen > decision or decision > expires:
            return False, 0.0
    # VERIFIED/PARTIAL/OBSERVED_ONLY have their PIT contracts checked upstream.
    return True, _clip(reliability, 0, 1) * STATUS_BASE_Q[status]


def calculate_dq_and_mc(evidences: list[FactorEvidence], horizon: str = 'short',
                        historical_sample_score: float | None = None,
                        signal_stability_score: float | None = None,
                        historical_oos_verified: bool = False) -> dict:
    weights = weights_for(horizon)
    by_id = {e.factor_id: e for e in evidences if e.factor_id in weights}
    # Missing known factors must remain in denominator; only proven N/A may be
    # excluded, so the algorithm cannot cherry-pick the factors it received.
    applicable = {fid: w for fid, w in weights.items()
                  if not (fid in by_id and by_id[fid].status == FactorStatus.NOT_APPLICABLE)}
    total = sum(applicable.values())
    if total <= 0:
        return {'dq_score': 0.0, 'coverage_pct': 0.0, 'mc_score': None,
                'v18_pure_alpha': None, 'independent_categories_count': 0,
                'category_coverage': {}, 'evidence_families': [], 'effective_evidence': 0.0,
                'factor_reliabilities': {}, 'historical_score': None, 'stability_score': None,
                'historical_oos_verified': False}
    dq, coverage, weighted_score = 0., 0., 0.
    group_cov = {k: False for k in GROUP_MEMBERS}
    group_weight = {k: 0.0 for k in GROUP_MEMBERS}
    by_factor = {}
    for fid, w in applicable.items():
        e = by_id.get(fid)
        valid, q = _validated(e) if e else (False, 0.0)
        normalized = w / total
        if valid:
            coverage += normalized
            dq += normalized * q
            weighted_score += normalized * q * float(e.score)
            group_cov[FACTOR_GROUP[fid]] = True
            group_weight[FACTOR_GROUP[fid]] += normalized * q
        by_factor[fid] = round(q, 3)
    groups = [k for k, weight in group_weight.items() if weight >= .03]
    # Technical and sector breadth share completed-price inputs. Count this
    # as one independent market evidence family rather than two.
    families = sorted({FAMILY_BY_GROUP[g] for g in groups})
    pure = round(_clip(weighted_score / dq), 1) if dq > 1e-12 else None
    h = _num(historical_sample_score)
    s = _num(signal_stability_score)
    if h is not None:
        h = _clip(h, 0, 100)
        # Historical analog-fit confidence is not verified future/OOS accuracy.
        if not historical_oos_verified:
            h = min(h, 74.0)
    if s is not None:
        s = _clip(s, 0, 100)
    mc = None
    # No fictitious fallback 70/80 scores. If one independent component is
    # unmeasured, expose pending MC instead of disguising it as a certainty.
    if h is not None and s is not None:
        mc = round(_clip(.50 * dq * 100 + .30 * h + .20 * s), 1)
    return {
        'dq_score': round(dq * 100, 1), 'coverage_pct': round(coverage * 100, 1),
        'mc_score': mc, 'v18_pure_alpha': pure, 'effective_evidence': round(dq, 5),
        'independent_categories_count': len(families),
        'category_coverage': group_cov, 'evidence_families': families,
        'factor_reliabilities': by_factor, 'historical_score': h,
        'stability_score': s, 'historical_oos_verified': bool(historical_oos_verified),
        'excluded_not_applicable': sorted(set(weights) - set(applicable)),
    }


def fuse_alpha_scores(base_alpha: float, v18_pure_alpha: float | None,
                      dq_score: float, lambda_base: float = .55) -> dict:
    base = _num(base_alpha)
    if base is None:
        raise ValueError('base alpha missing')
    pure = _num(v18_pure_alpha)
    quality = _num(dq_score)
    mix = _clip(lambda_base, 0, 1) * (_clip(quality, 0, 100) / 100.0) if pure is not None and quality is not None else 0.
    score = base * (1 - mix) + (pure if pure is not None else base) * mix
    return {'score': round(_clip(score), 1), 'overlay_weight': round(mix, 5),
            'score_v18': round(_clip(pure), 1) if pure is not None and mix > 0 else None,
            'score_v17': round(_clip(base), 1)}


def evaluate_recommendation_eligibility(final_alpha: float, dq_score: float,
                                         mc_score: float | None, indep_categories: int,
                                         horizon: str = 'short',
                                         *, historical_oos_verified: bool = False,
                                         pit_violations: int = 0) -> dict:
    gate = RECOMMENDATION_GATES[horizon]
    reasons = []
    if pit_violations:
        reasons.append('資料時間證據違規，不可列為正式推薦')
    if dq_score < gate['min_dq']:
        reasons.append(f"資料品質 {dq_score:.1f} 未達 {gate['min_dq']:.0f}")
    if mc_score is None:
        reasons.append('模型信心尚無可核實的歷史／訊號穩定證據')
    elif mc_score < gate['min_mc']:
        reasons.append(f"模型信心 {mc_score:.1f} 未達 {gate['min_mc']:.0f}")
    if indep_categories < gate['min_families']:
        reasons.append(f"獨立證據來源 {indep_categories} 類，需 {gate['min_families']} 類")
    # OOS status describes performance-evidence validation, not the historical
    # analog score; do not claim OOS verified before a genuine walk-forward run.
    eligible = not reasons
    return {'is_eligible': eligible,
            'status_label': '正式研究候選' if eligible else '研究觀察｜證據待補',
            'failed_reasons': reasons, 'thresholds_used': gate,
            'oos_status': '已驗證樣本外' if historical_oos_verified else '樣本外績效待驗證'}


def historical_reliability(forecast: dict | None) -> float | None:
    """Use actual historical analog count and original evidence confidence only."""
    forecast = forecast if isinstance(forecast, dict) else {}
    conf = _num(forecast.get('confidence_score'))
    n = _num(forecast.get('local_effective_n'))
    if conf is None or n is None or n <= 0:
        return None
    # Includes effective sample adequacy but never asserts independent OOS.
    sample = min(1.0, math.sqrt(n / 60.0))
    return round(min(74.0, conf * (.45 + .55 * sample)), 1)


def perturbation_stability(evidences: list[FactorEvidence], horizon: str) -> float | None:
    """Measured local leave-one-factor-out sensitivity, NOT global rank stability."""
    baseline = calculate_dq_and_mc(evidences, horizon)
    pure = baseline['v18_pure_alpha']
    if pure is None:
        return None
    weights = weights_for(horizon)
    total = 0.
    max_shift = 0.
    for e in evidences:
        valid, q = _validated(e)
        if valid:
            total += weights.get(e.factor_id, 0) * q
    if total < .15:
        return None
    for e in evidences:
        valid, q = _validated(e)
        part = weights.get(e.factor_id, 0) * q if valid else 0.0
        if part <= 0 or total - part <= .00001:
            continue
        change = abs(part * (float(e.score) - float(pure)) / (total - part))
        max_shift = max(max_shift, change)
    # Sensitivity measured, bounded and transparently labeled as local signal
    # robustness, not future prediction accuracy or global rank stability.
    return round(_clip(100 - 4 * max_shift), 1)


def profile_evidences(profile: dict, *, financial_industry: bool = False) -> list[FactorEvidence]:
    """Convert existing factor profile with per-section PIT provenance to typed evidence."""
    output = []
    for fid, category in FACTOR_GROUP.items():
        item = (profile.get('factors') or {}).get(fid) or {}
        status = item.get('status')
        if financial_industry and fid in {'margin', 'cash'}:
            status = 'NOT_APPLICABLE'
        try:
            status = FactorStatus(status or ('PARTIAL' if _num(item.get('score')) is not None else 'MISSING'))
        except ValueError:
            status = FactorStatus.INVALID
        q = _num(item.get('reliability'))
        if q is None:
            q = 0.0
        output.append(FactorEvidence(
            factor_id=fid, status=status, score=_num(item.get('score')),
            reliability=float(q), category=category,
            source=str(item.get('source') or ''),
            period_end=item.get('period_end'), published_at=item.get('published_at'),
            first_seen_at=item.get('first_seen_at'), note=str(item.get('note') or ''),
            pit_validated=bool(item.get('pit_verified')),
            valid_until=item.get('valid_until'),
            decision_at=item.get('decision_at'),
        ))
    return output
