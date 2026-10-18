"""Frozen core strategy configuration for Alpha Radar V16.6 Dynamic Intraday Freeze.

Architecture contract
---------------------
1. Selection score ranks *which stock* is attractive for a horizon.
2. Intraday score measures *current session strength* only.
3. Entry timing decides *whether the current price is attractive enough to buy*.
4. Market regime controls *position size*, not stock quality/ranking.

Keep changes to strategy thresholds in this file.  The lower-level feature engine
(return_first_model.py) is intentionally frozen through V16.6 to avoid accidental
model drift while live validation is running.
"""
from __future__ import annotations

ARCHITECTURE_VERSION = "v18.21.0-pattern-chip-shadow"
DOCTOR_VERSION = "v17.1.0-synced-briefing"
HORIZONS = ("short", "mid", "long")
HOLDING_SESSIONS = {"short": 5, "mid": 40, "long": 120}
HORIZON_LABELS = {"short": "5 sessions / 1 week", "mid": "40 sessions / 2 months", "long": "120 sessions / 6 months"}


# V17 canonical production contract.  The user-facing model has one reproducible
# definition: every liquid stock with enough completed-day history is evaluated,
# and all stocks are compared on the same evidence scale.  Research/diagnostic
# overrides are allowed only in explicit advanced mode and are never labelled as
# the canonical ranking.
CANONICAL_MODEL_POLICY = {
    "history_period": "5y",
    "model_family": "full",
    "evaluate_all_liquid": True,
    "min_history_rows": 140,
    "research_breadth": True,
    "research_detail_top_per_horizon": 20,
    "research_neutral_score": 50.0,
    "missing_evidence_penalty_max": 4.0,
    "intraday_promote_max": 24,
    # The baseline model includes a broader investable universe; the final
    # shortlist applies horizon-specific liquidity/evidence guardrails below.
    "baseline_min_avg_turnover": 3_000_000.0,
    "horizon_min_avg_turnover": {"short": 10_000_000.0, "mid": 5_000_000.0, "long": 3_000_000.0},
    "horizon_min_evidence_coverage": {"short": 0.60, "mid": 0.70, "long": 0.78},
    "long_require_company_evidence": True,
    # Daily official EOD keeps the model current; the expensive multi-year Yahoo
    # history rebuild is maintenance, not a daily necessity.
    "full_history_refresh_days": 30,
    "canonical_label": "V18.4 全市場 PIT 證據與信心模型",
}

# Daily stock-selection layer.  These are the high-level evidence weights.
SELECTION_WEIGHTS = {
    "short": {"technical": 0.56, "empirical": 0.22, "fundamental": 0.06, "flow": 0.16},
    "mid": {"technical": 0.42, "empirical": 0.22, "fundamental": 0.22, "flow": 0.14},
    "long": {"technical": 0.34, "empirical": 0.22, "fundamental": 0.34, "flow": 0.10},
}

# Historical-analog evidence is useful only in proportion to its own quality.
# The raw empirical score is therefore shrunk toward neutral (50) when the
# analog evidence confidence is weak. This prevents a small/noisy analog set
# from dominating the daily selector without deleting the evidence entirely.
EMPIRICAL_RELIABILITY_POLICY = {
    "neutral_score": 50.0,
    "confidence_floor_multiplier": 0.45,
    "minimum_confidence_for_action": 35.0,
    "minimum_effective_samples_for_action": 26,
}

# Intraday-strength layer.  No chase/overheat penalty belongs here; price
# extension is an execution issue handled by ENTRY_POLICY below.
INTRADAY_FACTOR_WEIGHTS = {
    "relative_market": 0.30,
    "vwap": 0.20,
    "volume": 0.18,
    "day_position": 0.12,
    "open_move": 0.08,
    "bid_pressure": 0.05,
    "turnover": 0.07,
}
INTRADAY_FACTOR_RANGES = {
    "relative_market": (-2.0, 4.0),
    "vwap": (-1.5, 2.0),
    "volume": (0.55, 2.50),
    "day_position": (0.18, 0.88),
    "open_move": (-2.5, 4.0),
    "bid_pressure": (0.35, 0.65),
    "turnover_twd": (30_000_000.0, 2_500_000_000.0),
}
INTRADAY_SPREAD_FREE_PCT = 0.60
INTRADAY_SPREAD_PENALTY_PER_PCT = 6.0

# Intraday signal should not dominate immediately after the open.  The weights
# mature as more of the session is observed.  Values are maximum blend weights
# with the complete daily model.  Key = minutes elapsed since 09:00.
INTRADAY_BLEND_SCHEDULE = {
    "short": ((0, 0.22), (20, 0.28), (60, 0.35), (150, 0.41), (225, 0.45)),
    "mid": ((0, 0.06), (20, 0.08), (60, 0.10), (150, 0.13), (225, 0.15)),
    "long": ((0, 0.02), (20, 0.025), (60, 0.03), (150, 0.04), (225, 0.05)),
}

# Entry/execution layer.  These rules NEVER change the selection score.
ENTRY_POLICY = {
    "base_score": 78.0,
    "extension": {
        "below_breakout": -0.60,
        "ideal_max": 0.45,
        "acceptable_max": 0.90,
        "warm_max": 1.50,
        "hot_max": 2.00,
    },
    "ma5_extension": {"warm": 1.20, "hot": 2.00},
    "one_day_move": {"watch": 3.0, "hot": 6.0, "very_hot": 8.0, "near_limit": 9.5},
    "market_surge": 2.0,
    "relative_strength_bonus": 2.0,
    "relative_weakness": -1.0,
    "volume": {"healthy_low": 1.10, "healthy_high": 2.50, "extreme": 3.50},
    "day_position": {"weak": 0.28, "strong": 0.72},
    "action": {"good": 82.0, "small": 70.0, "wait": 58.0, "watch": 42.0},
    "risk": {"medium": 15.0, "high": 34.0, "extreme": 55.0},
}

# Regime is a risk-budget layer only.  1.0 means 100% of the user's normal
# planned position, not 100% of portfolio capital.
REGIME_POSITION_MULTIPLIER = {
    "BULL": 1.00,
    "NEUTRAL": 0.70,
    "BEAR": 0.35,
    "UNKNOWN": 0.50,
}
ENTRY_POSITION_MULTIPLIER = {
    "good": 1.00,      # entry score >= 82
    "small": 0.55,     # >= 70
    "wait": 0.25,      # >= 58
    "avoid": 0.00,     # < 58
}
CHASE_POSITION_CAP = {
    "低": 1.00,
    "中": 0.55,
    "高": 0.25,
    "極高": 0.00,
    "待確認": 0.25,
}

# Data/scan defaults remain complete.  Faster operation comes from caching,
# batching and feature reuse, not shrinking the candidate universe.
SCAN_DEFAULTS = {
    "reference_size": 600,
    "candidate_size": 1000,
    "research_pool_per_horizon": 10,
    "history_period": "5y",
    "min_price": 10.0,
    "min_avg_turnover": 3_000_000.0,
}

TRADE_PLAN = {
    "short": {"trigger_mult": 1.005, "zone_low_mult": 0.98, "zone_high_mult": 1.01, "chase_mult": 1.025, "invalidation_atr": 1.5, "use_low20_floor": True, "entry_mode": "breakout_confirmed"},
    "mid": {"trigger_mult": 1.01, "zone_low_mult": 0.96, "zone_high_mult": 1.015, "chase_mult": 1.04, "invalidation_atr": 2.5, "use_low20_floor": False, "entry_mode": "zone_confirmed"},
    "long": {"trigger_mult": 1.02, "zone_low_mult": 0.94, "zone_high_mult": 1.02, "chase_mult": 1.05, "invalidation_atr": 3.0, "use_low20_floor": False, "entry_mode": "zone_confirmed"},
}

INTRADAY_STATE = {
    "relative_weak": -1.5,
    "vwap_weak": -1.2,
    "day_position_weak": 0.35,
    "strong_score": 70.0,
    "strong_relative": 0.8,
    "strong_volume": 1.05,
    "stable_score": 60.0,
    "stable_vwap": -0.2,
    "volume_watch": 1.20,
    "volume_watch_score": 50.0,
    "reason_relative": 1.0,
    "reason_volume": 1.25,
    "reason_vwap_up": 0.4,
    "reason_vwap_down": -0.5,
}

# V16.6 decision layer.  This does not feed back into the frozen selection score;
# it only ranks *actionable* opportunities after a stock has already passed the
# selection layer.
BUY_PRIORITY_POLICY = {
    "weights": {
        "selection": 0.30,
        "entry": 0.30,
        "expected_return": 0.20,
        "positive_rate": 0.10,
        "reward_risk": 0.10,
    },
    "expected_return_range": (-0.02, 0.20),
    "positive_rate_range": (0.45, 0.75),
    "reward_risk_range": (0.50, 3.00),
    "min_selection": 62.0,
    "min_entry": 70.0,
    "strong_entry": 82.0,
    "min_expected_return": 0.02,
    "min_positive_rate": 0.52,
    "min_reward_risk": 1.10,
    "min_forecast_confidence": 35.0,
    "min_sample_n": 26,
    # Positive expected return is discounted when historical analog confidence
    # is weak; negative estimates are never softened.
    "expected_confidence_floor": 0.55,
    # Reward/risk uses the worse of structural stop distance and historical
    # p10 downside, avoiding an unrealistically attractive ratio.
    "tail_risk_floor": 0.002,
}

NEXT_SESSION_POLICY = {
    "weights": {
        "selection": 0.55,
        "entry": 0.18,
        "expected_return": 0.15,
        "positive_rate": 0.07,
        "reward_risk": 0.05,
    },
    "strong_watch": 75.0,
    "normal_watch": 65.0,
}

# Dynamic intraday scan. The completed-day model remains frozen during the
# session. Every minute we fetch quotes for the liquid universe, use a cheap
# discovery pass to find new leaders, then run the heavier execution layer only
# on a bounded detail pool.
LIVE_SCAN = {
    "per_horizon": 80,          # completed-day core from each horizon
    "daily_core_max": 220,      # union cap after de-duplication
    "full_market_quote_max": 2500,
    "discovery_top": 140,       # strongest live movers from the liquid universe
    "detail_max": 360,          # max names entering entry/forecast execution pass
    "refresh_seconds": 60,
    "cache_ttl_seconds": 52,
    "quote_batch_size": 45,
    "gate_summary_top": 6,
}

# A stock that has moved above the prior-session pullback zone may still be an
# executable continuation entry, but only when the live tape confirms strength
# and the stock is not extended. This path is intentionally stricter than a
# normal pullback entry so "continuation" never becomes a synonym for chasing.
CONTINUATION_ENTRY_POLICY = {
    "min_selection": 65.0,
    "min_intraday_score": 68.0,
    "min_entry": 68.0,
    "min_relative_market_pct_pt": 0.80,
    "volume_ratio_low": 1.00,
    "volume_ratio_high": 2.80,
    "min_day_position": 0.55,
    "max_change_pct": 6.0,
    "max_extension_atr": 1.20,
    "min_extension_atr": 0.00,
    "min_vwap_gap_pct": -0.20,
    "max_vwap_gap_pct": 2.50,
    "max_spread_pct": 0.80,
    "min_expected_return": 0.025,
    "min_reward_risk": 1.20,
    "min_positive_rate": 0.52,
    "max_position_percent": 35,
}


# V16.8 global overnight / panic-reversal execution layer.  This context never
# changes the frozen Taiwan selection ranking.  It only adjusts risk budget and
# can enable a small contrarian entry after a broad external shock when the stock
# shows clear intraday resilience/recovery.
GLOBAL_OVERNIGHT_POLICY = {
    "weights": {
        "sox": 0.25,
        "ndx": 0.20,
        "tsm_adr": 0.20,
        "spx": 0.10,
        "vix": 0.10,
        "nq_fut": 0.08,
        "es_fut": 0.03,
        "tnx": 0.02,
        "dxy": 0.02,
    },
    # Percentage-point moves that map to ~100 risk for each factor.
    "risk_scale_pct": {
        "sox": 4.0,
        "ndx": 3.5,
        "tsm_adr": 4.5,
        "spx": 3.0,
        "vix": 25.0,
        "nq_fut": 2.5,
        "es_fut": 2.0,
        "tnx": 3.0,
        "dxy": 1.2,
    },
    "min_available_weight": 0.45,
    "stress_threshold": 40.0,
    "panic_threshold": 58.0,
    "max_relief_points": 14.0,
    "cache_minutes": 10,
    "position_multiplier": {
        "stress_at": 40.0,
        "panic_at": 58.0,
        "severe_at": 75.0,
        "stress": 0.80,
        "panic": 0.60,
        "severe": 0.40,
    },
}

PANIC_REVERSAL_POLICY = {
    # Either a confirmed external shock or a broad Taiwan selloff can activate
    # the reversal search.  This avoids relying on US data when the feed is down.
    "min_global_shock": 52.0,
    "taiwan_selloff_pct": -1.50,
    # The candidate itself must be sold down, but not pinned near limit-down.
    "min_change_pct": -8.50,
    "max_change_pct": -1.00,
    "min_daily_selection": 64.0,
    # New-entry timing may be low during a panic because the old breakout was
    # mechanically lost; resilience must compensate before a small contrarian buy.
    "min_entry_score": 44.0,
    "min_relative_market_pct_pt": 0.60,
    "min_rebound_from_low_pct": 1.00,
    "min_day_position": 0.36,
    "min_vwap_gap_pct": -0.60,
    "volume_ratio_low": 0.80,
    "volume_ratio_high": 3.20,
    "max_spread_pct": 0.90,
    "min_expected_return": 0.04,
    "min_reward_risk": 1.35,
    "min_positive_rate": 0.52,
    "min_forecast_confidence": 35.0,
    "min_sample_n": 26,
    "max_position_percent": 20,
    "severe_shock_position_percent": 12,
}


# Reproducible 2026 walk-forward validation defaults. These values do not alter
# production ranking; they define the external validation protocol.
VALIDATION_POLICY = {
    "year": 2026,
    "stocks": 100,
    "points_per_horizon": 12,
    "top_n": 5,
    "history_period": "8y",
    "seed": 1337,
    "calibration_fraction": 0.67,
    "technical_share_grid": (0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
    "min_history_rows": 380,
}


# V16.7 single-stock holding decision. This layer does not feed back into
# selection or entry ranking. It deliberately avoids using new-entry timing to
# decide whether an existing position should be held.
DOCTOR_HOLD_POLICY = {
    "weights": {
        "structure": 0.35,
        "trend": 0.25,
        "forecast": 0.25,
        "selection": 0.15,
    },
    "min_forecast_confidence": 35.0,
    "min_sample_n": 26,
    "strong_hold_score": 68.0,
    "hold_score": 52.0,
    "reduce_score": 40.0,
    "intraday_weak_penalty": 7.0,
    "intraday_strong_bonus": 4.0,
    # Profit protection is shown only after the position has a meaningful gain.
    # It is a model reference, not a guaranteed execution price.
    "profit_protect_activation": {"short": 0.08, "mid": 0.12, "long": 0.18},
    "profit_protect_atr_buffer": {"short": 0.80, "mid": 1.20, "long": 1.60},
}


# V18 controlled roll-out.  Positive four dimensions sum to 100% by horizon.
# The 5th, macro/overheat, adjusts EXECUTION only (up to 10 score points).
# Historical empirical forecasts remain in the original V17 selection score;
# blending prevents unvalidated factor calibration from erasing tested logic.
V18_DIMENSION_WEIGHTS = {
    "short": {"quality": 0.12, "chip": 0.18, "momentum": 0.48, "sector": 0.22},
    "mid":   {"quality": 0.28, "chip": 0.24, "momentum": 0.29, "sector": 0.19},
    "long":  {"quality": 0.40, "chip": 0.16, "momentum": 0.23, "sector": 0.21},
}
V18_OVERLAY_BLEND = {"short": 0.50, "mid": 0.50, "long": 0.45}
# Discourage missing data without pretending missing factors are positive.
# V18.4: fixed missing penalty removed. Effective evidence dynamically controls blending.
V18_MISSING_PENALTY = 0.0

# V18.11 one-week execution guardrails (research defaults, not OOS calibrated).
# Never enter simply because the quote has entered yesterday's buy zone.
ONE_WEEK_ENTRY_POLICY = {
    "observe_open_minutes": 15,
    "min_gap_for_fade_pct": 0.80,
    "open_fade_below_open_pct": -0.50,
    "red_to_black_change_pct": 0.0,
    "min_day_position_when_fading": 0.40,
    "high_to_now_reversal_pct": 1.50,
    "extreme_open_gap_pct": 3.50,
    "vwap_weak_pct": -0.30,
    # Minimal research screening only; NOT sufficient to issue a live BUY.
    "min_short_expected_net_return": 0.015,
    "min_short_reward_risk": 1.20,
    "max_hold_sessions": HOLDING_SESSIONS["short"],
    "profit_protection_activation_atr": 1.40,
    "profit_target_review_atr": 2.20,
    "profit_protection_trail_atr": 1.00,
}


# V18.12 controlled short-entry tiers. Parameters are hypotheses awaiting
# non-overlapping, real point-in-time walk-forward performance testing.
# WATCH only qualifies a stock for research. Priority/High must ALSO pass the
# separate structure, intraday freshness, open-fade, evidence and risk gates.
SHORT_TRADE_TIER_POLICY = {
    "watch": {"min_net_return": 0.015, "min_reward_risk": 1.20},
    "priority": {"min_net_return": 0.030, "min_reward_risk": 1.80},
    "high": {"min_net_return": 0.050, "min_reward_risk": 2.00},
}
