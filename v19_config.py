"""Reproducible V19 research configuration. No order execution."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json

VERSION = '19.8'
TZ = timezone(timedelta(hours=8))
TECH = ('trend', 'relative_strength', 'breakout', 'reversal', 'pattern', 'volume')
OPTIONAL = ('revenue', 'quality', 'foreign', 'trust', 'joint', 'dealer', 'margin', 'holders')
from v193_factors import EXTENDED
from v194_factors import NEW_FACTORS
from v196_factors import NEW_FACTORS as SCORECARD_FACTORS
from v197_ten_pillars import PILLARS
FACTORS = TECH + EXTENDED + OPTIONAL + NEW_FACTORS + SCORECARD_FACTORS + PILLARS
# Starting hypotheses, NOT optimized coefficients. Missing optional groups are
# excluded from a common training universe rather than being invented as zero.
PRIORS = {
    5: dict(trend=12, relative_strength=12, breakout=16, reversal=14, pattern=10, volume=12,
            revenue=7, quality=3, foreign=3, trust=5, joint=3, dealer=1, margin=1, holders=1),
    20: dict(trend=16, relative_strength=14, breakout=10, reversal=10, pattern=8, volume=8,
             revenue=12, quality=8, foreign=4, trust=4, joint=3, dealer=1, margin=1, holders=1),
    60: dict(trend=18, relative_strength=12, breakout=7, reversal=5, pattern=6, volume=5,
             revenue=16, quality=16, foreign=4, trust=4, joint=3, dealer=1, margin=1, holders=2),
}

@dataclass(frozen=True)
class Settings:
    years: int = 3
    stocks: int = 600
    dates: int = 100
    horizons: tuple[int, ...] = (5, 20, 60)
    trials: int = 24
    seed: int = 1920
    top_k: int = 5
    warmup: int = 80
    train_dates: int = 35
    min_universe: int = 10
    min_turnover: float = 5_000_000.0
    max_entry_gap: float = 0.04
    buy_fee: float = 0.001425
    sell_fee: float = 0.001425
    sell_tax: float = 0.003
    benchmark_sell_tax: float = 0.001
    slippage: float = 0.0005
    optional_data: bool = False
    request_budget: int = 900
    timeout_seconds: int = 12
    benchmark: str = '0050.TW'
    as_of: str = ''
    # Governance preferences, not empirically optimized coefficients.
    prefer_adjusted: bool = True
    validation_blocks: int = 4
    improvement_floor: float = 0.002
    weight_shrinkage: float = 0.5
    max_mae_deterioration: float = 0.01
    min_validation_fill: float = 0.6
    replay_cutoffs: tuple[str, ...] = ()
    evaluation_role: str = 'DEVELOPMENT_REPLAY_NOT_NEW_HOLDOUT'
    sampling_method: str = 'uniform'
    universe_seed: int = 1900
    refresh_history: bool = False
    factor_audit: bool = True
    download_workers: int = 3
    network_retries: int = 1
    factor_design: str = 'balanced_v198'
    universe_mode: str = 'balanced'
    min_download_coverage: float = 0.80
    tournament_blocks: int = 8
    tournament_uncertainty: float = 0.5
    sparse_max_inputs: int = 6
    sparse_budget: int = 160
    sparse_beam: int = 3
    sparse_cv_folds: int = 3
    sparse_complexity: float = 0.0005
    sparse_near_best: float = 0.001
    sparse_ridge: float = 0.02
    sparse_corr_limit: float = 0.95
    sparse_fit_coverage: float = 0.85
    financial_history: bool = True
    financial_timing: str = 'disclosure_required'
    statement_lag_days: int = 120



    def __post_init__(self):
        if not 4 <= self.tournament_blocks <= 20: raise ValueError("tournament_blocks must be 4..20")
        if not 0 <= self.tournament_uncertainty <= 3: raise ValueError("invalid tournament_uncertainty")
        if not 1 <= self.sparse_max_inputs <= 6: raise ValueError('sparse_max_inputs must be 1..6')
        if not 16 <= self.sparse_budget <= 512: raise ValueError('sparse_budget must be 16..512')
        if not 1 <= self.sparse_beam <= 5: raise ValueError('sparse_beam must be 1..5')
        if not 2 <= self.sparse_cv_folds <= 5: raise ValueError('sparse_cv_folds must be 2..5')
        if not 0 <= self.sparse_complexity <= .02: raise ValueError('invalid sparse_complexity')
        if not 0 <= self.sparse_near_best <= .02: raise ValueError('invalid sparse_near_best')
        if not 0 < self.sparse_ridge <= 10: raise ValueError('invalid sparse_ridge')
        if not .5 <= self.sparse_corr_limit <= 1: raise ValueError('invalid sparse_corr_limit')
        if not .5 <= self.sparse_fit_coverage <= 1: raise ValueError('invalid sparse_fit_coverage')
        if self.financial_timing not in ('disclosure_required','lagged_research'): raise ValueError('invalid financial_timing')
        if not 90 <= self.statement_lag_days <= 365: raise ValueError('statement_lag_days must be 90..365')
        if self.universe_mode not in ('balanced','proportional','full_current','legacy_random'):raise ValueError('invalid universe_mode')
        if not 0 < self.min_download_coverage <= 1:raise ValueError('invalid min_download_coverage')
        if self.factor_design not in ('legacy_v192','expanded_v193','sparse_v194','sparse_v195','tournament_v196','ten_angle_v197','balanced_v198'):raise ValueError('invalid factor_design')
        if self.sampling_method not in ('uniform', 'stratified'): raise ValueError('invalid sampling_method')
        if not 1 <= self.download_workers <= 4: raise ValueError('download_workers must be 1..4')
        if not 0 <= self.network_retries <= 2: raise ValueError('network_retries must be 0..2')
        if not 0 <= self.seed <= 2**32-1 or not 0 <= self.universe_seed <= 2**32-1: raise ValueError('seed out of range')
        if not 1 <= self.years <= 3: raise ValueError('years must be 1..3')
        if self.stocks < self.top_k or self.top_k < 1: raise ValueError('invalid stock count')
        if not 2 <= self.dates <= 100: raise ValueError('dates must be 2..100')
        if not self.horizons or any(h < 1 or h > 250 for h in self.horizons): raise ValueError('invalid horizons')
        if self.trials < 2 or self.trials > 100: raise ValueError('trials must be 2..100')
        if self.min_universe < self.top_k: raise ValueError('min_universe below top_k')
        if any(x < 0 or x >= .2 for x in (self.buy_fee, self.sell_fee, self.sell_tax, self.benchmark_sell_tax, self.slippage)):
            raise ValueError('invalid transaction cost')
        if self.as_of: datetime.strptime(self.as_of, '%Y-%m-%d')
        if not 4 <= self.validation_blocks <= 20: raise ValueError('validation_blocks must be 4..20')
        if not 0 < self.weight_shrinkage <= 1: raise ValueError('weight_shrinkage must be (0, 1]')
        if not 0 <= self.improvement_floor < .1: raise ValueError('invalid improvement_floor')
        if not 0 <= self.max_mae_deterioration < .2: raise ValueError('invalid max_mae_deterioration')
        if not 0 < self.min_validation_fill <= 1: raise ValueError('invalid min_validation_fill')
        if self.evaluation_role not in ('DEVELOPMENT_REPLAY_NOT_NEW_HOLDOUT','FROZEN_PROTOCOL_NEW_HOLDOUT'):
            raise ValueError('invalid evaluation_role')
        if self.replay_cutoffs:
            stamps = [datetime.strptime(x, '%Y-%m-%d') for x in self.replay_cutoffs]
            if len(set(stamps)) != len(stamps) or len(stamps) != self.dates:
                raise ValueError('replay_cutoffs must contain exactly dates unique dates')


    def end(self):
        if self.as_of: return self.as_of
        # Never assume an unfinished same-day candle is a completed daily bar.
        return (datetime.now(TZ).date() - timedelta(days=1)).isoformat()

    def start(self):
        return (datetime.strptime(self.end(), '%Y-%m-%d') - timedelta(days=365*self.years)).date().isoformat()

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def prior(horizon, factors=FACTORS):
    source = PRIORS[min(PRIORS, key=lambda h: abs(h-horizon))]
    selected = {k: float(source.get(k, 0)) for k in factors}
    total = sum(selected.values())
    if total <= 0: raise ValueError('no usable factors')
    return {k: v / total for k, v in selected.items()}
