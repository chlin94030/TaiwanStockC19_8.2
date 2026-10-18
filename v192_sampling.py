"""Cutoff draws use signal-date eligibility and calendar maturity, never P&L.

Sampling dates is not randomizing time-series training rows. Each sampled test
is replayed chronologically and its weights use only strictly earlier outcomes.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd


def cutoff_plan(panel, calendar, cfg):
    cal = pd.DatetimeIndex(calendar).sort_values().unique()
    needed = int(cfg.dates)
    info = {'requested': needed, 'actual': 0, 'seed': int(cfg.seed),
            'universe_seed': int(cfg.universe_seed), 'method': cfg.sampling_method,
            'replacement': False, 'eligible_dates': 0, 'cutoffs': [],
            'selection_uses_future_returns': False,
            'selection_uses_future_stock_availability': False,
            'independent_trials_claimed': False, 'status': 'INSUFFICIENT_ELIGIBLE_DATES'}
    if not len(cal):
        info['reason'] = 'EMPTY_COMMON_CALENDAR'
        return [], info
    last = len(cal) - 1 - max(cfg.horizons)
    counts = panel.loc[panel.eligible].groupby('date').ticker.nunique()
    # Need historical factor warmup, not four years of independent validation
    # for every horizon. Insufficient inner training retains prior weights.
    full_start = cfg.warmup + cfg.train_dates + max(cfg.horizons) + 10
    first = min(full_start, max(cfg.warmup, last - needed * 3))
    valid = [d for i, d in enumerate(cal) if first <= i <= last
             and int(counts.get(d, 0)) >= cfg.min_universe]
    info.update(eligible_dates=len(valid), minimum_calendar_index=int(first),
                maximum_calendar_index=int(last),
                eligible_first=str(valid[0].date()) if valid else None,
                eligible_last=str(valid[-1].date()) if valid else None,
                candidate_pool_hash=hashlib.sha256('|'.join(str(x.date()) for x in valid).encode()).hexdigest())
    if cfg.replay_cutoffs:
        dates = sorted(pd.Timestamp(x) for x in cfg.replay_cutoffs)
        bad = [str(d.date()) for d in dates if d not in cal or cal.get_loc(d) > last]
        info.update(method='explicit_development_replay', invalid_replay_dates=bad)
        if bad:
            info['reason'] = 'REPLAY_OUTSIDE_AVAILABLE_CALENDAR_OR_IMMATURE'
            return [], info
    elif len(valid) < needed:
        info['reason'] = 'FEWER_THAN_REQUESTED_UNIQUE_MATURE_DATES'
        # Do not reduce 100 to 10, duplicate dates, or choose only lucky outcomes.
        return [], info
    else:
        rng = np.random.default_rng(cfg.seed)
        if cfg.sampling_method == 'stratified':
            ids = [int(rng.choice(b)) for b in np.array_split(np.arange(len(valid)), needed)]
        else:
            ids = rng.choice(len(valid), needed, replace=False)
        dates = sorted(pd.Timestamp(valid[int(i)]) for i in ids)
    assert len(dates) == needed and len(set(dates)) == needed
    info.update(actual=len(dates), status='EXACT_REQUESTED_COUNT',
                cutoffs=[str(d.date()) for d in dates])
    info['disjoint_case_count'] = {str(h): len(disjoint_dates(dates, cal, h)) for h in cfg.horizons}
    return dates, info


def disjoint_dates(dates, calendar, horizon):
    """Greedy chronological subset of non-overlapping entry-to-exit windows.

    No claim of statistical independence: market regimes still correlate.
    """
    cal = pd.DatetimeIndex(calendar)
    picked = []
    last_exit = -1
    for d in sorted(pd.Timestamp(x) for x in dates):
        loc = int(cal.get_indexer([d])[0])
        if loc >= 0 and loc > last_exit:
            picked.append(d)
            last_exit = loc + horizon
    return picked


def fit_seed(seed, date, horizon):
    # Same cutoff gets the same candidate weights even if more dates are added.
    raw = f'{seed}|{pd.Timestamp(date).date()}|{horizon}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'big')
