"""In-memory causal group/IC cache shared by the 100-cutoff replay.

Only deterministic per-date arrays are cached. No future-return feedback is
used to pick test dates or their weights; access is bounded by the caller.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from v19_config import FACTORS


class ReplayContext:
    def __init__(self, panel, cfg):
        self.groups = [(d, g.sort_values('ticker')) for d, g in
                       panel.loc[panel.eligible].groupby('date', sort=True)
                       if len(g) >= cfg.min_universe]
        self.min_universe = cfg.min_universe
        self.ics = {}
        from v193_fast import PortfolioCache
        # The ten-lens engines evaluate only their own small fixed profile set.
        # PortfolioCache otherwise allocates ALL ~50 factors x every equity x
        # every session, which can exceed memory for a 600-2000 stock universe.
        self.portfolio = None if cfg.factor_design in ('ten_angle_v197','balanced_v198') else PortfolioCache(self.groups, cfg)
        self.exit_dates = {h:[g[f'exit_date_{h}'].max() for d,g in self.groups] for h in cfg.horizons}

    def before(self, cutoff, horizon):
        # Outcomes are stamped on the same exchange calendar for every ticker.
        return [(d,g) for (d,g),end in zip(self.groups,self.exit_dates[horizon])
                if d < cutoff and pd.notna(end) and end < cutoff]

    def diagnostics(self, train_pairs, features, h):
        values = {k: [] for k in features}
        for d, g in train_pairs:
            key = (str(d), h, tuple(g.ticker))
            if key not in self.ics:
                y = pd.to_numeric(g[f'net_{h}'], errors='coerce')
                ic = {}
                for f in FACTORS:
                    pair = pd.DataFrame({'x': g[f], 'y': y}).replace([np.inf, -np.inf], np.nan).dropna()
                    if len(pair) < 10 or pair.x.nunique() < 2 or pair.y.nunique() < 2:
                        ic[f] = None; continue
                    x = pair.x.rank(method='average').to_numpy(float); v = pair.y.rank(method='average').to_numpy(float)
                    x -= x.mean(); v -= v.mean()
                    ic[f] = float(np.dot(x, v)/np.sqrt(np.dot(x, x)*np.dot(v, v)))
                self.ics[key] = ic
            for f in features:
                value = self.ics[key].get(f)
                if value is not None:
                    values[f].append(value)
        return {k: {'train_rank_ic_mean': float(np.mean(v)) if v else None,
                    'train_ic_dates': len(v),
                    'positive_ic_fraction': float(np.mean(np.array(v)>0)) if v else None,
                    'not_causal_or_test_importance': True} for k, v in values.items()}
