"""Auditable factor diagnostics on frozen evaluation dates.

Report-only: outcomes here never feed the same date's selected weights. Missing
optional fields stay missing. Top/bottom baskets are selected BEFORE checking
future returns so missing losers cannot disappear. IC uses pairwise ranks.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from v19_config import FACTORS, TECH
from v193_factors import EXTENDED
from v192_sampling import disjoint_dates

SIGNALS = ('kd_low_cross_signal', 'macd_hist_flip_signal', 'rsi_recovery_signal',
           'negative_bias_repair_signal', 'w_breakout_signal', 'volume_breakout_signal')


def attach_signals(panel):
    parts = []
    for _, g in panel.groupby('ticker', sort=False):
        g = g.sort_values('date').copy()
        g['kd_low_cross_signal'] = (g.kd_cross.eq(1) & (g.k.shift(1) < 35)).astype(float).where(g.k.shift(1).notna())
        g['macd_hist_flip_signal'] = g.macd_cross.astype(float).where(g.macd_hist.notna())
        rmin = g.rsi.rolling(7, min_periods=7).min()
        g['rsi_recovery_signal'] = ((rmin < 30) & (g.rsi > g.rsi.shift()) & (g.rsi > rmin)).astype(float).where(rmin.notna())
        g['negative_bias_repair_signal'] = ((g.bias20 < 0) & (g.bias20 > g.bias20.shift())).astype(float).where(g.bias20.shift().notna())
        g['w_breakout_signal'] = g.pattern_stage.isin(['BREAKOUT', 'RETEST']).astype(float).where(g.pattern.notna())
        g['volume_breakout_signal'] = ((g.volume_ratio > 1.2) & (g.close > g.prior_high20) & (g.close_position > .6)).astype(float).where(g.prior_high20.notna())
        parts.append(g)
    return pd.concat(parts, ignore_index=True).sort_values(['date', 'ticker']).reset_index(drop=True)


def _rank_ic(x, y):
    pair = pd.DataFrame({'x': x, 'y': y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 10 or pair.x.nunique() < 2 or pair.y.nunique() < 2:
        return None, len(pair)
    a = pair.x.rank(method='average').to_numpy(float)
    b = pair.y.rank(method='average').to_numpy(float)
    a -= a.mean(); b -= b.mean()
    return float(np.dot(a, b)/np.sqrt(np.dot(a, a)*np.dot(b, b))), len(pair)


def _basket(frame, h):
    if frame.empty or frame[f'net_{h}'].isna().any():
        return None
    bases = set(frame[f'return_basis_{h}']) - {'CASH'}
    if bases and (len(bases) != 1 or not bases.issubset({'RAW_PRICE_ONLY', 'ADJUSTED_PROVIDER'})):
        return None
    return float(frame[f'net_{h}'].mean())


def case_factors(day, date, h, cfg, weights):
    from v19_backtest import selected_result
    rows = day[day.eligible].sort_values('ticker')
    baseline, _ = selected_result(day, date, weights, h, cfg, 'audit_full')
    full = baseline['mean_net']
    basis_full = baseline.get('price_basis')
    result = []
    for factor in tuple(FACTORS)+SIGNALS:
        g = rows[rows[factor].notna()].copy() if factor in rows else rows.iloc[:0].copy()
        entry = {'date': str(pd.Timestamp(date).date()), 'horizon': h, 'factor': factor,
                 'kind': 'factor_group' if factor in FACTORS else 'binary_signal',
                 'eligible_stocks': len(rows), 'available_stocks': len(g),
                 'coverage': len(g)/len(rows) if len(rows) else 0.,
                 'active_weight': float(weights.get(factor, 0.)), 'rank_ic': None,
                 'rank_ic_pairs': 0, 'top_mean_net': None, 'bottom_mean_net': None,
                 'top_minus_bottom': None, 'ablation_removed_mean_net': None,
                 'full_minus_removed': None, 'signal_on_count': None, 'signal_off_count': None,
                 'status': 'NO_DATA_NOT_ZERO', 'report_only_not_training': True}
        if g.empty:
            if factor not in TECH+EXTENDED and factor in FACTORS and not cfg.optional_data:
                entry['status'] = 'OPTIONAL_DISABLED_NOT_TESTED'
            result.append(entry); continue
        entry['status'] = 'DESCRIPTIVE_ONLY'
        if factor in SIGNALS:
            top = g[g[factor] > 0]; bottom = g[g[factor] <= 0]
            entry['signal_on_count'] = len(top); entry['signal_off_count'] = len(bottom)
        else:
            q = max(2, int(len(g)*.2))
            ranked = g.sort_values([factor, 'ticker'], ascending=[False, True])
            top = ranked.head(q); bottom = ranked.tail(q)
        if len(g) >= 10 and g[factor].nunique() >= 2:
            # IC is descriptive complete-case analysis; count and missing outcome
            # denominator remain explicit. No fabricated returns or p-values.
            valid = g[g[f'net_{h}'].notna()]
            bases = set(valid[f'return_basis_{h}']) - {'CASH'}
            if len(bases) <= 1 and bases.issubset({'RAW_PRICE_ONLY', 'ADJUSTED_PROVIDER'}):
                entry['rank_ic'], entry['rank_ic_pairs'] = _rank_ic(valid[factor], valid[f'net_{h}'])
            if len(top) >= 2 and len(bottom) >= 2:
                tm = _basket(top, h); bm = _basket(bottom, h)
                common_basis = set(pd.concat([top, bottom])[f'return_basis_{h}'])-{'CASH'}
                entry['top_mean_net'] = tm; entry['bottom_mean_net'] = bm
                if tm is not None and bm is not None and len(common_basis) <= 1:
                    entry['top_minus_bottom'] = tm-bm
        else:
            entry['status'] = 'INSUFFICIENT_CROSS_SECTION_OR_NO_VARIATION'
        if weights.get(factor, 0.) > 0 and sum(v for k, v in weights.items() if k != factor) > 0:
            ablated = {k: v for k, v in weights.items() if k != factor}
            s = sum(ablated.values()); ablated = {k: v/s for k, v in ablated.items()}
            record, _ = selected_result(day, date, ablated, h, cfg, 'audit_remove_'+factor)
            val = record['mean_net']; entry['ablation_removed_mean_net'] = val
            if full is not None and val is not None and record.get('price_basis') == basis_full:
                entry['full_minus_removed'] = full-val
        entry['outcomes_available'] = int(g[f'net_{h}'].notna().sum())
        result.append(entry)
    return result


def summary(rows, calendar, cfg):
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    output = []
    for (h, factor), g in df.groupby(['horizon', 'factor'], sort=True):
        g = g.sort_values('date')
        def values(col):
            return pd.to_numeric(g[col], errors='coerce').dropna()
        ic = values('rank_ic'); spread = values('top_minus_bottom'); delta = values('full_minus_removed')
        # Select a disjoint subset solely by calendar, before checking validity.
        chosen = set(str(d.date()) for d in disjoint_dates(g.date, calendar, int(h)))
        dis = g[g.date.isin(chosen)]
        dis_ic = pd.to_numeric(dis.rank_ic, errors='coerce').dropna()
        dis_delta = pd.to_numeric(dis.full_minus_removed, errors='coerce').dropna()
        coverage = float(g.coverage.mean())
        if coverage == 0:
            decision = 'NOT_TESTED_NO_DATA'
        elif not len(ic) and not len(delta):
            decision = 'INSUFFICIENT_VARIATION_OR_COMPARABLE_OUTCOMES'
        elif len(delta) and delta.mean() > 0:
            decision = 'POSITIVE_MARGINAL_DIAGNOSTIC_NOT_APPROVED'
        else:
            decision = 'REVIEW_IN_TRAINING_ONLY_NOT_AUTO_REWEIGHT'
        output.append({'horizon': int(h), 'factor': factor, 'kind': str(g.kind.iloc[0]),
            'sampled_cases': len(g), 'mean_coverage': coverage,
            'ic_valid_cases': len(ic), 'rank_ic_mean': float(ic.mean()) if len(ic) else None,
            'positive_ic_fraction': float((ic > 0).mean()) if len(ic) else None,
            'spread_valid_cases': len(spread), 'top_minus_bottom_mean': float(spread.mean()) if len(spread) else None,
            'ablation_paired_cases': len(delta), 'full_minus_removed_mean': float(delta.mean()) if len(delta) else None,
            'ablation_positive_fraction': float((delta > 0).mean()) if len(delta) else None,
            'disjoint_calendar_cases': len(dis), 'disjoint_ic_cases': len(dis_ic),
            'disjoint_rank_ic_mean': float(dis_ic.mean()) if len(dis_ic) else None,
            'disjoint_ablation_cases': len(dis_delta),
            'disjoint_full_minus_removed': float(dis_delta.mean()) if len(dis_delta) else None,
            'status': decision, 'auto_promoted_to_live_weights': False})
    return output


def markdown(report):
    plan = report.get('sampling_plan', {})
    out = ['## Random-date factor validation', '',
           f"Requested {plan.get('requested')}; selected {plan.get('actual')}; "
           f"eligible pool {plan.get('eligible_dates')}; method {plan.get('method')}; seed {plan.get('seed')}.",
           f"Resumed case/horizon checkpoints: {report.get('resumed_cases', 0)}.",
           'Date draws are without replacement; holding windows can overlap. These are not 100 independent trials.',
           'Cutoff membership never depends on future stock returns or future missing-price flags.',
           'IC is rank association, NOT return or probability. Binary-signal tests are distinct from factor-group weights.',
           'Ablation delta = full model minus model with that factor removed, on matching cases. Not a causal effect.',
           'Disjoint subset is calendar-selected; correlation may remain. No multiple-testing-adjusted significance is claimed.',
           '', '| Horizon | Factor | Coverage | IC cases | Mean IC | Ablation pairs | Full minus removed | Disjoint cases | Status |',
           '|---|---|---:|---:|---:|---:|---:|---:|---|']
    def num(x, pct=False):
        if x is None or not np.isfinite(x): return '--'
        return f'{x*100:+.2f}%' if pct else f'{x:+.3f}'
    for r in report.get('factor_summary', []):
        out.append(f"| {r['horizon']} | {r['factor']} | {r['mean_coverage']:.0%} | {r['ic_valid_cases']} | "
                   f"{num(r['rank_ic_mean'])} | {r['ablation_paired_cases']} | {num(r['full_minus_removed_mean'], True)} | "
                   f"{r['disjoint_calendar_cases']} | {r['status']} |")
    return '\n'.join(out)
