"""Atomic signals and bounded pair interactions for V19.4.

No future price, target, factor-audit result or fitted weight is read here.
Prices alone are not earnings forecasts or institutional order-flow evidence.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ATOMIC = ('kd_turn', 'rsi_rebound', 'macd_turn', 'bias_repair')
BUSINESS = ('earnings_growth', 'gross_profit_quality', 'earnings_yield',
            'cash_conversion', 'balance_safety')
INTERACTIONS = {
    'growth_pullback_pair': ('revenue', 'orderly_pullback'),
    'flow_breakout_pair': ('trust', 'breakout'),
    'rebound_accumulation_pair': ('rsi_rebound', 'accumulation'),
}
NEW_FACTORS = ATOMIC + BUSINESS + tuple(INTERACTIONS)
CONFLICTS = {'reversal': set(ATOMIC), 'quality': {'earnings_growth', 'gross_profit_quality'}}
BUSINESS_RAW = ('pe_ratio', 'cash_conversion_ratio', 'debt_ratio')
LABELS = {
    'kd_turn': '\u004b\u0044\u4f4e\u6a94\u8f49\u5f37',
    'rsi_rebound': '\u0052\u0053\u0049\u8d85\u8ce3\u5f8c\u56de\u5347',
    'macd_turn': '\u004d\u0041\u0043\u0044\u52d5\u80fd\u6539\u5584',
    'bias_repair': '\u8ca0\u4e56\u96e2\u4fee\u5fa9',
    'earnings_growth': '\u5df2\u516c\u5e03\u76c8\u9918\u6210\u9577',
    'gross_profit_quality': '\u6bdb\u5229\u7387\u54c1\u8cea',
    'earnings_yield': '\u76c8\u9918\u6b96\u5229\u7387',
    'cash_conversion': '\u71df\u696d\u73fe\u91d1\u6d41\u8f49\u63db',
    'balance_safety': '\u8cc7\u7522\u8ca0\u50b5\u7d50\u69cb',
    'growth_pullback_pair': '\u71df\u6536\u6210\u9577\u00d7\u6709\u5e8f\u56de\u6a94',
    'flow_breakout_pair': '\u6295\u4fe1\u8cb7\u76e4\u00d7\u7a81\u7834',
    'rebound_accumulation_pair': '\u0052\u0053\u0049\u4fee\u5fa9\u00d7\u50f9\u91cf\u627f\u63a5',
}


def input_support(keys):
    """Count underlying factor inputs; an interaction does not count as one."""
    result = set()
    for k in keys:
        result.update(INTERACTIONS.get(k, (k,)))
    return result


def compatible(keys):
    selected = set(keys)
    for parent, atoms in CONFLICTS.items():
        if parent in selected and selected.intersection(atoms):
            return False
    # Avoid simultaneously counting an interaction and its own component.
    for name, parents in INTERACTIONS.items():
        if name in selected and selected.intersection(parents):
            return False
    return True


def _number(f, key):
    return pd.to_numeric(f.get(key, pd.Series(np.nan, index=f.index)), errors='coerce')


def add_atomic(panel, cfg):
    pieces = []
    for _, g in panel.groupby('ticker', sort=False):
        g = g.sort_values('date').copy()
        k = g.k; d = g.d; rr = g.rsi; hist = g.macd_hist; bias = g.bias20
        lowk = k.shift().rolling(5, min_periods=5).min()
        cross = (k > d) & (k.shift() <= d.shift())
        g['kd_turn'] = (25 + 40*((lowk < 35) & cross) +
                        20*((lowk < 35) & (k > d) & (k > k.shift()))).astype(float)
        g.loc[lowk.isna() | d.shift().isna(), 'kd_turn'] = np.nan
        lowr = rr.shift().rolling(7, min_periods=7).min()
        g['rsi_rebound'] = (25 + ((rr-lowr).clip(0, 18)*3).where(
            (lowr < 38) & (rr > rr.shift()), 0)).clip(0, 100)
        g.loc[lowr.isna() | rr.isna(), 'rsi_rebound'] = np.nan
        # Histogram change normalized by observable ATR, not stock price level.
        scale = (g.atr * g.calc_close/g.close).replace(0, np.nan)
        turn = (hist.diff()/scale).clip(-.5, .5)
        g['macd_turn'] = (45 + 60*turn + 25*((hist > 0) & (hist.shift() <= 0))).clip(0, 100)
        repair = (bias < 0) & (bias > -20) & (bias > bias.shift())
        g['bias_repair'] = (25 + 30*repair +
                              2*(bias-bias.shift()).clip(0, 10)*repair).clip(0, 100)
        g.loc[bias.shift().isna(), 'bias_repair'] = np.nan
        eps = _number(g, 'eps_growth'); margin = _number(g, 'gross_margin')
        pe = _number(g, 'pe_ratio'); cash = _number(g, 'cash_conversion_ratio')
        debt = _number(g, 'debt_ratio')
        g['earnings_growth'] = (50 + .5*eps).clip(0, 100).where(eps.notna())
        g['gross_profit_quality'] = (margin*1.6).clip(0, 100).where(margin.notna())
        g['earnings_yield'] = (1000/pe).clip(0, 100).where(pe > 0)
        g['cash_conversion'] = (50*cash).clip(0, 100).where(cash.notna())
        g['balance_safety'] = (100-100*debt).clip(0, 100).where(debt.between(0, 2))
        pieces.append(g)
    f = pd.concat(pieces, ignore_index=True).sort_values(['date','ticker']).reset_index(drop=True)
    for key, parents in INTERACTIONS.items():
        a = f.reindex(columns=list(parents))
        f[key] = a.min(axis=1).where(a.notna().all(axis=1))
    for key in NEW_FACTORS:
        f[key] = pd.to_numeric(f[key], errors='coerce').replace([np.inf,-np.inf],np.nan).clip(0,100)
    return f
