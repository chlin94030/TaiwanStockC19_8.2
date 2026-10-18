"""V18.20 limit-up *research* shadow lane: completed-session, PIT and no BUY.

This computes a relative pre-limit setup score (0-100), NEVER a calibrated
probability of touching the exchange-defined upper price limit.  Actual event
labels require official, unadjusted, per-session upper-limit prices.  A five-day
outcome must not be used to change the score at the decision cutoff.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import math
import numpy as np
import pandas as pd

LAB_VERSION = 'v18.20-limitup-shadow-20261014-provisional'
# Explicit exploratory hypotheses, NOT fitted to the 10/02 -> 10/13 outcomes.
SIGNAL_WEIGHTS = {
    'breakout': 0.23,       # prior 20-session resistance, not today's high
    'volume': 0.19,         # last/5d turnover expansion vs preceding 20 days
    'relative_accel': 0.18,# early acceleration vs same-date broad index
    'compression': 0.16,    # volatility contraction with constructive setup
    'sector_breadth': 0.10, # same-date evidence or explicitly missing
    'candle': 0.07,         # last completed candle close strength
    'trend': 0.07,          # MA structure without 20MA overextension
}
BLEND_WEIGHTS = {'baseline_alpha': 0.40, 'limitup_setup': 0.60}
LIMITUP_POLICY = {
    'min_avg_turnover_twd': 30_000_000.0,
    'min_alpha': 60.0,
    'min_setup': 66.0,
    'min_evidence_coverage': 0.75,
    'max_bias_ma20_pct': 12.0,
    'max_daily_change_pct': 7.0,
    'max_candidate_n': 5,
}
assert abs(sum(SIGNAL_WEIGHTS.values()) - 1) < 1e-12
assert abs(sum(BLEND_WEIGHTS.values()) - 1) < 1e-12


def _finite(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def _cap(value):
    return float(np.clip(value, 0, 100))


def _entry(value, note):
    return {'score': round(_cap(value), 1) if value is not None else None, 'reason': note,
            'source': '同日已完成之日線（非盤中）'}


def price_setup(df: pd.DataFrame, *, benchmark_ret5=None, benchmark_ret20=None) -> dict:
    """Use only df rows up through the actual decision date. No future queries.

    Benchmark returns MUST be from the identical EOD cutoff; pass None if its
    official date is missing/mismatched. Missing RS is never imputed as neutral.
    """
    required = ('Open','High','Low','Close','Volume')
    if df is None or len(df) < 145 or not all(c in df for c in required):
        return {'available': False, 'reason': '需要至少145筆完整OHLCV日線'}
    px = df.loc[:, required].tail(145).apply(pd.to_numeric, errors='coerce')
    if px.isna().any().any() or (px[['Open','High','Low','Close']] <= 0).any().any():
        return {'available': False, 'reason': '日線有缺值或非正股價'}
    if (px['Volume'] < 0).any() or (px['High'] < px[['Open','Low','Close']].max(axis=1)).any() or (px['Low'] > px[['Open','Close']].min(axis=1)).any():
        return {'available': False, 'reason': 'OHLCV行情格式不一致'}
    close, open_, high, low, volume = (px[c] for c in required)
    current = float(close.iloc[-1]); previous = float(close.iloc[-2])
    ma5 = float(close.iloc[-5:].mean()); ma20 = float(close.iloc[-20:].mean()); ma60 = float(close.iloc[-60:].mean())
    r5 = current / float(close.iloc[-6]) - 1.0
    r20 = current / float(close.iloc[-21]) - 1.0
    day_change = (current / previous - 1.0) * 100
    bias = (current / ma20 - 1.0) * 100
    prior20_high = float(high.iloc[-21:-1].max())
    near_break_pct = (current / prior20_high - 1) * 100
    # Max score occurs around resistance: penalize both too far below and a
    # highly extended breakout; no hindsight on following-day highs.
    if near_break_pct < -8:
        breakout = 15.0
    elif near_break_pct < -3:
        breakout = 15.0 + (near_break_pct + 8) * 9.0
    elif near_break_pct <= 1.5:
        breakout = 60.0 + (near_break_pct + 3) * (34.0 / 4.5)
    else:
        breakout = 94.0 - (near_break_pct - 1.5) * 12.0
    previous20_mean = float(volume.iloc[-21:-1].mean())
    volume_ratio = float(volume.iloc[-1] / previous20_mean) if previous20_mean > 0 else None
    vol5_ratio = float(volume.iloc[-5:].mean() / previous20_mean) if previous20_mean > 0 else None
    if volume_ratio is None:
        volume_score = None
    else:
        volume_score = 50 + 23 * math.log2(max(0.05, volume_ratio)) + 12 * math.log2(max(0.05, vol5_ratio))
        if volume_ratio > 4.5:
            volume_score -= 20  # blow-off volume, not infinite positive signal
    b5 = _finite(benchmark_ret5); b20 = _finite(benchmark_ret20)
    rs_score = None
    if b5 is not None and b20 is not None:
        rs5 = r5 - b5
        rs20 = r20 - b20
        accel = rs5 - rs20 / 4
        rs_score = 50 + 700 * rs5 + 220 * accel
    mean20 = close.rolling(20).mean(); std20 = close.rolling(20).std(ddof=1)
    widths = (4 * std20 / mean20.replace(0, np.nan)).dropna().tail(121)
    width_percentile = None
    if len(widths) >= 85:
        width_percentile = float((widths <= widths.iloc[-1]).mean())
        compression = 90.0 - 65.0 * width_percentile
        if current < ma20 or r5 <= 0:
            compression = min(compression, 48.0)  # falling knife squeeze is not bullish
    else:
        compression = None
    span = float(high.iloc[-1] - low.iloc[-1])
    candle = None if span <= 0 else (23 + 62 * float((current-low.iloc[-1])/span) + 15*np.clip((current/open_.iloc[-1]-1)*35, -1, 1))
    trend = 34 + (24 if current > ma20 else -12) + (18 if ma5 > ma20 else -8) + (18 if ma20 > ma60 else -10)
    if bias > 9:
        trend -= (bias - 9) * 6
    if bias < -6:
        trend -= 12
    factors = {
        'breakout': _entry(breakout, f'距前20日最高價 {near_break_pct:+.1f}%'),
        'volume': _entry(volume_score, f'單日量比 {volume_ratio:.2f}／近5日 {vol5_ratio:.2f}' if volume_ratio is not None else '近期量能基準不足'),
        'relative_accel': _entry(rs_score, '已同步大盤之5日相對強度加速' if rs_score is not None else '大盤日期或5日／20日報酬不足'),
        'compression': _entry(compression, '布林寬度120窗歷史分位' if compression is not None else '歷史波動窗不足'),
        'candle': _entry(candle, '最後一根完整日K收盤位置' if candle is not None else '一字K不推估K棒位置'),
        'trend': _entry(trend, '5/20/60日線結構與過熱程度'),
    }
    return {
        'available': True, 'version': LAB_VERSION,
        'price_date': str(df.index[-1].date()) if isinstance(df.index, pd.DatetimeIndex) else str(df.index[-1])[:10],
        'factors': factors,
        'metrics': {
            'last_close': round(current, 3), 'ret5_pct': round(r5*100, 2),
            'day_change_pct': round(day_change,2), 'vol_ratio': round(volume_ratio,2) if volume_ratio is not None else None,
            'distance_prior20_high_pct': round(near_break_pct,2),
            'bias_ma20_pct': round(bias,2), 'bb_width_percentile': round(width_percentile,3) if width_percentile is not None else None,
        },
    }


def finalize_setup(raw: dict, *, sector: Mapping | None=None) -> dict:
    """Complete same-day peer signal without replacing data from another day."""
    if not isinstance(raw, dict) or not raw.get('available'):
        return {'available': False, 'reason': (raw or {}).get('reason', '資料不足') if isinstance(raw,dict) else '資料不足'}
    sector = sector if isinstance(sector, Mapping) else {}
    result = {**raw, 'factors': dict(raw['factors'])}
    breadth = _finite(sector.get('peers_above_ma20_pct'))
    if (sector.get('available') and str(sector.get('date') or '') == str(raw.get('price_date') or '')
            and (sector.get('peer_count') or 0) >= 3 and breadth is not None):
        result['factors']['sector_breadth'] = _entry(50 + (breadth - 50) * .8, f'同日同群強勢比例 {breadth:.0f}%')
    else:
        result['factors']['sector_breadth'] = _entry(None, '缺少同日同業可比資料，不以零分或虛構分數替代')
    points = 0.0; observed = 0.0
    for key, weight in SIGNAL_WEIGHTS.items():
        score = _finite((result['factors'].get(key) or {}).get('score'))
        if score is not None:
            points += score * weight
            observed += weight
        else:
            points += 50.0 * weight  # fair neutral with coverage penalty
    points -= (1 - observed) * 8  # modest explicit missing-evidence penalty
    # Existing strong breakout today may already be unbuyable tomorrow; preserve
    # setup data and penalize *research priority*, not its original source data.
    metrics = result['metrics']; bias = metrics.get('bias_ma20_pct', 0)
    if bias > 12:
        points -= min(18.0, (bias-12)*1.5)
    if metrics.get('day_change_pct', 0) > 6:
        points -= min(16.0, (metrics['day_change_pct']-6)*5)
    result.update(setup_score=round(_cap(points),1), evidence_coverage=round(observed,3),
                  is_calibrated_probability=False, is_buy_signal=False,
                  interpretation='漲停前兆相對排序（研究假說，非漲停機率）')
    return result


def decorate_candidate(stock: dict, raw: dict, sector: Mapping | None = None) -> dict:
    """Attach shadow ranking, independent of production Alpha and trades."""
    s = finalize_setup(raw,sector=sector)
    result = {'version':LAB_VERSION,'is_buy_signal':False, 'setup':s}
    alpha = _finite(((stock.get('horizons') or {}).get('short') or {}).get('ranking_score'))
    score = _finite(s.get('setup_score'))
    result['alpha'] = round(alpha,1) if alpha is not None else None
    result['research_rank_score'] = round(alpha*BLEND_WEIGHTS['baseline_alpha']+score*BLEND_WEIGHTS['limitup_setup'],1) if alpha is not None and score is not None else None
    m = s.get('metrics') or {}
    blockers=[]
    if not s.get('available'): blockers.append('日線資料不足')
    if str(s.get('price_date') or '') != str(stock.get('price_date') or ''):blockers.append('股票日線日期不一致')
    if alpha is None or alpha < LIMITUP_POLICY['min_alpha']:blockers.append('Alpha研究門檻')
    if score is None or score < LIMITUP_POLICY['min_setup']:blockers.append('前兆分數不足')
    if _finite(stock.get('avg_turnover_20d')) is None or float(stock.get('avg_turnover_20d')) < LIMITUP_POLICY['min_avg_turnover_twd']:blockers.append('成交金額不足')
    if float(s.get('evidence_coverage') or 0) < LIMITUP_POLICY['min_evidence_coverage']:blockers.append('資料覆蓋不足')
    if (_finite(m.get('vol_ratio')) is None or _finite(m.get('vol_ratio')) <= 0):blockers.append('當日無有效成交量')
    if (_finite(m.get('bias_ma20_pct')) or 0) > LIMITUP_POLICY['max_bias_ma20_pct']:blockers.append('均線乖離過熱')
    if (_finite(m.get('day_change_pct')) or 0) > LIMITUP_POLICY['max_daily_change_pct']:blockers.append('當日急漲不可追高')
    # A separate *selection* evidence check: exclude stocks whose core data is
    # not eligible even though the technical setup looks attractive.
    eligibility = ((stock.get('selection_quality') or {}).get('short') or {})
    if eligibility.get('eligible') is False:blockers.append('原模型資料／流動性不合格')
    result['eligible_research'] = not blockers
    result['blockers'] = blockers
    return result


def shadow_shortlist(snapshot: dict|None, n: int=5) -> list[dict]:
    """Never backfill earlier missing profiles from current data. No forced Top5."""
    if not isinstance(snapshot,dict):return []
    date = str(snapshot.get('price_date') or '')
    rows=[]
    for stock in snapshot.get('stocks') or []:
        sh=stock.get('limitup_shadow') or {}
        meta=sh.get('setup') or {}
        if sh.get('version') != LAB_VERSION or not sh.get('eligible_research') or meta.get('price_date') != date:
            continue
        rank=_finite(sh.get('research_rank_score'))
        if rank is None:continue
        rows.append({'ticker':stock.get('ticker'),'name':stock.get('name'),
                     'price_date':date,'alpha':sh.get('alpha'),
                     'setup_score':meta.get('setup_score'),
                     'research_rank_score':rank,
                     'coverage':meta.get('evidence_coverage'),
                     'metrics':meta.get('metrics'),
                     'is_buy_signal':False})
    rows.sort(key=lambda r:(-r['research_rank_score'],str(r['ticker'])))
    return rows[:max(0,min(n,20))]


def classify_official_limit_events(*, future_raw: pd.DataFrame, upper_limits: pd.Series|None,
                                   sessions: int =5, ipo_exempt_dates: set|None=None) -> dict:
    """Label after t using per-day OFFICIAL upper limit (never 9.5% guess).

    The caller must supply *only* t+1..t+sessions trading days, unadjusted High,
    exact exchange price-limits, and an independent verified listing archive.
    Corporate-action adjusted candles cannot be substituted here.  Missing a
    limit on any day -> NOT_VERIFIED; no partial negatives fabricated.
    """
    if (not isinstance(future_raw,pd.DataFrame) or len(future_raw) < sessions
        or upper_limits is None or not isinstance(upper_limits,pd.Series)):
        return {'status':'NOT_VERIFIED','reason':'缺少完整未來交易日與官方每日漲停價'}
    rows = future_raw.head(sessions)
    missing = rows.index.difference(upper_limits.index)
    if len(missing) or 'High' not in rows or rows['High'].isna().any():
        return {'status':'NOT_VERIFIED','reason':'後續日高價／每日漲停價不完整'}
    if ipo_exempt_dates is None:
        return {'status':'NOT_VERIFIED','reason':'缺少新掛牌漲跌幅例外核對'}
    if any(d in ipo_exempt_dates for d in rows.index):
        return {'status':'EXCLUDED','reason':'當日漲跌幅制度例外，排除'}
    limits=upper_limits.loc[rows.index].astype(float)
    highs=rows['High'].astype(float)
    if not np.isfinite(limits).all() or not np.isfinite(highs).all() or (limits<=0).any():
        return {'status':'NOT_VERIFIED','reason':'官方漲停價有非有限或不正值'}
    # Valid official unadjusted High is always <= that day's upper bound.
    if ((highs-limits)>np.maximum(0.001,limits*.0001)).any():
        return {'status':'NOT_VERIFIED','reason':'日高價超過官方漲停價，來源／除權還原有誤'}
    touched=bool((highs>=limits-np.maximum(0.001,limits*.0001)).any())
    return {'status':'VERIFIED','touch_limit_5d':touched,'sessions':sessions,
            'event_dates':[str(d.date()) if hasattr(d,'date') else str(d) for d in rows.index[highs>=limits-np.maximum(0.001,limits*.0001)]]}
