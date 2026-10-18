"""Causal broader factor families. Research scores, not buy probabilities.

All rolling windows end at the signal date. Cross-sectional percentiles use
only eligible equities observable on that date. Current sector names are
not historical evidence. No price path after the signal may enter a feature.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EXTENDED = ('trend_efficiency', 'momentum_persistence', 'accumulation',
            'downside_resilience', 'liquidity_quality', 'orderly_pullback',
            'squeeze_release', 'market_resilience')
FAMILIES = {
    'momentum': ('trend', 'relative_strength', 'trend_efficiency', 'momentum_persistence'),
    'structure': ('breakout', 'pattern', 'reversal', 'orderly_pullback', 'squeeze_release'),
    'participation': ('volume', 'accumulation', 'foreign', 'trust', 'joint', 'dealer', 'margin', 'holders'),
    'defense': ('downside_resilience', 'liquidity_quality', 'market_resilience'),
    'business': ('revenue', 'quality'),
}
FAMILY_CAPS = {'momentum': .40, 'structure': .45, 'participation': .30,
               'defense': .35, 'business': .40}
# Hand-specified research priors, frozen before this replay; no fitted claims.
EXPANDED_PRIORS = {
    5: dict(trend=6, relative_strength=4, breakout=14, reversal=8, pattern=10, volume=6,
        trend_efficiency=4, momentum_persistence=6, accumulation=10, downside_resilience=8,
        liquidity_quality=4, orderly_pullback=8, squeeze_release=8, market_resilience=4),
    20: dict(trend=8, relative_strength=6, breakout=8, reversal=4, pattern=10, volume=4,
        trend_efficiency=10, momentum_persistence=12, accumulation=10, downside_resilience=8,
        liquidity_quality=4, orderly_pullback=8, squeeze_release=4, market_resilience=4),
    60: dict(trend=6, relative_strength=4, breakout=6, reversal=2, pattern=6, volume=2,
        trend_efficiency=16, momentum_persistence=18, accumulation=10, downside_resilience=12,
        liquidity_quality=6, orderly_pullback=4, squeeze_release=2, market_resilience=6),
}


def normalize(weights):
    w={k:max(0.,float(v)) for k,v in weights.items()}
    total=sum(w.values())
    if not np.isfinite(total) or total <= 0:raise ValueError('No positive finite weights')
    return {k:v/total for k,v in w.items()}


def cap_families(weights):
    """Project family totals and within-family weights without changing signs."""
    w=normalize(weights)
    # Water-fill excess from over-budget families to the remaining capacity.
    sums={f:sum(w.get(k,0.) for k in ks) for f,ks in FAMILIES.items()}
    target=dict(sums)
    for _ in range(20):
        excess=0.
        for f in target:
            if target[f]>FAMILY_CAPS[f]:
                excess+=target[f]-FAMILY_CAPS[f];target[f]=FAMILY_CAPS[f]
        if excess<1e-12:break
        room={f:max(0.,FAMILY_CAPS[f]-target[f]) for f in target if sums[f]>0}
        available=sum(room.values())
        if available<=1e-12:break
        for f in room:target[f]+=excess*room[f]/available
    for f,ks in FAMILIES.items():
        if sums[f]>0:
            for k in ks:
                if k in w:w[k]*=target[f]/sums[f]
    return normalize(w)


def expanded_prior(horizon, features):
    source=EXPANDED_PRIORS[min(EXPANDED_PRIORS,key=lambda h:abs(h-horizon))]
    extras={'revenue':10.,'quality':10.,'foreign':4.,'trust':4.,'joint':2.,
            'dealer':1.,'margin':1.,'holders':2.}
    return cap_families({k:float(source.get(k,extras.get(k,0.))) for k in features})


def _roll_price_features(g):
    g=g.sort_values('date').copy()
    c=g.calc_close;hi=g.calc_high;lo=g.calc_low;o=g.calc_open;v=g.share_volume
    r=c.pct_change(fill_method=None)
    travel=c.diff().abs().rolling(60,min_periods=60).sum().replace(0,np.nan)
    efficiency=(c-c.shift(60))/travel
    g['trend_efficiency']=(50+100*efficiency).clip(0,100)
    # Exclude the recent five sessions rather than chasing the same short spike.
    vol=r.rolling(60,min_periods=50).std().replace(0,np.nan)
    g['_momentum_persistence']=(c.shift(5)/c.shift(65)-1)/(vol*np.sqrt(60))
    width=(hi-lo).replace(0,np.nan)
    close_location=((2*c-hi-lo)/width).where(width.notna(),0.).clip(-1,1)
    cmf=(close_location*v).rolling(20,min_periods=20).sum()/v.rolling(20,min_periods=20).sum().replace(0,np.nan)
    signed=(np.sign(r)*v).rolling(20,min_periods=20).sum()/v.rolling(20,min_periods=20).sum().replace(0,np.nan)
    g['money_flow20']=cmf
    g['accumulation']=(50+70*cmf+20*signed).clip(0,100)
    down=np.sqrt(r.clip(upper=0).pow(2).rolling(60,min_periods=50).mean())
    dd=c/c.rolling(60,min_periods=50).max()-1
    g['downside_vol60']=down;g['drawdown60']=dd
    g['_downside_resilience']=-(2.*down+.30*dd.abs())
    # Raw price times vendor share volume: dollar-liquidity proxy, not free float.
    turnover=(g.close*v).where(v>0)
    illiquidity=(r.abs()/turnover).rolling(20,min_periods=20).mean()
    g['_liquidity_quality']=-np.log(illiquidity.clip(lower=1e-15))
    ma20=c.rolling(20,min_periods=20).mean();ma60=c.rolling(60,min_periods=60).mean()
    trend_ok=(ma20>=ma60*.985)&(ma60>=ma60.shift(10))
    pullback=c/hi.shift(1).rolling(20,min_periods=20).max()-1
    near_support=(c-ma20).abs()/(g.atr*g.calc_close/g.close).replace(0,np.nan)
    recovering=(c>o)&(c>c.shift(1))&(lo>=lo.shift(1)*.99)
    orderly=(trend_ok&(pullback.between(-.12,-.01))&(near_support<2.5))
    g['orderly_pullback']=(30+35*orderly+20*(orderly&recovering)+10*(orderly&(g.volume_ratio<1.8))).astype(float)
    g.loc[ma60.shift(10).isna(),'orderly_pullback']=np.nan
    band=4*c.rolling(20,min_periods=20).std()/ma20
    history=band.shift(1).rolling(60,min_periods=40)
    compression=band.shift(1)<history.quantile(.3)
    distance=(c/g.prior_high20/(g.calc_close/g.close)-1)
    # prior_high20 is already in displayed raw coordinates.
    near=distance.between(-.035,.025)
    release=(c>c.shift(1))&(g.volume_ratio.between(1.05,2.8))&(g.close_position>.6)
    g['squeeze_release']=(30+30*(compression&near)+25*(compression&near&release)+10*(near&release)).astype(float)
    g.loc[band.shift(1).rolling(60,min_periods=40).count()<40,'squeeze_release']=np.nan
    g['compression_before_signal']=compression.where(band.shift(1).notna())
    return g


def extend_panel(panel, cfg):
    """Called after legacy eligibility so legacy baseline rankings are unchanged."""
    f=pd.concat([_roll_price_features(g) for _,g in panel.groupby('ticker',sort=False)],ignore_index=True)
    f=f.sort_values(['date','ticker']).reset_index(drop=True)
    b=f[f.ticker==cfg.benchmark].set_index('date').ret1
    # No fill: missing benchmark observations cannot turn into made-up flat days.
    med=f[f.ticker!=cfg.benchmark].groupby('date').ret1.median()
    m=f.date.map(b);f['resilience_market_proxy']=np.where(m.notna(),cfg.benchmark,'CROSS_SECTION_PROXY')
    f['_market_ret']=m.fillna(f.date.map(med))
    parts=[]
    for _,g in f.groupby('ticker',sort=False):
        g=g.sort_values('date').copy();down=g._market_ret<0
        valid=g.ret1.notna()&g._market_ret.notna()
        count=(down&valid).astype(float).rolling(60,min_periods=50).sum()
        alpha=(g.ret1-g._market_ret).where(down&valid,0.).rolling(60,min_periods=50).sum()/count.replace(0,np.nan)
        g['_market_resilience']=alpha.where(count>=10)
        parts.append(g)
    f=pd.concat(parts,ignore_index=True).sort_values(['date','ticker']).reset_index(drop=True)
    for factor in ('momentum_persistence','downside_resilience','liquidity_quality','market_resilience'):
        raw=f['_'+factor].replace([np.inf,-np.inf],np.nan)
        use=raw.where(f.eligible & f.ticker.ne(cfg.benchmark))
        count=use.groupby(f.date).transform('count')
        rank=use.groupby(f.date).rank(method='average',pct=True)
        f[factor]=(100*rank).where(count>=cfg.min_universe)
    # No unknown evidence replaced by zero; optional groups are attached separately.
    for factor in EXTENDED:f[factor]=pd.to_numeric(f[factor],errors='coerce').clip(0,100)
    return f.drop(columns=[c for c in f if c.startswith('_')],errors='ignore')
