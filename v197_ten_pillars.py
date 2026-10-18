"""V19.7 ten-perspective research factors, point-in-time price bars only.

Ten lenses: market context (risk gate), sector peer trend (research-only),
52-week price position, trend efficiency, compression/breakout, published
institutional flow, revenue growth, profits/valuation, downside risk,
liquidity/transaction capacity. Missing evidence stays missing.
All ranks are contemporaneous cross-sectional ranks, not probabilities.
The original current-sector mapping is not historical-membership certified.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

LENSES = {
 'market_context':'市場環境／廣度（資金風控，不做個股加分）',
 'industry_rotation':'產業輪動（歷史產業成分尚未核實，僅研究）',
 'year_high':'近一年高點與位置',
 'trend_quality':'趨勢效率與動能',
 'compression':'波動收斂與突破',
 'institutional':'外資與投信持續性',
 'revenue':'已公布營收成長',
 'profit_value':'獲利品質與估值',
 'downside':'下檔風險／波動',
 'liquidity':'成交量能／可成交性',
}
PILLARS = (
 'p197_annual_high','p197_trend_eff','p197_compression','p197_sector',
 'p197_foreign','p197_trust','p197_revenue','p197_quality','p197_valuation',
 'p197_downside','p197_liquidity','p197_relative',
)
REQUIRES_DISCLOSURE = frozenset(('p197_revenue','p197_quality','p197_valuation'))
UNVERIFIED_SECTOR = frozenset(('p197_sector',))
NAMES = {
 'p197_annual_high':'近一年高點位置',
 'p197_trend_eff':'趨勢行進效率',
 'p197_compression':'波動收斂後突破',
 'p197_sector':'同業強勢（歷史分類待核）',
 'p197_foreign':'外資買盤20日',
 'p197_trust':'投信買盤20日',
 'p197_revenue':'營收成長（須公告時點）',
 'p197_quality':'獲利與現金流品質',
 'p197_valuation':'估值（本益比）',
 'p197_downside':'下檔波動防護',
 'p197_liquidity':'成交金額容量',
 'p197_relative':'中期相對大盤強度',
}

def _safe(x):
    return pd.to_numeric(x,errors='coerce').replace([np.inf,-np.inf],np.nan)


def add_pillars(panel,cfg):
    """Use bars through date T, and only already available optional evidence."""
    p=panel.copy()
    for k in PILLARS:p[k]=np.nan
    p['p197_market_breadth']=np.nan
    p['p197_market_gate']='UNKNOWN'
    for ticker,ix in p.groupby('ticker',sort=False).groups.items():
        g=p.loc[ix].sort_values('date'); idx=g.index
        c=_safe(g.calc_close); hi=_safe(g.calc_high);lo=_safe(g.calc_low)
        v=_safe(g.share_volume);ret=c.pct_change(fill_method=None)
        # Historical 252-bar position, a signal rather than a future maximum.
        max252=hi.rolling(252,min_periods=180).max()
        p.loc[idx,'p197_annual_high']=(c/max252).where(max252.gt(0)).to_numpy()
        path=c.diff().abs().rolling(40,min_periods=40).sum()
        p.loc[idx,'p197_trend_eff']=((c-c.shift(40))/path.replace(0,np.nan)).to_numpy()
        # Compare previous contraction with today's break of a pre-existing range.
        shortvol=ret.shift(1).rolling(10,min_periods=10).std()
        longvol=ret.shift(1).rolling(50,min_periods=50).std()
        priorhigh=hi.shift(1).rolling(20,min_periods=20).max()
        contraction=(1-shortvol/longvol.replace(0,np.nan)).clip(-1,1)
        release=((c/priorhigh-1)*15).clip(-1,1)
        p.loc[idx,'p197_compression']=(.6*contraction+.4*release).to_numpy()
        downside=ret.clip(upper=0).rolling(40,min_periods=30).std()
        max60=c.rolling(60,min_periods=40).max()
        drawdown=(c/max60-1)
        p.loc[idx,'p197_downside']=(-.6*downside-.4*drawdown.abs()).to_numpy()
        p.loc[idx,'p197_liquidity']=np.log1p(_safe(g.turnover20).clip(lower=0)).to_numpy()
        # True vendor flow is not inferred from price. Missing days stay unknown.
        den=v.rolling(20,min_periods=20).sum().replace(0,np.nan)
        for col,dest in (('foreign_net','p197_foreign'),('trust_net','p197_trust')):
            if col in g:
                flow=_safe(g[col]);obs=flow.rolling(20,min_periods=20).count()
                scaled=(flow.rolling(20,min_periods=20).sum()/den).where(obs>=20)
                p.loc[idx,dest]=scaled.to_numpy()
        if 'revenue_yoy' in g:p.loc[idx,'p197_revenue']=_safe(g.revenue_yoy).to_numpy()
        if 'gross_margin' in g and 'cash_conversion_ratio' in g:
            # Require both components. A missing cash-flow observation is not 50.
            gm=_safe(g.gross_margin);cc=_safe(g.cash_conversion_ratio)
            p.loc[idx,'p197_quality']=(gm.clip(-100,100)/100+cc.clip(-2,3)*.25).where(gm.notna()&cc.notna()).to_numpy()
        if 'pe_ratio' in g:
            pe=_safe(g.pe_ratio)
            p.loc[idx,'p197_valuation']=(1/pe).where(pe>0).to_numpy()
        if 'ret60' in g:
            r=_safe(g.ret60)
            p.loc[idx,'p197_relative']=r.to_numpy()
    # Subtract known 0050 60-bar return where available; never manufacture an ETF quote.
    benchmark=p[p.ticker.eq(cfg.benchmark)][['date','ret60']].drop_duplicates('date').set_index('date').ret60
    bench=p.date.map(benchmark)
    p['p197_relative']=_safe(p.p197_relative)-_safe(bench)
    # Sector peer strength must EXCLUDE the stock itself; current-sector mapping
    # is only a research proxy until dated historical classification is supplied.
    sector_ok=p.eligible & p.ticker.ne(cfg.benchmark) & p.sector.notna() & p.ret20.notna()
    group=p.loc[sector_ok].groupby(['date','sector']).ret20.agg(['sum','count'])
    current=p.loc[sector_ok,['date','sector','ret20']].join(group,on=['date','sector'])
    valid=(current['count']>=4)
    peer=((current['sum']-current.ret20)/(current['count']-1)).where(valid)
    p.loc[current.index,'p197_sector']=peer
    # Market breadth is an aggregate with no cross-sectional discriminating power.
    stocks=p.eligible & p.ticker.ne(cfg.benchmark)
    good=p.loc[stocks & p.ma60.notna()].copy()
    breadth=(good.calc_close>good.ma60).groupby(good.date).mean()
    coverage=good.groupby('date').ticker.nunique()
    breadth=breadth.where(coverage>=cfg.min_universe)
    p['p197_market_breadth']=p.date.map(breadth)
    # This is an observation-only context label, NOT a direct trade signal.
    p['p197_market_gate']=np.select([p.regime.eq('BULL')&p.p197_market_breadth.ge(.55),
                                   p.regime.eq('BEAR')|p.p197_market_breadth.lt(.35)],
                                   ['SUPPORTIVE','CAUTION'],default='NEUTRAL')
    # Convert raw observed signals to SAME-DATE cross-sectional ranks. Critically,
    # completeness and eligible membership do not use future outcome availability.
    for col in PILLARS:
        x=_safe(p[col]).where(stocks)
        valid=x.groupby(p.date).transform('count')>=cfg.min_universe
        p[col]=(x.groupby(p.date).rank(pct=True,method='average')*100).where(valid)
    return p


def ten_angle_coverage(panel):
    eligible=panel.loc[panel.eligible] if 'eligible' in panel else panel
    out=[]
    for k in PILLARS:
        value=_safe(eligible[k]) if k in eligible else pd.Series(dtype=float)
        out.append({'factor':k,'name':NAMES[k],
                    'coverage':float(value.notna().mean()) if len(eligible) else 0.,
                    'research_only':k in UNVERIFIED_SECTOR,
                    'reason':'HISTORICAL_SECTOR_MEMBERSHIP_UNVERIFIED' if k in UNVERIFIED_SECTOR else
                             'REQUIRES_RELEASE_PROVENANCE' if k in REQUIRES_DISCLOSURE else 'PRICE_OR_DATED_FLOW'})
    return out
