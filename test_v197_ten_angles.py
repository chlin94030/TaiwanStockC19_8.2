"""V19.7 ten-angle factor and prequential validation regression contracts."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from v19_config import Settings,VERSION, FACTORS,prior,TECH
from v19_features import build_panel,score_rows
from v197_ten_pillars import PILLARS,add_pillars,ten_angle_coverage
from v197_selection import profiles_for, TenPrepared, TenTournament,ten_summary

@pytest.fixture(scope='module')
def panel197():
    from pathlib import Path
    path=Path(__file__).parent/'input_prices.csv.gz'
    if path.exists():
        raw=pd.read_csv(path)
        evidence_path=Path(__file__).parent/'input_evidence.csv.gz'
        evidence=pd.read_csv(evidence_path) if evidence_path.exists() else pd.DataFrame()
    else:
        # Fully portable synthetic fixtures are engineering checks, never market results.
        rng=np.random.default_rng(1970)
        dates=pd.bdate_range('2025-01-01',periods=430)
        pieces=[]
        for i in range(21):
            ticker='0050.TW' if i==0 else f'{1000+i}.TW'
            close=30*(1+(i%5)*.2)*np.exp(np.cumsum(rng.normal(.0005,.012,len(dates))))
            opening=close*np.exp(rng.normal(0,.003,len(dates)))
            high=np.maximum(opening,close)*1.012
            low=np.minimum(opening,close)*.988
            pieces.append(pd.DataFrame({'date':dates,'ticker':ticker,'open':opening,
                'high':high,'low':low,'close':close,'adj_close':close,
                'volume':500000+rng.integers(0,50000,len(dates)),
                'source':'SYNTHETIC_TEST_ONLY','name':ticker,'sector':f'sector_{(i-1)//5}' if i else 'benchmark'}))
        raw=pd.concat(pieces,ignore_index=True)
        evidence=pd.DataFrame()
    cfg=Settings(as_of='2026-10-15',optional_data=True)
    return build_panel(raw,evidence,cfg)

def test_defaults_and_complete_factor_names():
    assert VERSION=='19.8'
    assert Settings().factor_design=='balanced_v198'
    assert len(PILLARS)==12 and all(x in FACTORS for x in PILLARS)
    assert replace(Settings(),factor_design='tournament_v196').factor_design=='tournament_v196'

def test_profiles_do_not_force_every_factor():
    for h in (5,20,60):
        profiles=profiles_for(h,Settings(factor_design='ten_angle_v197'))
        assert profiles['legacy_control']==prior(h,TECH)
        for name,w in profiles.items():
            if name!='legacy_control':assert len(w)<=3
            assert sum(w.values())==pytest.approx(1.)
            assert all(v>0 for v in w.values())
    assert 'revenue_growth' not in profiles_for(5,Settings(factor_design='ten_angle_v197'))
    assert 'revenue_growth' in profiles_for(5,Settings(optional_data=True,factor_design='ten_angle_v197'))

def test_historical_unaudited_sector_not_trade_authorization():
    assert 'p197_sector' in profiles_for(5,Settings(factor_design='ten_angle_v197'))['sector_leadership_RESEARCH_ONLY']
    assert 'p197_sector' not in profiles_for(5,Settings(factor_design='ten_angle_v197'))['high_liquid']

def test_ten_angle_scores_within_percentiles(panel197):
    for k in PILLARS:
        x=panel197[k].dropna()
        assert x.between(0.,100.).all(),k
    assert panel197.p197_market_breadth.dropna().between(0,1).all()

def test_no_undated_financial_data_masquerades_as_score(panel197):
    assert panel197.revenue_yoy.isna().all()
    assert panel197.p197_revenue.isna().all()
    assert panel197.p197_quality.isna().all()
    assert panel197.p197_valuation.isna().all()
    coverage=ten_angle_coverage(panel197)
    assert next(x for x in coverage if x['factor']=='p197_revenue')['coverage']==0

def test_sector_leave_one_out_with_at_least_three_peers(panel197):
    available=panel197.loc[panel197.p197_sector.notna() & panel197.eligible]
    assert len(available)>50
    labels=panel197.loc[available.index,['date','sector','ret20']]
    for _,r in labels.head(30).iterrows():
        peers=panel197[(panel197.date==r.date)&(panel197.sector==r.sector)&panel197.eligible&panel197.ret20.notna()]
        assert len(peers)>=4

def test_observed_market_context_is_not_a_stock_ranking(panel197):
    d=panel197.date.unique()[-30]
    values=panel197.loc[panel197.date.eq(d),'p197_market_breadth'].dropna().unique()
    assert len(values)<=1
    assert not any('market_breadth' in w for p in profiles_for(5,Settings(factor_design='ten_angle_v197')).values() for w in p)

def test_no_new_penalty_in_ten_angle_scorecards(panel197):
    frame=panel197.loc[panel197.eligible & panel197.p197_liquidity.notna()].head(25).copy()
    if frame.empty:pytest.skip('No eligible valid frame')
    base,_=score_rows(frame,{'p197_liquidity':1.},1.)
    assert np.allclose(base,frame.p197_liquidity)

def test_prefix_invariance(panel197):
    # Last available price data cannot alter a factor assigned one year earlier.
    d=pd.Timestamp('2025-10-01');p=panel197.copy()
    p.loc[p.date>d,'close']*=2
    p.loc[p.date>d,'calc_close']*=2
    p.loc[p.date>d,'turnover20']*=2
    old=add_pillars(panel197.copy(), Settings(as_of='2026-10-15'))
    new=add_pillars(p,Settings(as_of='2026-10-15'))
    fields=['p197_annual_high','p197_liquidity','p197_trend_eff','p197_compression']
    x=old.loc[old.date<=d,fields].reset_index(drop=True)
    y=new.loc[new.date<=d,fields].reset_index(drop=True)
    pd.testing.assert_frame_equal(x,y,check_dtype=False)

def test_shadow_summary_never_mix_unpaired_bases():
    def line(strategy,v,basis='ADJUSTED_PROVIDER'):
        return dict(date='2025-01-02',horizon=5,strategy=strategy,mean_net=v,price_basis=basis,complete=True)
    test=[line('ten_shadow_high_52',.1),line('baseline',.03),line('benchmark_0050',.04)]
    a=ten_summary({'results':test})
    assert len(a)==1 and a[0]['vs_0050']==pytest.approx(.06)
    test[0]['price_basis']='RAW_PRICE_ONLY'
    assert ten_summary({'results':test})==[]


def test_unknown_stock_specific_fundamental_never_imputed():
    f=pd.DataFrame({'p197_revenue':[np.nan,50.], 'p197_liquidity':[75.,40.]})
    a,c=score_rows(f,{'p197_revenue':.5,'p197_liquidity':.5},1.)
    assert np.isnan(a.iloc[0]) and a.iloc[1]==pytest.approx(45.)
    assert c.iloc[0]==pytest.approx(.5)
