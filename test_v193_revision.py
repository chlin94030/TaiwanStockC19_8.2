"""Broader-factor regression tests. Synthetic fixtures are NOT investment results."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from test_v19_core import prices,cfg,panel
from test_v19_ui import ui
from v19_config import Settings, TECH, OPTIONAL, FACTORS, prior
from v19_features import build_panel, rank_date, score_rows
from v193_factors import EXTENDED, FAMILIES, FAMILY_CAPS, expanded_prior, cap_families
from v193_selection import candidate_weights, expanded_fit
from v193_fast import PortfolioCache
from v192_fast import ReplayContext
from v191_selection import portfolio_matrix, stability_gate
from v193_reporting import expanded_summary
from v193_ui import LABELS


def test_design_switch_explicit():
    assert Settings().factor_design=='balanced_v198'
    assert Settings(factor_design='expanded_v193').factor_design=='expanded_v193'
    assert replace(Settings(),factor_design='legacy_v192').factor_design=='legacy_v192'
    with pytest.raises(ValueError):replace(Settings(),factor_design='magic_best')


def test_eight_new_groups_distinct_and_labeled():
    assert len(EXTENDED)==8 and len(set(FACTORS))==len(FACTORS)==56
    assert set(EXTENDED).issubset(LABELS)
    assert set(EXTENDED).isdisjoint(TECH+OPTIONAL)

@pytest.mark.parametrize('h',[5,20,60])
def test_prior_is_finite_and_capped(h):
    w=expanded_prior(h,TECH+EXTENDED)
    assert sum(w.values())==pytest.approx(1)
    assert all(np.isfinite(v) and v>=0 for v in w.values())
    for family,keys in FAMILIES.items():
        assert sum(w.get(k,0) for k in keys)<=FAMILY_CAPS[family]+1e-8

@pytest.mark.parametrize('seed',range(5))
def test_family_caps_reduce_redundancy(seed):
    rng=np.random.default_rng(seed);w=cap_families(dict(zip(FACTORS,rng.lognormal(0,3,len(FACTORS)))))
    assert sum(w.values())==pytest.approx(1)
    for family,keys in FAMILIES.items():
        assert sum(w.get(k,0) for k in keys)<=FAMILY_CAPS[family]+1e-7


def test_extended_prefix_no_future(prices,cfg):
    dates=sorted(prices.date.unique());cut=dates[260]
    allframe=build_panel(prices,cfg=cfg)
    prefix=build_panel(prices[prices.date<=cut],cfg=cfg)
    pd.testing.assert_frame_equal(prefix[['date','ticker']+list(EXTENDED)].reset_index(drop=True),
       allframe[allframe.date<=cut][['date','ticker']+list(EXTENDED)].reset_index(drop=True),check_dtype=False)


def test_future_prices_do_not_rewrite_past_new_features(prices,cfg):
    p=prices.copy();cut=sorted(p.date.unique())[260]
    f=build_panel(p,cfg=cfg)
    mask=p.date>cut
    for col in ['open','high','low','close','adj_close']:p.loc[mask,col]*=2
    q=build_panel(p,cfg=cfg)
    pd.testing.assert_frame_equal(f[f.date<=cut][list(EXTENDED)].reset_index(drop=True),
                                 q[q.date<=cut][list(EXTENDED)].reset_index(drop=True))

@pytest.mark.parametrize('factor',EXTENDED)
def test_new_factor_values_within_research_scale(panel,factor):
    s=panel[factor].dropna()
    assert len(s)>100 and s.between(0,100).all()


def test_missing_business_not_synthesized(panel):
    assert panel[list(OPTIONAL)].isna().all().all()
    assert panel.eps_ttm.isna().all()

@pytest.mark.parametrize('h',[5,20,60])
def test_array_cache_matches_reference_engine(panel,cfg,h):
    pairs=[(d,g) for d,g in panel[panel.eligible].groupby('date') if len(g)>=cfg.min_universe][90:100]
    cache=PortfolioCache(pairs,cfg)
    ws=[prior(h,TECH+EXTENDED),expanded_prior(h,TECH+EXTENDED)]
    a=portfolio_matrix([g for d,g in pairs],ws,h,cfg);b=cache.matrix(pairs,ws,h)
    for key in ('net','mae','fill'):np.testing.assert_allclose(a[key],b[key],equal_nan=True,rtol=1e-10)
    assert (a['basis']==b['basis']).all()


def test_baseline_ranking_does_not_pick_up_zero_weight_factors(panel,cfg):
    d=panel.date.max()
    a=rank_date(panel,d,prior(5,TECH),cfg)
    b=rank_date(panel,d,prior(5,TECH+EXTENDED),cfg)
    assert list(a.ticker)==list(b.ticker)
    np.testing.assert_allclose(a.score,b.score)


def test_candidate_budget_capped_after_shrink(panel,cfg):
    train=panel[panel.eligible].head(1000)
    keys=list(TECH+EXTENDED)
    candidates,names=candidate_weights(60,keys,cfg,{},train)
    assert len(candidates)<=cfg.trials and names[0]=='legacy'
    for w in candidates[1:]:
        for family,group in FAMILIES.items():
            assert sum(w.get(k,0) for k in group)<=FAMILY_CAPS[family]+1e-8


def test_ic_inputs_cannot_create_negative_weights(panel,cfg):
    diag={k:{'train_rank_ic_mean':-1,'positive_ic_fraction':0} for k in TECH+EXTENDED}
    ws,names=candidate_weights(5,list(TECH+EXTENDED),cfg,diag,panel[panel.eligible].head(1000))
    assert all(v>=0 for w in ws for v in w.values())


def test_past_only_context(panel,cfg):
    c=ReplayContext(panel,cfg);cut=sorted(panel.date.unique())[300]
    for h in cfg.horizons:
        for date,g in c.before(cut,h):
            assert date<cut and (g[f'exit_date_{h}']<cut).all()


def test_gate_still_rejects_one_lucky_block():
    g=stability_gate([.2,-.01,-.01,-.01],[0]*4,[0]*4,[-.02]*4,[-.02]*4,[1]*4,Settings())
    assert not g['accepted'] and not g['tests']['not_one_winner']


def test_gate_requires_positive_benchmark_edge():
    g=stability_gate([.02]*4,[.01]*4,[.03]*4,[-.02]*4,[-.02]*4,[1]*4,Settings())
    assert not g['accepted'] and not g['tests']['beats_benchmark_on_average']


def test_ui_has_descriptive_labels(ui,panel,cfg):
    import v19_ui
    assert all(k in v19_ui.LABELS for k in FACTORS)
    rows=rank_date(panel,panel.date.max(),expanded_prior(5,TECH+EXTENDED),cfg)
    row=rows.iloc[0].to_dict();row['setup']='WATCH';row['stop_distance_pct']=3.
    v19_ui.candidate_card(row,{'state':'INSUFFICIENT','count':0},5,expanded_prior(5,TECH+EXTENDED),1)


def test_summary_never_fills_missing_with_zero():
    cfg=Settings(horizons=(5,),dates=2)
    cal=pd.bdate_range('2025-01-01',periods=50);dates=[str(cal[i].date()) for i in (1,10)]
    r=[]
    for i,d in enumerate(dates):
        for label,value in [('baseline',.01),('expanded_fixed_research',.02 if i==0 else None),('benchmark_0050',.015)]:
            r.append(dict(date=d,horizon=5,strategy=label,mean_net=value,price_basis='ADJUSTED_PROVIDER'))
    s=expanded_summary(r,cal,cfg)[0]
    assert s['paired_cases']==1 and s['candidate_mean']==pytest.approx(.02)


def test_source_code_keeps_acceptance_checks():
    text=Path(__file__).with_name('v193_selection.py').read_text()
    assert 'stability_gate(' in text and "gate['training_better']" in text
    assert "holdout_used_for_selection':False" in text
