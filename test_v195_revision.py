"""V19.5 regression tests. Synthetic fixtures do NOT establish market returns."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from v19_config import Settings, VERSION, TECH, FACTORS, prior
from v195_selection import (TrainingMoments, compatible, inner_folds,
    reserve_validation, sparse_search, sparse_fit, _fit_subset)
from v194_selection import _stats, inner_folds as old_folds
from v194_ui import choose_model
from v19_service import analog_summary, field_health
from test_v194_sparse import synthetic_labels
from test_v19_ui import ui
from test_v19_core import prices,cfg,panel


@pytest.fixture(scope='module')
def sp():
    x=synthetic_labels(n=330,stocks=14)
    for k in FACTORS:
        if k not in ('rsi_rebound','trend','relative_strength'):x[k]=50.
    return x


@pytest.fixture
def settings():
    return Settings(as_of='2025-12-31',horizons=(5,),dates=2,top_k=3,
                    min_universe=8,train_dates=24,sparse_budget=48,factor_design='sparse_v195')


def test_new_default_and_old_engine_available():
    assert VERSION=='19.8' and Settings().factor_design=='balanced_v198'
    assert replace(Settings(),factor_design='sparse_v194').factor_design=='sparse_v194'


def test_unused_missing_columns_cannot_drop_fit_rows(sp):
    groups=list(sp[sp.eligible].groupby('date'))[:80]
    keys=['rsi_rebound','trend','relative_strength']
    a=TrainingMoments(groups,keys,5).stats(['rsi_rebound','trend'])
    bad=[(d,g.assign(relative_strength=np.nan)) for d,g in groups]
    b=TrainingMoments(bad,keys,5).stats(['rsi_rebound','trend'])
    np.testing.assert_allclose(a['gram'],b['gram']);np.testing.assert_allclose(a['rhs'],b['rhs'])
    assert a['rows']==b['rows']==80*14
    assert _stats(bad,keys,5)['rows']==0  # the former all-library cohort defect


def test_selected_missing_rows_are_excluded_not_zero_filled(sp):
    groups=list(sp[sp.eligible].groupby('date'))[:30]
    edited=[]
    for d,g in groups:
        g=g.copy();g.loc[g.index[0],'trend']=np.nan;edited.append((d,g))
    a=TrainingMoments(edited,['rsi_rebound','trend'],5)
    one=a.stats(['rsi_rebound']);two=a.stats(['rsi_rebound','trend'])
    assert one['rows']==420 and two['rows']==390
    assert two['coverage']==pytest.approx(13/14)


def test_support_moments_match_direct_selected_reference(sp):
    pairs=list(sp[sp.eligible].groupby('date'))[:25]
    keys=['rsi_rebound','trend']
    a=TrainingMoments(pairs,keys+['relative_strength'],5).stats(keys)
    b=_stats(pairs,keys,5)
    for k in ('gram','rhs','corr'):np.testing.assert_allclose(a[k],b[k],atol=1e-13)


def test_unknown_training_price_basis_not_used(sp):
    x=sp[sp.eligible].copy();x['return_basis_5']='UNKNOWN'
    stats=TrainingMoments(list(x.groupby('date')),['rsi_rebound'],5).stats(['rsi_rebound'])
    assert stats['rows']==0


@pytest.mark.parametrize('h,n',[(5,240),(20,280),(60,410)])
def test_inner_score_intervals_disjoint_across_all_folds(h,n,settings):
    x=synthetic_labels(n=n+70,stocks=9,horizon=h)
    pairs=list(x[x.eligible].groupby('date'))[:n]
    folds=inner_folds(pairs,h,settings)
    assert len(folds)>=2
    sequence=[]
    for fit,test in folds:
        assert max(g[f'exit_date_{h}'].max() for _,g in fit)<test[0][0]
        sequence+=test
    for (_,a),(d,b) in zip(sequence,sequence[1:]):
        assert a[f'exit_date_{h}'].max()<d


def test_old_fold_boundary_overlap_reproduced_but_fixed(sp,settings):
    pairs=list(sp[sp.eligible].groupby('date'))[:237]
    old=old_folds(pairs,5,settings)
    assert any(a[1][-1][1].exit_date_5.max()>=b[1][0][0] for a,b in zip(old,old[1:]))
    new=inner_folds(pairs,5,settings)
    assert all(a[1][-1][1].exit_date_5.max()<b[1][0][0] for a,b in zip(new,new[1:]))


@pytest.mark.parametrize('h,n',[(5,280),(20,400),(60,700)])
def test_final_blocks_prespecified_and_train_purged(h,n,settings):
    x=synthetic_labels(n=n+h+3,stocks=9,horizon=h)
    pairs=list(x[x.eligible].groupby('date'))[:n]
    train,valid=reserve_validation(pairs,h,settings)
    assert len(valid)==settings.validation_blocks
    assert max(g[f'exit_date_{h}'].max() for _,g in train)<valid[0][0]
    assert len(train)<=max(252,8*h+settings.train_dates)
    for (_,a),(d,b) in zip(valid,valid[1:]):assert a[f'exit_date_{h}'].max()<d
    if h==60:assert len(train)>252


@pytest.mark.parametrize('support',[
    ['reversal','rebound_accumulation_pair'],
    ['revenue','growth_pullback_pair'],
    ['trust','flow_breakout_pair'],
    ['quality','earnings_growth'],
    ['rsi_rebound','rebound_accumulation_pair']])
def test_indirect_composite_conflicts_rejected(support):assert not compatible(support)


def test_nonoverlapping_interactions_allowed():
    assert compatible(['growth_pullback_pair','flow_breakout_pair'])
    assert compatible(['kd_turn','volume'])


def test_candidate_methods_compared_inside_cv(sp,settings):
    pairs=list(sp[sp.eligible].groupby('date'))[:240]
    w,trace,info=sparse_search(pairs,sp,['rsi_rebound','trend','relative_strength'],5,settings)
    assert w and w['rsi_rebound']==1 and not any(w[k]>0 for k in TECH)
    assert len(trace)<=settings.sparse_budget
    assert {'equal','ridge'} <= {r['weight_method'] for r in trace}
    assert info['test_windows_globally_disjoint']
    assert info['refit_rows']==240*14
    assert all(r['objective'] is None or np.isfinite(r['objective']) for r in trace)


def test_synthetic_causal_signal_still_can_be_accepted(sp,settings):
    w,info,_=sparse_fit(sp,sorted(sp.date.unique())[310],5,settings)
    assert w['rsi_rebound']==1
    assert info['status']=='SPARSE_INNER_VALIDATION_ACCEPTED'
    assert len(info['validation_cutoffs'])==4
    assert info['failed_adoption_checks']==[]


def test_outer_future_outcomes_not_used(sp,settings):
    cut=sorted(sp.date.unique())[310]
    a,ai,at=sparse_fit(sp,cut,5,settings)
    q=sp.copy();q.loc[q.date>=cut,'net_5']=1e6
    b,bi,bt=sparse_fit(q,cut,5,settings)
    assert a==b and ai==bi and at==bt


def test_insufficient_data_does_not_fabricate_weights(sp,settings):
    w,info,trace=sparse_fit(sp,sorted(sp.date.unique())[10],5,settings)
    assert info['status']=='SPARSE_INSUFFICIENT_PAST_BLOCKS'
    assert 'challenger_weights' not in info and not info['allocation_allowed']


def test_constant_score_is_not_a_predictive_singleton(settings):
    stats={'dates':30,'coverage':1.,'gram':np.zeros((1,1)), 'rhs':np.zeros(1),'corr':np.zeros((1,1))}
    assert _fit_subset(stats,['trend'],'equal',settings) is None


def test_sparse_ui_reads_new_design_and_fit_dates(ui,settings):
    _,st=ui
    report={'version':VERSION,'settings':asdict(settings),'weights':{'5':{'weights':prior(5,TECH),
        'fit':{'cutoff':'2025-01-01','factor_design':'sparse_v195','status':'SPARSE_CANDIDATE_NOT_ADOPTED',
        'challenger_weights':{'rsi_rebound':1.},'failed_adoption_checks':['beats_benchmark_on_average']}}}}
    w,used=choose_model(report,5,settings,'2025-01-02',{})
    assert w=={'rsi_rebound':1.} and used.factor_design=='sparse_v195'
    assert any('2025-01-01' in str(a) for k,a,b in st.calls if k=='caption')
    assert any('0050' in str(a) for k,a,b in st.calls if k=='caption')


@pytest.mark.parametrize('field,value',[('warmup',81),('min_validation_fill',.8),('max_mae_deterioration',.02)])
def test_changed_risk_settings_do_not_reuse_weights(ui,settings,field,value):
    _,st=ui
    report={'version':VERSION,'settings':asdict(settings),'weights':{'5':{'weights':{'rsi_rebound':1.},
        'fit':{'cutoff':'2025-01-01','factor_design':'sparse_v195','challenger_weights':{'rsi_rebound':1.}}}}}
    w,_=choose_model(report,5,replace(settings,**{field:value}),'2025-01-02',{})
    assert w==prior(5,TECH)


def test_analog_query_requires_every_selected_input(panel):
    row=panel.iloc[-1].copy();row['trend']=np.nan
    answer=analog_summary(panel,row,5,{'trend':.1,'volume':.9})
    assert answer['state']=='MISSING_SELECTED_INPUTS' and answer['count']==0


def test_health_infinite_is_not_available(panel):
    p=panel.copy();p.loc[p.date==p.date.max(),'trend']=np.inf
    health=field_health(p)
    assert next(x for x in health if x['field']=='trend')['state']=='MISSING_NOT_ZERO'


def test_checkpoint_digest_contains_new_engine():
    import v192_checkpoint
    assert 'v195_selection.py' in Path(v192_checkpoint.__file__).read_text()


def test_main_visible_release_is_current(ui,panel,monkeypatch):
    m,f=ui
    monkeypatch.setattr(m,'_display_data',lambda *a,**k:(panel,{}))
    monkeypatch.setattr(m,'news_block',lambda:None)
    m.main()
    titles=[args[0] for name,args,kw in f.calls if name=='title']
    assert titles==['Alpha Radar V19.8']


def test_comparison_checks_exported_prices_not_only_serializer_hash():
    from compare_v195 import equivalent_prices
    a=pd.DataFrame({'ticker':['2330.TW'],'date':['2024-01-01'],'close':[100.1],'source':['vendor']})
    b=a.copy();b.loc[0,'close']+=1e-13
    assert equivalent_prices(a,b)
    b.loc[0,'close']+=.01
    assert not equivalent_prices(a,b)
    b=a.copy();b.loc[0,'ticker']='2317.TW'
    assert not equivalent_prices(a,b)


def test_comparison_rejects_missing_snapshot_with_hash_mismatch():
    from compare_v195 import compare
    with pytest.raises(ValueError,match='Input prices differ'):
        compare({'cutoffs':[],'data_hash':'old','settings':{}},
                {'cutoffs':[],'data_hash':'new','settings':{}})
