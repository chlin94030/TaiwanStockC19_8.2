"""V19.6 selection, as-of, pricing and explanation regression contracts."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from v19_config import Settings, FACTORS, TECH, VERSION, prior
from v196_factors import add_scorecards,scorecard_weights,NEW_FACTORS
from v196_selection import Tournament,ScorecardPrepared
from v19_features import score_rows,rank_date
from v19_backtest import selected_result
from test_v194_sparse import synthetic_labels

@pytest.fixture(scope='module')
def p196():
    p=synthetic_labels(n=190,stocks=12,horizon=5)
    p['turnover20']=1e7+pd.factorize(p.ticker)[0]*1e6
    p['ret60']=pd.factorize(p.ticker)[0]/100.
    p['downside_resilience']=50.
    p['path_complete_5']=True
    p['mdd_5']=p['mae_5']
    p['eligible']=p.ticker.ne('0050.TW')
    return add_scorecards(p,Settings(min_universe=8,as_of='2025-12-31'))

@pytest.fixture
def c196():return Settings(as_of='2025-12-31',horizons=(5,),dates=2,top_k=3,min_universe=8)

@pytest.mark.parametrize('field,value',[('tournament_blocks',3),('tournament_blocks',21),('tournament_uncertainty',-1),('tournament_uncertainty',4)])
def test_settings_reject_bad(field,value):
    with pytest.raises(ValueError):Settings(**{field:value})

def test_default_and_old_engine():
    assert VERSION=='19.8' and Settings().factor_design=='balanced_v198'
    assert replace(Settings(),factor_design='sparse_v195').factor_design=='sparse_v195'

@pytest.mark.parametrize('h',[5,20,60])
def test_profiles_nonnegative_normalized_sparse(h,c196):
    profiles=scorecard_weights(h,c196)
    assert len(profiles)==8
    for name,w in profiles.items():
        assert sum(w.values())==pytest.approx(1.) and all(v>0 for v in w.values())
        assert all(k in FACTORS for k in w)
        if name!='legacy_control':assert len(w)<=2
    assert len(scorecard_weights(h,replace(c196,optional_data=True)))==13

def test_same_day_future_invariance(p196,c196):
    d=p196.date.unique()[100]
    a=add_scorecards(p196,c196)
    b=p196.copy();b.loc[b.date>d,'turnover20']*=1000;b=add_scorecards(b,c196)
    pd.testing.assert_frame_equal(a.loc[a.date<=d,list(NEW_FACTORS)],b.loc[b.date<=d,list(NEW_FACTORS)])

def test_optional_missing_is_not_neutral(p196,c196):
    for k in ['r196_revenue','r196_earnings_growth','r196_earnings_yield']:
        # synthetic_labels supplies no observations of these optional parents.
        x=p196.copy();raw=k[5:];x[raw]=np.nan;x=add_scorecards(x,c196)
        assert x[k].isna().all()

def test_percentiles_are_not_benchmark_quotes(p196):
    assert p196.loc[p196.ticker.eq('0050.TW'),list(NEW_FACTORS)].isna().all().all()

def test_scorecard_no_hidden_overheat_penalty_and_legacy_unchanged():
    f=pd.DataFrame({'r196_dollar_turnover':[90.,80.],'bias20':[24.,0.],'trend':[90.,80.]})
    a,_=score_rows(f,{'r196_dollar_turnover':1.})
    b,_=score_rows(f,{'trend':1.})
    assert a.tolist()==[90.,80.] and b.tolist()==[81.6,80.]

def test_missing_selected_input_not_renormalized(p196,c196):
    d=p196.date.max();q=p196[p196.date==d].copy();q.loc[q.index[0],'r196_dollar_turnover']=np.nan
    s,_=score_rows(q,{'r196_dollar_turnover':.5,'r196_medium_return':.5},min_coverage=1.-1e-9)
    assert np.isnan(s.iloc[0])

def test_zero_deleted_weight_stays_zero(c196):
    w=scorecard_weights(5,c196)['turnover_only']
    assert not any(w.get(k,0)>0 for k in TECH)

def test_blocks_disjoint_matured_only(p196,c196):
    obj=Tournament(p196,5,c196);d=p196.date.max()
    ix=obj.blocks_before(d,8);assert len(ix)==8
    assert all(obj.ends[i]<d for i in ix)
    assert all(obj.ends[a]<obj.dates[b] for a,b in zip(ix,ix[1:]))

def test_future_labels_cannot_change_choice_or_gate(p196,c196):
    d=pd.Timestamp(p196.date.unique()[155]);a=Tournament(p196,5,c196).fit(d)
    x=p196.copy();bad=x.exit_date_5>=d;x.loc[bad,'net_5']=999.;x.loc[bad,'mae_5']=-999.
    b=Tournament(x,5,c196).fit(d)
    assert a==b

def test_selected_missing_outcome_does_not_replace_date(p196,c196):
    obj=Tournament(p196,5,c196);d=p196.date.max();ix=obj.blocks_before(d,8)
    target=ix[-1];obj.outcomes['turnover_only']['net'][target]=np.nan
    _,_,tr=obj.select(d);row=next(x for x in tr if x['candidate_name']=='turnover_only')
    assert not row['valid'] and len(row['block_dates'])==8

def test_past_chosen_policy_precedes_every_gate_date(p196,c196):
    _,i,_=Tournament(p196,5,c196).fit(p196.date.max())
    for row in i['prequential_choices']:
        assert row['selection_max_exit'] is None or row['selection_max_exit']<row['cutoff']
    assert i['allocation_allowed'] is False or i['stability_gate']['accepted']

def test_fast_exact_scorecard_replay(p196,c196):
    obj=Tournament(p196,5,c196);w={'r196_dollar_turnover':.25,'r196_downside_resilience':.75}
    a=obj.prepared.evaluate(w)
    for j in [30,80,120]:
        d=obj.dates[j];rec,_=selected_result(p196,d,w,5,c196,'test')
        if rec['mean_net'] is None:assert np.isnan(a['net'][j])
        else:assert a['net'][j]==pytest.approx(rec['mean_net'])

def test_low_history_no_fabricated_approval(p196,c196):
    d=p196.date.unique()[10];w,i,tr=Tournament(p196,5,c196).fit(d)
    assert not i['allocation_allowed'];assert i['selected_profile']=='legacy_control'

@pytest.mark.parametrize('metric',['net','mae'])
def test_unknown_comparison_never_authorizes(p196,c196,metric):
    obj=Tournament(p196,5,c196)
    for o in obj.outcomes.values():o[metric][:]=np.nan
    _,info,_=obj.fit(p196.date.max())
    assert not info['allocation_allowed']

def test_gate_info_distinguishes_research_from_approved(p196,c196):
    _,info,_=Tournament(p196,5,c196).fit(p196.date.max())
    assert 'challenger_weights' in info
    assert info['weights_are_portfolio_allocations'] is False
    assert not info['holdout_used_for_selection']

def test_control_default_is_not_eagerly_evaluated_for_new_weights(p196,c196,monkeypatch):
    import v19_backtest as bt
    day=p196[p196.date==p196.date.unique()[120]].copy()
    monkeypatch.setattr(bt,'fit_before',lambda *a,**k:({'r196_dollar_turnover':1.},
        {'control_weights':prior(5,TECH),'adopted_is_legacy_control':False,'status':'test'},[]))
    for col in ['mfe_5','base_return_5','exit_close_5','entry_open','entry_gap']:
        if col not in day:day[col]=0.
    payload=bt._case(day,day,day.date.iloc[0],5,replace(c196,factor_audit=False),1,None)
    assert any(x['strategy']=='adaptive' for x in payload['records'])


def test_simple_summary_uses_matched_dates_not_missing_as_cash():
    from v196_factors import simple_summary
    rows=[dict(date='2025-01-01',horizon=5,strategy='simple_control_turnover_only',mean_net=.01,price_basis='ADJUSTED_PROVIDER',complete=True),
          dict(date='2025-01-01',horizon=5,strategy='benchmark_0050',mean_net=.02,price_basis='ADJUSTED_PROVIDER',complete=True),
          dict(date='2025-01-02',horizon=5,strategy='simple_control_turnover_only',mean_net=.5,price_basis='ADJUSTED_PROVIDER',complete=True),
          dict(date='2025-01-02',horizon=5,strategy='benchmark_0050',mean_net=None,price_basis='UNKNOWN',complete=False)]
    answer=simple_summary({'results':rows})
    assert answer[0]['paired_cases']==1
    assert answer[0]['net']==pytest.approx(.01)
    assert answer[0]['excess']==pytest.approx(-.01)


def test_report_and_profile_labels_do_not_claim_optimized_profit():
    from v196_factors import profile_label,simple_summary
    from v194_reporting import tournament_markdown
    assert '75%' in profile_label('turnover_defense_75')
    assert simple_summary({})==[]
    text=tournament_markdown({'weights':{},'results':[]})
    assert 'not automatically promoted' in text
    assert 'not new independent factors' in text


@pytest.mark.parametrize('module_name',['v194_selection','v195_selection'])
def test_old_sparse_optional_mode_cannot_silently_add_new_scorecards(module_name,p196,c196,monkeypatch):
    import importlib
    mod=importlib.import_module(module_name); captured=[]
    def reserve(groups,h,cfg):return groups[:20],groups[-4:]
    def search(pairs,panel,features,h,cfg):
        captured.extend(features);return None,[],{'reason':'CAPTURE_ONLY'}
    monkeypatch.setattr(mod,'reserve_validation' if module_name=='v195_selection' else 'separated_validation',reserve)
    monkeypatch.setattr(mod,'sparse_search',search)
    cfg=replace(c196,optional_data=True,factor_design='sparse_v195' if module_name=='v195_selection' else 'sparse_v194')
    mod.sparse_fit(p196,p196.date.max()+pd.Timedelta(days=90),5,cfg)
    assert captured and not any(k.startswith('r196_') for k in captured)
