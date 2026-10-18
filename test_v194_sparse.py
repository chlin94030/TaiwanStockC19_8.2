"""Synthetic logic and integration tests; none represent market performance."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from v19_config import FACTORS, TECH, Settings, prior
from v194_factors import ATOMIC, BUSINESS, INTERACTIONS, compatible, input_support
from v194_selection import (support_weights, _solve, inner_folds, Prepared, sparse_search, sparse_fit)
from v194_evidence import financial_records
from v19_features import rank_date, score_rows, build_panel
from test_v19_core import prices, cfg, panel
from test_v19_ui import ui


def synthetic_labels(n=330, stocks=14, horizon=5):
    rng=np.random.default_rng(1940);dates=pd.bdate_range('2024-01-01',periods=n)
    rows=[]
    for i,d in enumerate(dates):
        for j in range(stocks+1):
            r={k:float(rng.uniform(0,100)) for k in FACTORS}
            ticker='0050.TW' if j==stocks else f'{1100+j}.TW'
            y=(r['rsi_rebound']/100-.5)*.08 if j<stocks else .001
            r.update(ticker=ticker,date=d,eligible=j<stocks,bias20=0.,
                     **{f'net_{horizon}':y,f'mae_{horizon}':-.02,
                        f'exit_date_{horizon}':dates[i+horizon] if i+horizon<n else pd.NaT,
                        f'outcome_state_{horizon}':'ASSUMED_OPEN_FILL',
                        f'return_basis_{horizon}':'ADJUSTED_PROVIDER'})
            rows.append(r)
    return pd.DataFrame(rows)


@pytest.fixture(scope='module')
def sparse_panel():return synthetic_labels()

@pytest.fixture
def sparse_cfg():return Settings(factor_design='sparse_v194',as_of='2025-12-31',horizons=(5,),dates=2,top_k=3,
                                 min_universe=8,train_dates=24,sparse_budget=48)


def test_default_design_new_and_legacy_still_selectable():
    assert Settings().factor_design=='balanced_v198'
    for name in ('legacy_v192','expanded_v193','sparse_v194'):
        assert replace(Settings(),factor_design=name).factor_design==name

@pytest.mark.parametrize('shrink',[.1,.5,1.])
def test_zero_support_never_restored(shrink):
    w=support_weights([2,0,1],['a','removed','b'],shrink)
    assert w['removed']==0 and sum(w.values())==pytest.approx(1)
    assert set(k for k,v in w.items() if v>0)=={'a','b'}


def test_ridge_can_zero_irrelevant_or_negative_terms():
    w=_solve(np.eye(3),np.array([1.,-1.,0.]),['a','b','c'],.02,.5)
    assert w=={'a':1.,'b':0.,'c':0.}


def test_no_weight_vector_for_all_nonpositive():
    assert _solve(np.eye(2),np.array([-1.,-2.]),['a','b'],.02,.5) is None
    assert support_weights([0,0],['a','b']) is None

@pytest.mark.parametrize('keys',[('reversal','kd_turn'),('quality','earnings_growth'),
                                  ('flow_breakout_pair','trust'),('rebound_accumulation_pair','accumulation')])
def test_composite_and_component_not_double_counted(keys):assert not compatible(keys)


def test_pair_counts_underlying_inputs():
    assert input_support(['flow_breakout_pair','liquidity_quality'])=={'trust','breakout','liquidity_quality'}


def test_sparse_search_can_really_remove_all_originals(sparse_panel,sparse_cfg):
    groups=list(sparse_panel[sparse_panel.eligible].groupby('date'))[:240]
    keys=['rsi_rebound','trend','relative_strength']
    w,traces,summary=sparse_search(groups,sparse_panel,keys,5,sparse_cfg)
    assert w and w['rsi_rebound']>0
    assert all(w[k]==0 for k in TECH)
    assert len(traces)<=sparse_cfg.sparse_budget
    assert summary['selected_factors']==['rsi_rebound']
    assert summary['search_exhaustive'] is False


def test_folds_are_purged(sparse_panel,sparse_cfg):
    groups=list(sparse_panel[sparse_panel.eligible].groupby('date'))[:240]
    folds=inner_folds(groups,5,sparse_cfg)
    assert len(folds)>=2
    for train,test in folds:
        assert max(g.exit_date_5.max() for _,g in train)<test[0][0]
        for i in range(1,len(test)):assert test[i-1][1].exit_date_5.max()<test[i][0]


def test_sparse_fit_does_not_use_outer_future(sparse_panel,sparse_cfg):
    cut=sorted(sparse_panel.date.unique())[300]
    small=sparse_panel.copy()
    # Limit test complexity with constant other factors, not altered labels.
    for k in FACTORS:
        if k not in ('rsi_rebound','trend','relative_strength'):small[k]=50.
    a,ai,at=sparse_fit(small,cut,5,sparse_cfg)
    changed=small.copy();changed.loc[changed.date>=cut,'net_5']=999
    b,bi,bt=sparse_fit(changed,cut,5,sparse_cfg)
    assert a==b and ai==bi and at==bt
    assert ai.get('challenger_weights',{}).get('rsi_rebound',0)>0
    assert all(ai['challenger_weights'][k]==0 for k in TECH)


def test_fit_adopts_when_independent_synthetic_gate_passes(sparse_panel,sparse_cfg):
    x=sparse_panel.copy()
    for k in FACTORS:
        if k not in ('rsi_rebound','trend','relative_strength'):x[k]=50.
    w,info,_=sparse_fit(x,sorted(x.date.unique())[310],5,sparse_cfg)
    assert info['status']=='SPARSE_INNER_VALIDATION_ACCEPTED'
    assert w['rsi_rebound']==1 and not info['adopted_is_legacy_control']


def test_strict_missing_selected_factor_not_renormalized(sparse_panel,sparse_cfg):
    d=sparse_panel.date.iloc[0];g=sparse_panel[sparse_panel.date==d].copy()
    g['trend']=10.;g['rsi_rebound']=10.
    one=g.index[0];g.loc[one,'trend']=100.;g.loc[one,'rsi_rebound']=np.nan
    weights={'trend':.8,'rsi_rebound':.2}
    assert score_rows(g,weights)[0].loc[one]>50
    ranked=rank_date(g,d,weights,sparse_cfg)
    assert g.loc[one,'ticker'] not in set(ranked.ticker)


def test_prepared_matches_actual_sparse_ranking(sparse_panel,sparse_cfg):
    groups=list(sparse_panel[sparse_panel.eligible].groupby('date'))[:5]
    w={k:0. for k in FACTORS};w.update(rsi_rebound=.7,trend=.3)
    calc=Prepared(groups,sparse_panel,FACTORS,5,sparse_cfg).evaluate(w)
    for i,(d,g) in enumerate(groups):
        selected=rank_date(g,d,w,sparse_cfg)
        assert calc['net'][i]==pytest.approx(selected.net_5.mean())


def test_selected_missing_outcome_invalidates_not_replaced(sparse_panel,sparse_cfg):
    d=sparse_panel.date.iloc[0];g=sparse_panel[sparse_panel.date==d].copy()
    j=g[g.eligible].rsi_rebound.idxmax();g.loc[j,'net_5']=np.nan
    x=Prepared([(d,g[g.eligible])],g,FACTORS,5,sparse_cfg).evaluate({'rsi_rebound':1.})
    assert np.isnan(x['net'][0])


def test_atomic_features_prefix_invariant(prices,cfg):
    c=replace(cfg,factor_design='sparse_v194');cut=sorted(prices.date.unique())[250]
    full=build_panel(prices,cfg=c);past=build_panel(prices[prices.date<=cut],cfg=c)
    cols=['date','ticker']+list(ATOMIC)+list(INTERACTIONS)
    pd.testing.assert_frame_equal(full[full.date<=cut][cols].reset_index(drop=True),past[cols].reset_index(drop=True))

@pytest.mark.parametrize('key',ATOMIC)
def test_atomic_scale(panel,key):
    values=panel[key].dropna();assert len(values)>50 and values.between(0,100).all()


def test_financial_date_not_misused_as_publication():
    data={'TaiwanStockFinancialStatements':[{'date':'2024-03-31','type':'GrossProfit','value':30},
                                           {'date':'2024-03-31','type':'Revenue','value':100}]}
    f,omitted=financial_records('2330.TW',data,Settings())
    assert f.empty and omitted[0]['reason']=='PUBLICATION_TIME_MISSING_CACHED_NOT_SCORED'
    lag,_=financial_records('2330.TW',data,replace(Settings(),financial_timing='lagged_research'))
    assert len(lag)==1 and pd.Timestamp(lag.available_at.iloc[0])>=pd.Timestamp('2024-03-31')+pd.Timedelta(days=120)
    assert 'NOT_PIT' in lag.timing.iloc[0]


def test_statement_availability_uses_max_input_dates():
    data={'TaiwanStockFinancialStatements':[
      {'date':'2024-03-31','type':'EPS','value':2,'published_at':'2024-05-15'},
      {'date':'2023-03-31','type':'EPS','value':1,'published_at':'2024-06-01'}]}
    f,_=financial_records('2330.TW',data,Settings())
    r=f[f.date=='2024-03-31'].iloc[0]
    assert r.eps_growth==100 and pd.Timestamp(r.available_at)>pd.Timestamp('2024-06-01')


def test_negative_earnings_base_does_not_fake_growth():
    data={'TaiwanStockFinancialStatements':[
      {'date':'2024-03-31','type':'EPS','value':2,'published_at':'2024-05-15'},
      {'date':'2023-03-31','type':'EPS','value':-1,'published_at':'2023-05-15'}]}
    f,_=financial_records('2330.TW',data,Settings())
    assert f.empty or 'eps_growth' not in f


def test_daily_pe_next_day_and_invalid_missing():
    f,_=financial_records('2330.TW',{'TaiwanStockPER':[{'date':'2024-05-01','PER':25},
        {'date':'2024-05-02','PER':-1}]},Settings())
    assert len(f)==1 and f.pe_ratio.iloc[0]==25 and str(f.available_at.iloc[0]).startswith('2024-05-02')


def test_proxy_filtered_in_strict_mode(prices,cfg):
    ev=pd.DataFrame([{'ticker':str(prices.ticker.iloc[0]),'available_at':str(prices.date.min()),
        'source':'vendor','timing':'ASSUMED_120D_PERIOD_LAG_NOT_PIT','gross_margin':60}])
    strict=build_panel(prices,ev,replace(cfg,optional_data=True,financial_timing='disclosure_required'))
    assert strict.gross_margin.isna().all()
    lag=build_panel(prices,ev,replace(cfg,optional_data=True,financial_timing='lagged_research'))
    assert lag.gross_margin.notna().any()


def test_new_state_table_and_no_fit_ui(ui,panel,cfg):
    mod, fake=ui
    mod.page_decisions(panel,{},replace(cfg,factor_design='sparse_v194'))
    assert not any(name=='error' for name,_,_ in fake.calls)
    assert any(name=='dataframe' and len(args[0])==len(FACTORS) for name,args,k in fake.calls)


def test_checkpoint_includes_new_algorithm():
    import v192_checkpoint
    text=Path(v192_checkpoint.__file__).read_text()
    for key in ('v194_selection.py','v194_factors.py','v194_evidence.py','v193_factors.py'):assert key in text

@pytest.mark.parametrize('field,value',[('sparse_max_inputs',0),('sparse_max_inputs',7),('sparse_budget',1),
                                       ('financial_timing','pretend_pit'),('statement_lag_days',30)])
def test_invalid_new_settings_rejected(field,value):
    with pytest.raises(ValueError):replace(Settings(),**{field:value})


def test_export_contains_explicit_zeros(tmp_path):
    from v194_reporting import export_sparse
    report={'fits':[{'cutoff':'2025-01-02','horizon':5,'challenger_weights':{'rsi_rebound':1.},
                      'weights':{'rsi_rebound':1.},'status':'synthetic_test'}]}
    export_sparse(tmp_path,report)
    table=pd.read_csv(tmp_path/'sparse_factor_selection.csv')
    assert table[table.factor.isin(TECH)].candidate_weight.eq(0).all()
    assert json.loads((tmp_path/'sparse_protocol.json').read_text())['deleted_factors_restored_by_shrinkage'] is False


def test_additional_online_datasets_are_cached(tmp_path):
    from v194_evidence import additional_evidence, DATASETS
    class Client:
        root=tmp_path
        cfg=Settings()
        events=[]
        calls=[]
        def finmind(self,dataset,code,start,end):
            self.calls.append(dataset)
            if dataset=='TaiwanStockPER':return {'status':200,'data':[{'date':'2025-01-02','PER':20}]}
            return {'status':200,'data':[{'date':'2024-09-30','type':'Revenue','value':100}]}
    client=Client()
    one=additional_evidence(client,'2330.TW','2024-01-01','2025-01-10')
    two=additional_evidence(client,'2330.TW','2024-01-01','2025-01-10')
    assert client.calls==list(DATASETS)
    assert one.pe_ratio.iloc[0]==20
    pd.testing.assert_frame_equal(one,two)
    assert any(e.get('reason')=='PUBLICATION_TIME_MISSING_CACHED_NOT_SCORED' for e in client.events)


def test_one_optional_endpoint_failure_does_not_stop_other_datasets(tmp_path):
    from v194_evidence import additional_evidence
    from v19_data import SourceError
    class Client:
        root=tmp_path;cfg=Settings();events=[]
        def finmind(self,dataset,code,start,end):
            if dataset!='TaiwanStockPER':raise SourceError('No entitlement')
            return {'status':200,'data':[{'date':'2025-01-02','PER':20}]}
    client=Client();result=additional_evidence(client,'2330.TW','2024-01-01','2025-01-10')
    assert len(result)==1 and len(client.events)==3


def test_imported_evidence_switch_and_deduplication(tmp_path):
    from v19_service import merge_imported_evidence
    ev=pd.DataFrame([{'ticker':'2330.TW','available_at':'2025-01-02','source':'uploaded','gross_margin':50}])
    ev.to_csv(tmp_path/'optional_evidence.csv',index=False)
    assert merge_imported_evidence(tmp_path,ev,Settings(optional_data=False)).empty
    result=merge_imported_evidence(tmp_path,ev,Settings(optional_data=True))
    assert len(result)==1


def test_online_worker_includes_imported_evidence(tmp_path,monkeypatch):
    import v19_service as service
    ev=pd.DataFrame([{'ticker':'2330.TW','available_at':'2025-01-02','source':'uploaded','gross_margin':50}])
    ev.to_csv(tmp_path/'optional_evidence.csv',index=False)
    calls=[]
    monkeypatch.setattr(service,'download_history',lambda *a,**k:(pd.DataFrame([{'ticker':'2330.TW'}]),pd.DataFrame(),{}))
    def analyze(prices,evidence,*args,**kwargs):
        calls.append(evidence.copy())
        return {'status':'TEST_ONLY','cutoffs':['2025-01-03']},pd.DataFrame(),pd.DataFrame()
    monkeypatch.setattr(service,'analyze',analyze)
    monkeypatch.setattr(service,'save_analysis',lambda *a:None)
    service.start_job(tmp_path,Settings(optional_data=True),mode='download_validate')
    with service._LOCK:thread=service._JOBS[str(tmp_path.resolve())]
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(calls)==1 and len(calls[0])==1 and calls[0].source.iloc[0]=='uploaded'


def test_changed_factor_limit_does_not_reuse_previous_fit(ui,sparse_cfg):
    from dataclasses import asdict
    from v19_config import VERSION
    from v194_ui import choose_model
    mod,fake=ui
    report={'version':VERSION,'settings':asdict(sparse_cfg),'weights':{'5':{'weights':{'rsi_rebound':1.},
            'fit':{'factor_design':'sparse_v194','cutoff':'2025-01-02','challenger_weights':{'rsi_rebound':1.},
                   'status':'SPARSE_INNER_VALIDATION_ACCEPTED'}}}}
    good,_=choose_model(report,5,sparse_cfg,'2025-01-03',mod.LABELS)
    assert good.get('rsi_rebound')==1.
    changed=replace(sparse_cfg,sparse_max_inputs=2)
    rejected,scoring=choose_model(report,5,changed,'2025-01-03',mod.LABELS)
    assert rejected.get('rsi_rebound',0)==0 and scoring.factor_design=='legacy_v192'


def test_candidate_table_uses_readable_percentage(ui,panel,cfg):
    from v19_config import VERSION
    mod,fake=ui
    report={'version':VERSION,'results':[{'date':'2025-01-02','horizon':5,
              'strategy':'training_challenger_research','mean_net':.1234}], 'settings':{}}
    mod.page_validation(panel,report,cfg)
    tables=[args[0] for name,args,_ in fake.calls if name=='dataframe' and args]
    label='\u4e8b\u524d\u8a13\u7df4\u5019\u9078\uff08\u672a\u5fc5\u63a1\u7528\uff09'
    assert any(label in table and table[label].iloc[0]=='+12.34%' for table in tables)


def test_cash_conversion_requires_annual_alignment():
    fs=[]
    for day,stamp in [('2024-03-31','2024-05-15'),('2024-06-30','2024-08-15'),
                      ('2024-09-30','2024-11-15'),('2024-12-31','2025-03-20')]:
        fs.append({'date':day,'type':'IncomeAfterTaxes','value':25,'published_at':stamp})
    cash=[{'date':'2024-12-31','type':'CashFlowsFromOperatingActivities','value':120,'published_at':'2025-03-21'},
          {'date':'2024-06-30','type':'CashFlowsFromOperatingActivities','value':60,'published_at':'2024-08-15'}]
    result,_=financial_records('2330.TW',{'TaiwanStockFinancialStatements':fs,'TaiwanStockCashFlowsStatement':cash},Settings())
    good=result[result.cash_conversion_ratio.notna()]
    assert len(good)==1 and good.cash_conversion_ratio.iloc[0]==pytest.approx(1.2)
    assert pd.Timestamp(good.available_at.iloc[0])>pd.Timestamp('2025-03-21')


def test_missing_financial_values_never_imputed(panel):
    for k in BUSINESS:assert panel[k].isna().all()
