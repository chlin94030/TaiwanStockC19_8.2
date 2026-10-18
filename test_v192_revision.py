"""V19.2 offline tests; fixtures are not market returns."""
from dataclasses import replace
from pathlib import Path
import json
import sqlite3
import time
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pytest
import requests
from test_v19_core import prices, cfg, panel
from test_v19_ui import ui
from v19_config import Settings, TECH, VERSION
from v19_backtest import analyze, add_outcomes, calendar_for, save_analysis
from v19_features import build_panel
from v19_data import Client, SourceError, atomic_json, atomic_csv, read_json, clean_prices
from v192_sampling import cutoff_plan, fit_seed, disjoint_dates
from v192_checkpoint import run_identity, CaseStore, RunCancelled
from v192_factors import attach_signals, case_factors, summary, _rank_ic
from v192_history import export_database


def test_new_defaults_are_100_dates_not_100_trials():
    s=Settings()
    assert s.dates==100 and s.trials==24 and s.stocks==600
    assert s.sampling_method=='uniform' and not s.replay_cutoffs


@pytest.mark.parametrize('method',['uniform','stratified'])
def test_exact_100_unique_mature_and_deterministic(panel,cfg,method):
    cfg=replace(cfg,dates=100,sampling_method=method)
    cal=calendar_for(panel,cfg)
    a,plan=cutoff_plan(panel,cal,cfg);b,_=cutoff_plan(panel.sample(frac=1,random_state=4),cal,cfg)
    assert len(a)==100 and len(set(a))==100 and a==b and a==sorted(a)
    assert all(cal.get_loc(d)+60<len(cal) for d in a)
    assert plan['actual']==plan['requested']==100 and not plan['selection_uses_future_returns']


def test_sampling_does_not_select_only_successful_prices(panel,cfg):
    cfg=replace(cfg,dates=100);cal=calendar_for(panel,cfg)
    expected,_=cutoff_plan(panel,cal,cfg)
    bad=panel.copy()
    for h in cfg.horizons:bad[f'net_{h}']=np.nan
    assert cutoff_plan(bad,cal,cfg)[0]==expected


def test_short_history_explicitly_not_silently_ten(panel,cfg):
    cfg=replace(cfg,dates=100)
    p=panel[panel.date.isin(sorted(panel.date.unique())[:145])]
    d,plan=cutoff_plan(p,calendar_for(p,cfg),cfg)
    assert not d and plan['requested']==100 and plan['actual']==0


def test_seed_variation_and_fit_seed_stability(panel,cfg):
    cfg=replace(cfg,dates=100);cal=calendar_for(panel,cfg)
    a,_=cutoff_plan(panel,cal,cfg);b,_=cutoff_plan(panel,cal,replace(cfg,seed=1950))
    assert a!=b and fit_seed(5,a[0],20)==fit_seed(5,a[0],20)
    assert fit_seed(5,a[0],20)!=fit_seed(5,a[0],60)


def test_disjoint_cases_do_not_claim_hundred_independent(panel,cfg):
    cfg=replace(cfg,dates=100);cal=calendar_for(panel,cfg)
    d,p=cutoff_plan(panel,cal,cfg)
    assert p['disjoint_case_count']['60']<100
    x=disjoint_dates(d,cal,60)
    assert all(cal.get_loc(x[i+1])>cal.get_loc(x[i])+60 for i in range(len(x)-1))


def test_stock_pool_seed_is_independent(tmp_path):
    rows=[{'ticker':f'{2100+i}.TW','name':str(i),'sector':'test'} for i in range(30)]
    atomic_json(tmp_path/'universe.json',{'at':time.time(),'rows':rows,'scope':'TEST_ONLY'})
    a=Client(tmp_path,Settings(stocks=15,seed=1));b=Client(tmp_path,Settings(stocks=15,seed=999))
    assert a.universe()==b.universe()


def test_pairwise_spearman_ranks_only_observed_pairs():
    x=pd.Series([2.,1.,np.nan,5.,4.,6.,10.,8.,7.,11.,9.,12.])
    y=pd.Series([5.,3.,2.,4.,np.nan,6.,11.,7.,8.,12.,9.,10.])
    value,count=_rank_ic(x,y)
    pair=pd.DataFrame({'x':x,'y':y}).dropna().rank()
    assert count==10 and abs(value-pair.x.corr(pair.y))<1e-12


def test_factors_missing_not_zero_and_binary_separate(panel,cfg):
    p=attach_signals(panel);date=sorted(p.date.unique())[220];day=p[p.date==date]
    w={k:1/len(TECH) for k in TECH}
    r=case_factors(day,date,5,cfg,w)
    by={x['factor']:x for x in r}
    assert by['revenue']['coverage']==0 and by['revenue']['rank_ic'] is None
    assert by['revenue']['status']=='OPTIONAL_DISABLED_NOT_TESTED'
    assert by['macd_hist_flip_signal']['kind']=='binary_signal'
    assert by['macd_hist_flip_signal']['full_minus_removed'] is None
    assert all(x['report_only_not_training'] for x in r)


def test_missing_top_factor_outcome_not_replaced(panel,cfg):
    p=attach_signals(panel);date=sorted(p.date.unique())[220];day=p[p.date==date].copy()
    ix=day[day.eligible].sort_values(['trend','ticker'],ascending=[False,True]).index[0]
    day.loc[ix,'net_5']=np.nan
    r=case_factors(day,date,5,cfg,{k:1/6 for k in TECH})
    assert [x for x in r if x['factor']=='trend'][0]['top_mean_net'] is None


def test_signals_prefix_causal(panel):
    p=panel[panel.ticker=='2101.TW'].copy();a=attach_signals(p);b=attach_signals(p.head(200))
    names=[c for c in a if c.endswith('_signal')]
    pd.testing.assert_frame_equal(a[names].head(200).reset_index(drop=True),b[names].reset_index(drop=True))


def test_database_export_integrity_no_credentials(tmp_path,prices,cfg):
    out=export_database(tmp_path,prices,pd.DataFrame(),{'scope':'SYNTHETIC_TEST_ONLY'},cfg)
    with sqlite3.connect(f'file:{tmp_path}/history.sqlite?mode=ro',uri=True) as c:
        assert c.execute('SELECT count(*) FROM prices').fetchone()[0]==len(prices)
        assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    with ZipFile(out) as z:
        assert 'history.sqlite' in z.namelist() and 'input_prices.csv.gz' in z.namelist()
        assert not any('secrets' in name or 'token' in name for name in z.namelist())
    from v191_reporting import read_bundle
    restored,ev,manifest=read_bundle(out)
    assert len(restored)==len(prices) and ev.empty


def test_same_range_cache_persists_beyond_eight_hours(tmp_path,prices,cfg):
    p=prices[prices.ticker=='2101.TW'];cache=tmp_path/'prices'/'2101.TW.csv.gz'
    atomic_csv(cache,p)
    atomic_json(cache.with_suffix('.json'),{'requested_start':cfg.start(),'requested_end':cfg.end(),
                  'at':time.time()-864000,'adjustment_policy':'v191_prefer_adjusted','source':'SYNTHETIC_TEST_ONLY'})
    def never(*a,**k):raise AssertionError('unnecessary network call')
    got,meta=Client(tmp_path,cfg,transport=never).prices('2101.TW',cfg.start(),cfg.end())
    assert len(got)==len(p) and meta['cached'] and meta['state']=='CACHED_RANGE'


class Response:
    def __init__(self,code=200,body=None):self.status_code=code;self.body=body or {'status':200,'data':[]};self.headers={}
    def json(self):return self.body
    def raise_for_status(self):pass


def test_retry_transient_server_but_not_rate_limit(tmp_path,monkeypatch):
    monkeypatch.setattr('v19_data.time.sleep',lambda _:None)
    replies=iter([Response(503),Response(200)])
    c=Client(tmp_path,transport=lambda *a,**k:next(replies))
    assert c.request('https://example.invalid',provider='FinMind')['status']==200 and c.calls==2
    limited=Client(tmp_path/'limited',transport=lambda *a,**k:Response(429))
    with pytest.raises(SourceError):limited.request('https://example.invalid',provider='FinMind')
    assert limited.calls==1
    again=Client(tmp_path/'limited',transport=lambda *a,**k:pytest.fail('cooldown must prevent HTTP'))
    with pytest.raises(SourceError):again.request('https://example.invalid',provider='FinMind')
    assert again.calls==0


def test_dataset_entitlement_does_not_block_other_dataset(tmp_path):
    replies=iter([Response(403),Response(200)])
    c=Client(tmp_path,transport=lambda *a,**k:next(replies))
    with pytest.raises(SourceError):c.finmind('PremiumThing')
    assert c.finmind('TaiwanStockPrice')['status']==200


def test_optional_dataset_resume_not_combined_cache_shortcircuit(tmp_path,cfg,monkeypatch):
    c=Client(tmp_path,replace(cfg,optional_data=True));calls=[]
    def fetch(dataset,*args):
        calls.append(dataset)
        if 'MonthRevenue' in dataset and calls.count(dataset)==1:raise SourceError('transient failure')
        if 'Institutional' in dataset:
            return {'status':200,'data':[{'date':'2025-01-02','name':'Foreign_Investor','buy':50,'sell':20}]}
        return {'status':200,'data':[]}
    monkeypatch.setattr(c,'finmind',fetch)
    first=c.evidence('2101.TW',cfg.start(),cfg.end());second=c.evidence('2101.TW',cfg.start(),cfg.end())
    assert not first.empty and not second.empty
    assert calls.count('TaiwanStockInstitutionalInvestorsBuySell')==1
    assert calls.count('TaiwanStockMonthRevenue')==2
    assert calls.count('TaiwanStockMarginPurchaseShortSale')==1


def test_checkpoint_data_evidence_and_config_fingerprints(tmp_path,prices,cfg):
    i,m=run_identity(prices,None,cfg);store=CaseStore(tmp_path,i,m)
    payload={'records':[],'trades':[],'fit':{'status':'test'},'trials':[],'factors':[]}
    store.save('2025-01-02',5,payload)
    assert store.load('2025-01-02',5)==payload
    changed=prices.copy();changed.loc[0,'close']+=.001
    assert run_identity(changed,None,cfg)[0]!=i
    assert run_identity(prices,None,replace(cfg,seed=13))[0]!=i
    ev=pd.DataFrame([{'ticker':'2101.TW','available_at':'2024-01-05','revenue_yoy':15}])
    cfg2=replace(cfg,optional_data=True)
    assert run_identity(prices,ev,cfg2)[0]!=run_identity(prices,None,cfg2)[0]
    f=next(store.root.glob('2025*.json'));obj=read_json(f);obj['payload']['fit']['status']='TAMPERED';atomic_json(f,obj)
    assert store.load('2025-01-02',5) is None


def test_pause_and_resume_exact_cases(prices,cfg,tmp_path):
    c=replace(cfg,dates=2,trials=4);stop=[False]
    def progress(stage,done,total,message):
        if stage=='validate' and done>=2:stop[0]=True
    with pytest.raises(RunCancelled):analyze(prices,cfg=c,progress=progress,checkpoint_dir=tmp_path,cancel_check=lambda:stop[0])
    files=list(tmp_path.glob('*/20*.json'));assert len(files)==2
    r,p,t=analyze(prices,cfg=c,checkpoint_dir=tmp_path)
    assert r['resumed_cases']==2 and r['expected_adaptive_cases']==6
    from v19_config import FACTORS
    from v192_factors import SIGNALS
    assert len(r['factor_cases'])==6*(len(FACTORS)+len(SIGNALS))
    output=tmp_path/'report';save_analysis(output,r,p,t,prices)
    assert (output/'factor_validation_summary.csv').is_file()
    with ZipFile(output/'v192_diagnostic_bundle.zip') as z:
        assert {'sampling_plan.json','factor_validation_cases.csv','input_prices.csv.gz'}.issubset(z.namelist())


def test_main_default_starts_100_not_old_ten_replay(ui,panel,monkeypatch):
    m,f=ui;calls=[]
    original=f.__getattr__
    def button(*a,**k):return k.get('key')=='v19_start'
    f.button=button
    monkeypatch.setattr(m,'start_job',lambda root,cfg,mode:(calls.append((cfg,mode)) or {'state':'TEST_ONLY'}))
    monkeypatch.setattr(m,'_display_data',lambda *a,**k:(panel,{}));monkeypatch.setattr(m,'news_block',lambda:None)
    m.main()
    assert len(calls)==1 and calls[0][0].dates==100 and not calls[0][0].replay_cutoffs


def test_optional_summary_shows_not_tested(panel,cfg):
    p=attach_signals(panel);date=sorted(p.date.unique())[220]
    cases=case_factors(p[p.date==date],date,20,cfg,{k:1/6 for k in TECH})
    rows=summary(cases,calendar_for(p,cfg),cfg)
    assert [r for r in rows if r['factor']=='quality'][0]['status']=='NOT_TESTED_NO_DATA'


def test_new_version_and_trading_gates_preserved():
    assert VERSION=='19.8'
    assert Settings().validation_blocks==4 and Settings().weight_shrinkage==.5


def test_cache_only_stock_limit_does_not_silently_ignore_setting(tmp_path,prices,cfg):
    from v19_service import load_inputs
    atomic_csv(tmp_path/'history_prices.csv.gz',prices)
    p,e=load_inputs(tmp_path,replace(cfg,stocks=10))
    assert p.ticker.nunique()==11 and cfg.benchmark in set(p.ticker)
    # A different cutoff seed must not change the equity pool.
    q,_=load_inputs(tmp_path,replace(cfg,stocks=10,seed=17))
    assert set(p.ticker)==set(q.ticker)


def test_online_download_builds_database_once_not_per_cutoff(tmp_path):
    from v19_data import download_history
    calls=[]
    def transport(url,params=None,**kwargs):
        calls.append(dict(params or {}))
        code=params['data_id']
        body={'status':200,'data':[
            {'date':d,'stock_id':code,'open':100,'max':102,'min':99,'close':101,'Trading_Volume':1000000}
            for d in ('2025-01-02','2025-01-03','2025-01-06')]}
        return Response(body=body)
    symbols=[f'{2100+i}.TW' for i in range(5)]
    cfg=Settings(stocks=5,dates=100,as_of='2025-01-06',prefer_adjusted=False)
    p,e,m=download_history(tmp_path,cfg,transport=transport,tickers=symbols)
    assert len(calls)==6 and p.ticker.nunique()==6
    assert (tmp_path/'v192_history_database.zip').is_file()
    q,_,m2=download_history(tmp_path,cfg,transport=transport,tickers=symbols)
    assert len(calls)==6 and len(q)==len(p) and m2['requests']==0


def test_replay_10_requires_explicit_10_and_cannot_masquerade_as_100(panel,cfg):
    d=sorted(panel.date.unique())[200:210]
    with pytest.raises(ValueError):replace(cfg,dates=100,replay_cutoffs=tuple(str(pd.Timestamp(x).date()) for x in d))
    c=replace(cfg,dates=10,replay_cutoffs=tuple(str(pd.Timestamp(x).date()) for x in d))
    dates,plan=cutoff_plan(panel,calendar_for(panel,c),c)
    assert len(dates)==10 and plan['method']=='explicit_development_replay'


def test_download_request_budget_respected_across_retries(tmp_path,monkeypatch):
    from v19_data import BudgetExceeded
    monkeypatch.setattr('v19_data.time.sleep',lambda _:None)
    c=Client(tmp_path,Settings(request_budget=1),transport=lambda *a,**k:Response(503))
    with pytest.raises(BudgetExceeded):c.request('https://example.invalid',provider='FinMind')
    assert c.calls==1


@pytest.mark.parametrize("value", [[], None, "bad-shape"])
def test_checkpoint_valid_json_wrong_shape_ignored(tmp_path,value):
    import json
    store=CaseStore(tmp_path,"identity",{})
    (store.root/"2025-01-02_5.json").write_text(json.dumps(value))
    assert store.load("2025-01-02",5) is None
