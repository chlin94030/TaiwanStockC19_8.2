"""V19.1 report-driven regression tests. Fixtures are explicitly synthetic."""
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import json
import gzip
import numpy as np
import pandas as pd
import pytest
from test_v19_core import prices, cfg, panel
from v19_config import Settings, TECH, prior
from v19_features import build_panel,technical_frame,current_candidates,score_rows
from v19_backtest import add_outcomes,fit_before,selected_result,sample_cutoffs,calendar_for,analyze,save_analysis
from v19_data import Client,SourceError,atomic_csv,atomic_json
from v19_service import weights_for,load_inputs
from v191_selection import stability_gate,portfolio_matrix,paired_cohort,separated_validation
from v191_audit import parse_markdown,audit_markdown,paired_summary
from v191_reporting import summarize_results,case_rows,read_bundle


def test_supplied_report_audit_exact():
    text=Path(__file__).with_name('reference_validation_v190.md').read_text()
    r=audit_markdown(text);m={x['horizon']:x for x in r['summary']}
    assert len(r['records'])==30 and len(r['cutoffs'])==10
    assert m[5]['adaptive_mean']==pytest.approx(-.0119222222222222)
    assert m[20]['weight_change_mean']==pytest.approx(-.01314444444444)
    assert m[60]['mean_excess']==pytest.approx(-.181942857142857)
    assert [m[h]['matched_benchmark_cases'] for h in (5,20,60)]==[8,8,7]
    assert [m[h]['benchmark_beats'] for h in (5,20,60)]==[2,3,1]
    assert r['settings']['optional_data'] is False


def test_audit_zero_is_not_missing():
    r=paired_summary([{'date':'2025-01-02','horizon':5,'baseline':0.,'adaptive':0.,'benchmark':0.,'complete':True}])[0]
    assert r['matched_benchmark_cases']==1 and r['mean_excess']==0


def test_audit_incomplete_is_not_average_or_zero():
    rows=[dict(date='2025-01-02',horizon=5,baseline=.1,adaptive=.2,benchmark=None,complete=True),
          dict(date='2025-02-03',horizon=5,baseline=.9,adaptive=.9,benchmark=.1,complete=False)]
    r=paired_summary(rows)[0]
    assert r['adaptive_mean']==.2 and r['matched_benchmark_cases']==0 and r['mean_excess'] is None


@pytest.mark.parametrize('text',['nothing','| 2025-01-02 | 5 | surprise | +1.0% | +0.0% | True |',
    '| 2025-01-02 | 5 | +1.0% | +1.0% | +0.0% | True |\n| 2025-01-02 | 5 | +1.0% | +1.0% | +0.0% | True |'])
def test_bad_audit_rejected(text):
    with pytest.raises(ValueError):parse_markdown(text)


def _signal(prices,index=100,ticker='2101.TW'):
    return prices[prices.ticker==ticker].sort_values('date').iloc[index].date


def test_missing_entry_is_unknown_not_free_cash(prices,cfg):
    entry=_signal(prices,101);raw=prices[~((prices.ticker=='2101.TW')&(prices.date==entry))]
    p,_=add_outcomes(build_panel(raw,cfg=cfg),cfg)
    r=p[(p.ticker=='2101.TW')&(p.date==_signal(prices))].iloc[0]
    assert pd.isna(r.net_5) and r.outcome_state_5=='NO_ENTRY_BAR'


def test_interior_gap_keeps_endpoints_but_not_path_risk(prices,cfg):
    p,_=add_outcomes(build_panel(prices,cfg=cfg),cfg)
    r=p[(p.ticker=='2101.TW')&(p.date==_signal(prices))].iloc[0]
    hole=_signal(prices,103);raw=prices[~((prices.ticker=='2101.TW')&(prices.date==hole))]
    q,_=add_outcomes(build_panel(raw,cfg=cfg),cfg)
    x=q[(q.ticker=='2101.TW')&(q.date==_signal(prices))].iloc[0]
    assert x.net_5==pytest.approx(r.net_5)
    assert pd.isna(x.mae_5) and pd.isna(x.mdd_5)
    assert x.outcome_state_5=='ASSUMED_OPEN_FILL_PATH_INCOMPLETE'
    assert not x.path_complete_5 and x.missing_path_bars_5==1


def test_benchmark_halt_does_not_erase_stock_rs(prices,cfg):
    day=_signal(prices,120)
    raw=prices[~((prices.ticker==cfg.benchmark)&(prices.date==day))]
    p=build_panel(raw,cfg=cfg);g=p[p.date==day]
    assert g.relative_strength.notna().all() and g.eligible.any()
    assert set(g.benchmark_kind)=={'CROSS_SECTION_MEDIAN_NOT_INDEX'}


def test_adjustment_boundary_does_not_fabricate_fifty_percent_drop(prices):
    g=prices[prices.ticker=='2101.TW'].head(180).copy().reset_index(drop=True)
    g['adj_close']=g.close*.5;g.loc[100:,'adj_close']=np.nan
    f=technical_frame(g)
    assert f.ret1.iloc[100]==pytest.approx(g.close.iloc[100]/g.close.iloc[99]-1)
    assert abs(f.ret1.iloc[100])<.1 and f.adjustment_transition.iloc[100]


def test_verified_split_gap_uses_comparable_price(prices,cfg):
    raw=prices.copy();mask=raw.ticker=='2101.TW';day=_signal(prices,101)
    after=mask&(raw.date>=day)
    for c in ('open','high','low','close'):raw.loc[after,c]/=4
    p,_=add_outcomes(build_panel(raw,cfg=cfg),cfg)
    r=p[(p.ticker=='2101.TW')&(p.date==_signal(prices))].iloc[0]
    assert r.entry_raw_gap < -.7 and abs(r.entry_gap)<.1
    assert r.return_basis_5=='ADJUSTED_PROVIDER' and pd.notna(r.net_5)


def test_benchmark_does_not_use_stock_chase_gap_cap(prices,cfg):
    raw=prices.copy();ix=raw.index[(raw.ticker==cfg.benchmark)&(raw.date==_signal(prices,101))][0]
    prev=raw.loc[ix-1,'close']
    raw.loc[ix,['open','high','low','close','adj_close']]=[prev*1.05,prev*1.06,prev*1.01,prev*1.03,prev*1.03]
    p,_=add_outcomes(build_panel(raw,cfg=cfg),cfg)
    r=p[(p.ticker==cfg.benchmark)&(p.date==_signal(prices))].iloc[0]
    assert r.fill_state=='ASSUMED_OPEN_FILL' and r.net_5!=0


def test_optional_disabled_does_not_read_cached_fundamentals(prices,cfg,tmp_path):
    ev=pd.DataFrame([dict(ticker='2101.TW',available_at='2024-01-01',source='SYNTHETIC_TEST_ONLY',revenue_yoy=50,eps_ttm=10,gross_margin=50)])
    atomic_csv(tmp_path/'historical_evidence.csv.gz',ev)
    atomic_csv(tmp_path/'history_prices.csv.gz',prices)
    _,read=load_inputs(tmp_path,cfg)
    assert read.empty
    p=build_panel(prices,ev,cfg)
    assert p.revenue.isna().all()


def gate(c,b=None,bm=None,mae=None,bmae=None,fills=None):
    n=len(c)
    return stability_gate(c,b if b is not None else np.zeros(n),bm if bm is not None else np.zeros(n),
              mae if mae is not None else np.full(n,-.03),bmae if bmae is not None else np.full(n,-.03),
              fills if fills is not None else np.ones(n),Settings())


def test_stable_candidate_can_pass():assert gate([.015,.02,.01,.025,.03])['accepted']

def test_one_large_winner_cannot_drive_promotion():assert not gate([.50,-.01,-.01,-.01,-.01])['accepted']

def test_better_than_bad_baseline_still_must_beat_market():
    r=gate([.01]*5,b=[-.01]*5,bm=[.03]*5)
    assert not r['accepted'] and not r['tests']['beats_benchmark_on_average']

def test_risk_deterioration_blocks_candidate():assert not gate([.02]*5,mae=[-.12]*5)['accepted']

def test_sparse_fill_blocks_candidate():assert not gate([.02]*5,fills=[.2]*5)['accepted']

def test_three_overlapping_dates_not_enough_blocks():assert not gate([.1]*3)['accepted']


def test_pairing_does_not_give_each_candidate_its_own_cohort():
    m={'net':np.array([[.1,np.nan,.03],[.1,.02,.03]]),
       'mae':np.full((2,3),-.03),'basis':np.full((2,3),'ADJUSTED_PROVIDER',dtype=object)}
    common=paired_cohort(m,np.array([.01,.02,.03]),np.array(['ADJUSTED_PROVIDER']*3))
    assert common.tolist()==[True,False,True]


def test_mixed_or_mismatching_price_basis_not_promoted():
    m={'net':np.ones((2,3))*.01,'mae':np.ones((2,3))*-.02,
       'basis':np.array([['RAW_PRICE_ONLY','ADJUSTED_PROVIDER','MIXED_UNVERIFIED']]*2)}
    assert paired_cohort(m,np.zeros(3),np.array(['ADJUSTED_PROVIDER']*3)).tolist()==[False,True,False]


def test_vectorized_same_selection_as_display(panel,cfg):
    ds=sorted(panel[panel.eligible].date.unique())[100:105]
    gs=[panel[(panel.date==d)&panel.eligible] for d in ds]
    w=prior(5,TECH);matrix=portfolio_matrix(gs,[w],5,cfg)
    for j,g in enumerate(gs):
        s,_=score_rows(g,w)
        q=g.assign(score=s).sort_values(['score','ticker'],ascending=[False,True]).head(cfg.top_k)
        assert matrix['net'][0,j]==pytest.approx(q.net_5.mean())


@pytest.mark.parametrize('h',[5,20,60])
def test_validation_blocks_do_not_overlap(panel,cfg,h):
    cutoff=panel.date.max()
    available=panel[(panel.date<cutoff)&(panel[f'exit_date_{h}']<cutoff)&panel.eligible]
    grouped=[(d,g) for d,g in available.groupby('date') if len(g)>=cfg.min_universe]
    train,valid=separated_validation(grouped,h,cfg)
    if not valid:return  # Insufficient long-horizon evidence must not be fabricated.
    for (_,g),(nextdate,_) in zip(valid,valid[1:]):assert g[f'exit_date_{h}'].max()<nextdate
    assert all(g[f'exit_date_{h}'].max()<valid[0][0] for _,g in train)


def test_old_weights_not_silently_reused():
    r={'version':'19.0.0','weights':{'5':{'weights':{'trend':1},'fit':{'cutoff':'2025-01-01'}}}}
    w,status=weights_for(r,5,'2025-01-01')
    assert status=='OLD_REPORT_NOT_APPLIED' and len(w)>1


def test_reference_dates_not_resampled_around_bad_data(panel,cfg):
    cal=calendar_for(panel,cfg);dates=sample_cutoffs(panel,cal,cfg)
    c=replace(cfg,replay_cutoffs=tuple(str(d.date()) for d in dates))
    broken=panel.copy();broken.loc[broken.date==dates[0],'eligible']=False
    assert sample_cutoffs(broken,cal,c)==dates


def test_immature_reference_rejected_not_shifted(panel,cfg):
    cal=calendar_for(panel,cfg);dates=sample_cutoffs(panel,cal,cfg)
    dates[-1]=cal[-1]
    c=replace(cfg,replay_cutoffs=tuple(str(d.date()) for d in dates))
    assert sample_cutoffs(panel,cal,c)==[]


def test_full_pool_setup_filter_before_head20(panel,cfg):
    p=panel[panel.date==panel.date.max()].copy()
    p['eligible']=True;p['pattern_stage']='NONE'
    for k in TECH:p[k]=80.
    p['reversal']=20.;p['revenue_yoy']=-1.;p['eps_ttm']=np.nan;p['gross_margin']=np.nan
    last=p.index[-1];p.loc[last,'reversal']=90.;p.loc[last,'bias20']=-1.
    r=current_candidates(p,cfg,prior(5,TECH),5,setup='REVERSAL')
    assert p.loc[last,'ticker'] in set(r.ticker)


def test_return_complete_is_not_path_complete(panel,cfg):
    date=sorted(panel.date.unique())[200];x=panel.copy();x.loc[x.date==date,'path_complete_5']=False
    r,_=selected_result(x,date,prior(5,TECH),5,cfg,'test')
    assert r['complete'] and not r['path_complete'] and r['mean_mae'] is None


def test_report_summary_denominators():
    records=[]
    for d,a,b,m in [('2025-01-02',.1,.2,.3),('2025-02-03',.2,.2,None)]:
        for strategy,value in [('adaptive',a),('baseline',b),('benchmark_0050',m)]:
            records.append(dict(date=d,horizon=5,strategy=strategy,mean_net=value,complete=value is not None))
    s,c=summarize_results(records,[5])
    assert s[0]['paired_model_cases']==2 and s[0]['matched_benchmark_cases']==1
    assert s[0]['mean_excess']==pytest.approx(-.2)


def test_bundle_refuses_duplicate_entries(tmp_path):
    path=tmp_path/'bad.zip'
    with pytest.warns(UserWarning):
        with ZipFile(path,'w') as z:
            z.writestr('input_prices.csv.gz',b'a');z.writestr('input_prices.csv.gz',b'b')
    with pytest.raises(ValueError):read_bundle(path)


def test_bundle_refuses_large_inner_gzip(tmp_path):
    path=tmp_path/'bad.zip'
    with ZipFile(path,'w') as z:z.writestr('input_prices.csv.gz',gzip.compress(b'x'*10000))
    with pytest.raises(ValueError):read_bundle(path,max_uncompressed=1000)


def test_full_diagnostic_bundle_roundtrip(prices,cfg,tmp_path):
    report,p,trades=analyze(prices,cfg=cfg)
    save_analysis(tmp_path,report,p,trades,prices,pd.DataFrame())
    restored,evidence,manifest=read_bundle(tmp_path/'v191_diagnostic_bundle.zip')
    assert len(restored)==len(prices) and evidence.empty
    assert report['version']=='19.8' and report['new_independent_holdout_proven'] is False
    assert len(report['case_diagnostics'])==30
    guarded=[r for r in report['results'] if r['strategy']=='guarded']
    assert len(guarded)==30
    assert all(r['mean_net']==0 for r in guarded if r['case_reason']=='PREDECLARED_CASH_POLICY')
    assert {'source','trend','weight_trend','return_basis_5'}.issubset(trades)
    assert (tmp_path/'source_gap_ledger.csv').exists()


@pytest.mark.parametrize('field,value',[('validation_blocks',3),('validation_blocks',21),('weight_shrinkage',0),
    ('weight_shrinkage',2),('improvement_floor',-.1),('min_validation_fill',0),('evaluation_role','unknown')])
def test_new_config_constraints(field,value):
    with pytest.raises(ValueError):Settings(**{field:value})


def test_real_fit_accepts_stable_training_challenger(monkeypatch):
    # Synthetic engineering fixture, NOT a claim that reversal predicts returns.
    dates=pd.bdate_range('2024-01-02',periods=240);parts=[]
    for i in range(16):
        g=pd.DataFrame({'date':dates,'ticker':'0050.TW' if i==0 else f'{2200+i}.TW',
            'eligible':i>0,'bias20':0.,'net_5':.01 if i==0 else .04 if i>=11 else -.02,
            'mae_5':-.03,'return_basis_5':'ADJUSTED_PROVIDER','outcome_state_5':'ASSUMED_OPEN_FILL'})
        for k in TECH:
            g[k]=(100. if k in ('trend','relative_strength','breakout') else 0.) if i<11 else (100. if k=='reversal' else 50. if k in ('volume','pattern') else 0.)
        from v19_config import OPTIONAL
        for k in OPTIONAL:g[k]=np.nan
        g['exit_date_5']=pd.Series(dates).shift(-5)
        parts.append(g)
    p=pd.concat(parts,ignore_index=True)
    def candidates(h,features,cfg,seed):
        return [prior(h,features),{k:float(k=='reversal') for k in features}]
    monkeypatch.setattr('v19_backtest.weight_candidates',candidates)
    w,info,trace=fit_before(p,dates[-1],5,Settings(factor_design='legacy_v192'))
    assert info['status']=='ROBUST_INNER_VALIDATION_ACCEPTED'
    assert info['allocation_allowed'] and w['reversal']>.5
    assert info['validation_common_blocks']>=4
    future=p.copy();future.loc[future.exit_date_5>=dates[-1],'net_5']=-999
    w2,info2,_=fit_before(future,dates[-1],5,Settings(factor_design='legacy_v192'))
    assert w==w2 and info==info2


def test_raw_benchmark_split_cannot_drive_fake_rs_regime(prices,cfg):
    raw=prices.copy();b=raw.ticker==cfg.benchmark;cut=_signal(prices,120)
    raw.loc[b,'adj_close']=np.nan
    for col in ('open','high','low','close'):raw.loc[b&(raw.date>=cut),col]/=4
    p=build_panel(raw,cfg=cfg);q=p[p.date==cut]
    assert set(q.benchmark_kind)=={'CROSS_SECTION_MEDIAN_NOT_INDEX'}
    assert set(q.regime)=={'UNKNOWN'}


def test_adjusted_provider_is_attempted_before_raw(tmp_path):
    calls=[]
    stamp=int(pd.Timestamp('2025-01-02 09:00',tz='Asia/Taipei').timestamp())
    payload={'chart':{'result':[{'meta':{'symbol':'2330.TW'},'timestamp':[stamp],
        'indicators':{'quote':[{'open':[100],'high':[102],'low':[98],'close':[101],'volume':[10000]}],
                      'adjclose':[{'adjclose':[99]}]}}]}}
    c=Client(tmp_path)
    def request(url,params=None,provider='other'):
        calls.append(provider)
        if provider=='Yahoo':return payload
        raise AssertionError('Raw provider should not be needed')
    c.request=request
    df,meta=c.prices('2330.TW','2025-01-02','2025-01-02')
    assert calls==['Yahoo'] and df.adj_close.notna().all()
    assert meta['adjustment_policy']=='v191_prefer_adjusted'


def test_source_failures_keep_stale_cache_not_fake_current(tmp_path,prices):
    old=prices[prices.ticker=='2101.TW'].head(3)
    atomic_csv(tmp_path/'prices'/'2101.TW.csv.gz',old)
    c=Client(tmp_path)
    def unavailable(*args,**kwargs):raise SourceError('Simulated offline provider')
    c.request=unavailable
    out,meta=c.prices('2101.TW','2024-01-01','2025-01-01')
    assert len(out)==3 and meta['state']=='STALE_CACHE' and meta['errors']


def test_summary_excludes_incomparable_price_bases():
    from v191_reporting import summarize_results
    records=[dict(date='2025-01-02',horizon=5,strategy=kind,mean_net=value,
                  complete=True,price_basis=basis)
             for kind,value,basis in [('baseline',.02,'RAW_PRICE_ONLY'),
                                     ('adaptive',.03,'ADJUSTED_PROVIDER'),
                                     ('benchmark_0050',.01,'ADJUSTED_PROVIDER')]]
    summary,cases=summarize_results(records,[5])
    assert summary[0]['paired_model_cases']==0
    assert summary[0]['matched_benchmark_cases']==0
    assert cases[0]['complete'] and not cases[0]['model_price_comparable']


def test_same_basis_stock_models_do_not_mix_benchmark_basis():
    from v191_reporting import summarize_results
    records=[dict(date='2025-01-02',horizon=5,strategy=kind,mean_net=value,
                  complete=True,price_basis=basis)
             for kind,value,basis in [('baseline',.02,'RAW_PRICE_ONLY'),
                                     ('adaptive',.03,'RAW_PRICE_ONLY'),
                                     ('benchmark_0050',.01,'ADJUSTED_PROVIDER')]]
    summary,cases=summarize_results(records,[5])
    assert summary[0]['paired_model_cases']==1
    assert summary[0]['matched_benchmark_cases']==0
    assert summary[0]['mean_excess'] is None


def test_saving_empty_optional_inputs_removes_stale_evidence(tmp_path):
    from v19_backtest import save_analysis
    import zipfile
    old=tmp_path/'input_evidence.csv.gz';old.write_bytes(b'stale')
    report={'status':'INSUFFICIENT_HISTORY','settings':{'horizons':[5]},'cutoffs':[],
            'results':[],'data_kind':'SYNTHETIC_TEST_ONLY'}
    prices=pd.DataFrame(columns=['date','ticker','open','high','low','close','volume','source'])
    save_analysis(tmp_path,report,pd.DataFrame(),pd.DataFrame(),prices,pd.DataFrame())
    assert not old.exists()
    with zipfile.ZipFile(tmp_path/'v191_diagnostic_bundle.zip') as z:
        assert 'input_evidence.csv.gz' not in z.namelist()
