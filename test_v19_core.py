"""Offline tests; synthetic fixtures are never real market performance."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
import pytest
from v19_config import Settings, TECH, FACTORS, prior
from v19_data import (clean_prices,parse_finmind,parse_yahoo,parse_twse,safe_symbol,
                      SourceError,Client,BudgetExceeded,atomic_json,read_json,import_legacy_sqlite)
from v19_features import technical_frame,build_panel,w_patterns,score_rows,rank_date,current_candidates,attach_optional,rsi
from v19_backtest import (net_return,add_outcomes,analyze,save_analysis,sample_cutoffs,fit_before,
                          objective,batch_objectives,selected_result)
from v19_service import position_size,field_health,weights_for

@pytest.fixture(scope='module')
def prices():
    rng=np.random.default_rng(33);dates=pd.bdate_range('2024-01-02',periods=420);frames=[]
    for i in range(16):
        ticker='0050.TW' if not i else f'{2100+i}.TW'
        close=100*np.exp(np.cumsum(rng.normal(.0003,.013,len(dates))))
        op=close*np.exp(rng.normal(0,.005,len(close)))
        frames.append(pd.DataFrame({'date':dates,'ticker':ticker,'open':op,'high':np.maximum(close,op)*1.007,
                                  'low':np.minimum(close,op)*.993,'close':close,'volume':2000000.,'adj_close':close,
                                  'source':'SYNTHETIC_TEST_ONLY','name':ticker}))
    return pd.concat(frames,ignore_index=True)

@pytest.fixture(scope='module')
# Legacy regression fixture explicitly exercises the preserved V19.3 design.
def cfg(prices):return Settings(years=3,stocks=15,dates=10,trials=8,seed=1900,as_of=str(prices.date.max().date()),factor_design='expanded_v193')

@pytest.fixture(scope='module')
def panel(prices,cfg):return add_outcomes(build_panel(prices,cfg=cfg),cfg)[0]

@pytest.mark.parametrize('field,value',[('years',0),('years',4),('stocks',2),('dates',1),('dates',101),('trials',1),('trials',101),('horizons',()),('horizons',(0,)),('buy_fee',-.1),('sell_tax',.3),('min_universe',3)])
def test_invalid_config(field,value):
    with pytest.raises(ValueError):Settings(**{field:value})

@pytest.mark.parametrize('h',[5,20,60,120])
def test_prior_sum(h):assert abs(sum(prior(h).values())-1)<1e-12

@pytest.mark.parametrize('sym',['2330.TW','6488.TWO','0050.TW','^TWII'])
def test_safe_symbol(sym):assert safe_symbol(sym.lower())==sym

@pytest.mark.parametrize('sym',['../key','AAPL','2330;rm','abc.TW','2330.TW/a','2330','',None])
def test_bad_symbol(sym):
    with pytest.raises(ValueError):safe_symbol(sym)

def test_price_clean_rejects_bad_ohlc(prices):
    f=prices.head(8).copy();f.loc[0,'high']=1
    assert len(clean_prices(f))==7

def test_price_clean_rejects_nonfinite(prices):
    f=prices.head(8).copy();f.loc[0,'volume']=np.inf
    assert len(clean_prices(f))==7

def test_conflicting_duplicates(prices):
    a=prices.head(1);b=a.copy();b['volume']=3e6
    with pytest.raises(SourceError):clean_prices(pd.concat([a,b]))

def test_identical_duplicates(prices):assert len(clean_prices(pd.concat([prices.head(1)]*2)))==1

def test_finmind_symbol_mismatch():
    with pytest.raises(SourceError):parse_finmind({'data':[{'stock_id':'2317'}]},'2330.TW')

def test_finmind_schema():
    f=parse_finmind({'status':200,'data':[{'date':'2025-01-02','stock_id':'2330','open':100,'max':102,'min':98,'close':101,'Trading_Volume':10000}]},'2330.TW')
    assert f.high.iloc[0]==102 and pd.isna(f.adj_close.iloc[0])

def test_yahoo_symbol_mismatch():
    with pytest.raises(SourceError):parse_yahoo({'chart':{'result':[{'meta':{'symbol':'2317.TW'}}]}},'2330.TW')

def test_yahoo_timestamp_and_adjustment():
    stamp=int(pd.Timestamp('2025-01-02 09:00',tz='Asia/Taipei').timestamp())
    body={'chart':{'result':[{'meta':{'symbol':'2330.TW'},'timestamp':[stamp],
        'indicators':{'quote':[{'open':[100],'high':[102],'low':[98],'close':[101],'volume':[1e5]}],
                      'adjclose':[{'adjclose':[99.]}]}}]}}
    f=parse_yahoo(body,'2330.TW');assert f.date.iloc[0]==pd.Timestamp('2025-01-02');assert f.adj_close.iloc[0]==99

def test_twse_wrong_month():
    with pytest.raises(SourceError):parse_twse({'stat':'OK','date':'20250901','title':'2330'},'2330.TW','20250801')

def test_real_official_twse_sample():
    path=Path(__file__).with_name('v19_twse_sample.json')
    payload=json.loads(path.read_text())
    f=parse_twse(payload,'2330.TW','20250901')
    assert len(f)==21 and f.close.iloc[0]==1165 and f.close.iloc[-1]==1305
    assert f.date.min()==pd.Timestamp('2025-09-01')

def test_atomic_json_nan_safe(tmp_path):
    atomic_json(tmp_path/'x.json',{'x':np.nan,'a':np.int64(2)})
    assert read_json(tmp_path/'x.json')=={'x':None,'a':2}

def test_request_budget(tmp_path):
    c=Client(tmp_path,replace(Settings(),request_budget=0))
    with pytest.raises(BudgetExceeded):c.request('https://example.com')

def test_rate_limit_stops_provider(tmp_path):
    class Response:status_code=429
    c=Client(tmp_path,transport=lambda *a,**k:Response())
    with pytest.raises(SourceError):c.request('https://example.com',provider='FinMind')
    assert 'FinMind' in c.blocked
    before=c.calls
    with pytest.raises(SourceError):c.request('https://example.com',provider='FinMind')
    assert c.calls==before

def test_legacy_readonly(tmp_path):
    import sqlite3
    db=tmp_path/'old.sqlite'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE daily_prices(ticker,date,open,high,low,close,volume)')
        conn.execute('INSERT INTO daily_prices VALUES (?,?,?,?,?,?,?)',('2330.TW','2025-01-02',100,102,99,101,3000))
    before=db.read_bytes();result=import_legacy_sqlite(db,'2025-01-01','2025-01-31')
    assert len(result)==1 and db.read_bytes()==before

def test_feature_prefix_no_future_price_leak(prices):
    g=prices[prices.ticker=='2101.TW'].reset_index(drop=True)
    a=technical_frame(g.head(220));b=technical_frame(g)
    fields=list(TECH[:-1])+['volume','rsi','k','d','macd_hist','bias20']
    fields=[x for x in fields if x!='relative_strength']
    pd.testing.assert_frame_equal(a[fields],b[fields].head(220),check_dtype=False)

def test_future_adjustment_missing_no_leak(prices):
    g=prices[prices.ticker=='2101.TW'].reset_index(drop=True).copy()
    a=technical_frame(g);g.loc[300:,'adj_close']=np.nan;b=technical_frame(g)
    pd.testing.assert_frame_equal(a.head(299)[['reversal','trend','volume','pattern']],b.head(299)[['reversal','trend','volume','pattern']])

def test_rsi_flat_neutral():assert rsi(pd.Series([100.]*60)).iloc[-1]==50

def test_rsi_only_down():assert rsi(pd.Series(np.arange(100,40,-1,dtype=float))).iloc[-1]==0

def test_w_pivots_require_confirmation():
    low=np.ones(65)*100;low[10]=89;low[40]=90;high=low+2;close=low+1
    full=w_patterns(low,high,close)[0];prefix=w_patterns(low[:41],high[:41],close[:41])[0]
    np.testing.assert_allclose(full[:41],prefix,equal_nan=True)

def test_optional_future_not_used(prices):
    raw=technical_frame(prices[prices.ticker=='2101.TW'].head(80))
    date=raw.date.iloc[-1]+pd.Timedelta(days=2)
    ev=pd.DataFrame([{'ticker':'2101.TW','available_at':date,'source':'test','revenue_yoy':100}])
    f=attach_optional(raw,ev);assert f.revenue.isna().all()

def test_optional_no_timestamp_not_used(prices):
    raw=technical_frame(prices[prices.ticker=='2101.TW'].head(80))
    ev=pd.DataFrame([{'ticker':'2101.TW','source':'test','revenue_yoy':100}])
    assert attach_optional(raw,ev).revenue.isna().all()

def test_flow_one_publication_not_three_buys(prices):
    raw=technical_frame(prices[prices.ticker=='2101.TW'].head(80))
    ev=pd.DataFrame([{'ticker':'2101.TW','available_at':raw.date.iloc[60],'source':'test','foreign_net':10000,'trust_net':5000}])
    f=attach_optional(raw,ev)
    assert f.foreign_net.notna().sum()==1 and f.joint.isna().all()

def test_future_prices_not_in_panel(prices,cfg):
    earlier=str(prices.date.sort_values().iloc[len(prices)//2].date())
    out=build_panel(prices,cfg=replace(cfg,as_of=earlier))
    assert out.date.max()<=pd.Timestamp(earlier)

def test_liquidity_filter(prices,cfg):
    raw=prices.copy();raw.loc[raw.ticker=='2101.TW','volume']=1
    p=build_panel(raw,cfg=cfg);assert not p.loc[p.ticker=='2101.TW','eligible'].any()

def test_no_revenue_quality_claim(panel,cfg):
    rows=current_candidates(panel,cfg)
    assert not (rows.setup=='GROWTH_PULLBACK').any()

def test_score_missing_not_zero():
    f=pd.DataFrame({'trend':[np.nan,0,100],'volume':[50,50,50]})
    s,c=score_rows(f,{'trend':.8,'volume':.2})
    assert pd.isna(s.iloc[0]) and s.iloc[1]==10 and s.iloc[2]==90

@pytest.mark.parametrize('entry,exit',[(100,100),(100,110),(100,90),(5,5.5),(1234.5,1432.)])
def test_net_cost_exact(entry,exit):
    cfg=Settings();expected=exit*.9995*(1-.001425-.003)/(entry*1.0005*(1+.001425))-1
    assert float(net_return(entry,exit,cfg))==pytest.approx(expected)
    assert float(net_return(entry,exit,cfg))<exit/entry-1

@pytest.mark.parametrize('entry,exit',[(0,100),(-1,100),(np.nan,100),(100,0),(100,np.inf)])
def test_invalid_return(entry,exit):assert np.isnan(net_return(entry,exit))

def test_entry_next_open_not_signal_close(panel,cfg):
    g=panel[panel.ticker=='2101.TW'].sort_values('date').reset_index(drop=True)
    i=100
    assert g.entry_open.iloc[i]==pytest.approx(g.open.iloc[i+1])
    assert g.net_5.iloc[i]==pytest.approx(float(net_return(g.calc_open.iloc[i+1],g.calc_close.iloc[i+5],cfg)))

def test_no_fill_gap_keeps_cash(prices,cfg):
    raw=prices.copy();idx=raw.index[raw.ticker=='2101.TW'][101];prev=raw.loc[idx-1,'close']
    raw.loc[idx,['open','high','low','close','adj_close']]=[prev*1.09,prev*1.10,prev*1.08,prev*1.09,prev*1.09]
    p,_=add_outcomes(build_panel(raw,cfg=cfg),cfg)
    row=p[(p.ticker=='2101.TW')&(p.date==raw.loc[idx-1,'date'])].iloc[0]
    assert row.net_5==0 and row.outcome_state_5=='ABOVE_PREDECLARED_GAP_CAP'

def test_unmatured_not_zero(panel):
    g=panel[panel.ticker=='2101.TW'].sort_values('date')
    assert g.tail(5).net_5.isna().all()

def test_missing_exit_invalid_not_dropped(prices,cfg):
    raw=prices.copy();target=raw[raw.ticker=='2101.TW'].iloc[105].date
    raw=raw[~((raw.ticker=='2101.TW')&(raw.date==target))]
    p,_=add_outcomes(build_panel(raw,cfg=cfg),cfg)
    row=p[(p.ticker=='2101.TW')&(p.date==prices[prices.ticker=='2101.TW'].iloc[100].date)].iloc[0]
    assert pd.isna(row.net_5) and row.outcome_state_5=='MISSING_FORWARD_BAR'

def test_raw_split_is_not_counted_as_trade_loss(prices,cfg):
    raw=prices.copy();mask=(raw.ticker=='2101.TW')
    raw.loc[mask,'adj_close']=np.nan
    tail=mask&(raw.date>=raw[mask].iloc[103].date)
    for col in ('open','high','low','close'):raw.loc[tail,col]*=.2
    p,_=add_outcomes(build_panel(raw,cfg=cfg),cfg)
    row=p[(p.ticker=='2101.TW')&(p.date==raw[mask].iloc[100].date)].iloc[0]
    assert pd.isna(row.net_5) and row.outcome_state_5=='RAW_CORPORATE_ACTION_DISCONTINUITY'

def test_cutoffs_deterministic_mature(panel,cfg):
    from v19_backtest import calendar_for
    cal=calendar_for(panel,cfg);a=sample_cutoffs(panel,cal,cfg);b=sample_cutoffs(panel,cal,cfg)
    assert a==b and len(a)==10 and len(set(a))==10
    assert cal.get_loc(a[-1])+60<len(cal)

def test_vectorized_matches_reference_objective(panel,cfg):
    dates=sorted(panel[panel.eligible].date.unique())[100:110]
    groups=[panel[(panel.date==d)&panel.eligible] for d in dates]
    w=prior(5,TECH)
    a,sa=objective(groups,w,5,cfg);b,sb=batch_objectives(groups,[w],5,cfg)[0]
    assert a==pytest.approx(b,abs=1e-12) and sa['valid_dates']==sb['valid_dates']

def test_tuning_cannot_read_future_outcomes(panel,cfg):
    date=sorted(panel.date.unique())[300]
    w,info,_=fit_before(panel,date,5,cfg)
    changed=panel.copy();future=changed.exit_date_5>=date
    changed.loc[future,'net_5']=999.;changed.loc[future,'mae_5']=-999.
    w2,info2,_=fit_before(changed,date,5,cfg)
    assert w==w2 and info==info2 and not info['holdout_used_for_selection']
    assert info['status']!='BASELINE_INSUFFICIENT_TRAINING'

def test_future_weight_rejected():
    report={'weights':{'5':{'weights':{'trend':1},'fit':{'cutoff':'2030-01-01'}}}}
    w,state=weights_for(report,5,'2025-01-01');assert state=='STARTING_PRIOR_NO_FIT' and len(w)>1

@pytest.mark.parametrize('capital,entry,stop',[(100000,100,95),(50000,1000,970),(10000,10,9)])
def test_position_risk_budget(capital,entry,stop):
    r=position_size(capital,entry,stop)
    assert r['planned_max_loss']<=capital*.005+.00001
    assert r['position_value']<=capital*.2

@pytest.mark.parametrize('capital,entry,stop',[(0,100,90),(100000,100,101),(100000,0,0),(100000,100,np.nan)])
def test_position_invalid(capital,entry,stop):assert position_size(capital,entry,stop)['shares']==0

def test_field_health_optional_missing(panel):
    health={r['field']:r for r in field_health(panel)}
    assert health['revenue']['present']==0 and health['close']['present']>0
    assert health['real_market_validation']['state']=='NOT_RUN'

def test_complete_ten_date_engine_and_exports(prices,cfg,tmp_path):
    report,p,trades=analyze(prices,cfg=cfg)
    assert len(report['cutoffs'])==10 and report['expected_adaptive_cases']==30
    assert len(trades[trades.strategy.isin(['baseline','adaptive'])])==300
    assert len(trades[trades.strategy=='expanded_fixed_research'])==150
    assert report['global_optimum_proven'] is False
    assert report['status']=='RESEARCH_COMPLETE'
    save_analysis(tmp_path,report,p,trades)
    loaded=read_json(tmp_path/'v19_results.json')
    assert loaded['settings']['seed']==1900 and len(loaded['cutoffs'])==10
    assert 'SYNTHETIC' not in loaded['status'] # status isn't a source attestation
    assert set(p.source)=={'SYNTHETIC_TEST_ONLY'}
    assert (tmp_path/'comparison.csv').exists() and (tmp_path/'selected_trades.csv').exists()

def test_zero_volume_exit_cannot_be_sold(prices,cfg):
    raw=prices.copy();idx=raw.index[raw.ticker=='2101.TW'][105]
    raw.loc[idx,'volume']=0
    p,_=add_outcomes(build_panel(raw,cfg=cfg),cfg)
    signal=raw[raw.ticker=='2101.TW'].iloc[100].date
    row=p[(p.ticker=='2101.TW')&(p.date==signal)].iloc[0]
    assert pd.isna(row.net_5) and row.outcome_state_5=='EXIT_NO_VOLUME'

def test_calendar_retains_missing_benchmark_day(prices,cfg):
    from v19_backtest import calendar_for
    target=prices[prices.ticker==cfg.benchmark].iloc[120].date
    raw=prices[~((prices.ticker==cfg.benchmark)&(prices.date==target))]
    p=build_panel(raw,cfg=cfg);cal=calendar_for(p,cfg)
    assert target in cal


def test_physical_volume_separate_from_factor(prices):
    raw=prices[prices.ticker=='2101.TW'].reset_index(drop=True)
    f=technical_frame(raw)
    np.testing.assert_allclose(f.share_volume,raw.volume)
    assert f.volume.dropna().between(0,100).all()

def test_zero_volume_entry_remains_cash(prices,cfg):
    raw=prices.copy();idx=raw.index[raw.ticker=='2101.TW'][101]
    raw.loc[idx,'volume']=0
    p,_=add_outcomes(build_panel(raw,cfg=cfg),cfg)
    signal=raw[raw.ticker=='2101.TW'].iloc[100].date
    row=p[(p.ticker=='2101.TW')&(p.date==signal)].iloc[0]
    assert row.net_5==0 and row.fill_state=='NO_VOLUME'


def test_benchmark_has_separate_etf_tax(prices,cfg):
    p,_=add_outcomes(build_panel(prices,cfg=cfg),cfg)
    r=p[(p.ticker==cfg.benchmark)].iloc[100]
    entry=p[(p.ticker==cfg.benchmark)].iloc[101].calc_open
    end=p[(p.ticker==cfg.benchmark)].iloc[105].calc_close
    expected=float(net_return(entry,end,replace(cfg,sell_tax=cfg.benchmark_sell_tax)))
    assert abs(r.net_5-expected)<1e-12

def test_synthetic_health_never_says_real_market_validated(panel):
    report={'data_kind':'SYNTHETIC_TEST_ONLY','cutoffs':['2025-01-02'],'status':'RESEARCH_COMPLETE'}
    health=field_health(panel,report)
    item=[x for x in health if x['field']=='real_market_validation'][0]
    assert item['present']==0 and item['state']=='SYNTHETIC_TEST_ONLY'
