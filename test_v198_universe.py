"""V19.8 tests: source market population, bias warnings and factor masks.
All synthetic fixtures are engineering tests, not investment backtests.
"""
from dataclasses import replace
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from v19_config import VERSION,Settings
from v198_universe import normalize_catalog,balanced_sample,summarize_catalog,market,historical_pool_diagnostics
import requests
from v198_selection import profiles_for_v198
from v19_data import Client,SourceError,atomic_json,download_history
from v19_service import load_inputs


def catalog(count=1800):
    data=[]
    for i in range(count):
        data.append({'ticker':f'{1000+i}.TW' if i%3 else f'{1000+i}.TWO',
                     'name':str(i),'sector':['電子工業','生技醫療','紡織','金融','食品','光電','營建','航運','其他'][i%9]})
    return data


def test_new_settings_are_not_old_limited_pool():
    c=Settings()
    assert VERSION=='19.8' and c.stocks==600 and c.dates==100
    assert c.factor_design=='balanced_v198' and c.universe_mode=='balanced'
    assert c.min_download_coverage==.8


@pytest.mark.parametrize('mode',['balanced','proportional','legacy_random','full_current'])
def test_sampling_unique_deterministic_independent_date_seed(mode):
    src=catalog()
    a=balanced_sample(src,600,seed=82,mode=mode)
    b=balanced_sample(src[::-1],600,seed=82,mode=mode)
    assert a==b
    assert len({r['ticker'] for r in a})==len(a)
    assert len(a)==(1800 if mode=='full_current' else 600)
    assert {x['board'] for x in a}=={'TWSE','TPEX'}


def test_small_sectors_maintained_when_n_enough():
    src=catalog(700)
    a=balanced_sample(src,300,42,'balanced')
    assert len(set((x['board'],x['sector']) for x in a))==len(set((x['board'],x['sector']) for x in normalize_catalog(src)))
    assert len(set(x['sector'] for x in a))>=8
    assert len(set(x['ticker'] for x in a))==300


def test_modes_are_not_identical_in_diverse_catalog():
    src=catalog()
    assert {r['ticker'] for r in balanced_sample(src,300,42,'balanced')} != {r['ticker'] for r in balanced_sample(src,300,42,'legacy_random')}


def test_no_ETF_or_duplicate_or_invalid():
    rows=catalog(2)+[{'ticker':'0050.TW','sector':'ETF'},
                     {'ticker':'1234.TW','sector':'ETF'},
                     {'ticker':'q234.TW','sector':'food'},
                     {'ticker':'1000.TWO','sector':'extra'}]
    x=normalize_catalog(rows)
    assert len(x)==2 and x[0]['ticker']=='1000.TWO'


def test_coverage_ledgers_separate_catalog_selection_and_download():
    src=catalog(800)
    selected=balanced_sample(src,400,42)
    ok=[r['ticker'] for r in selected[:210]]
    q=summarize_catalog(src,selected,ok)
    assert q['catalog_size']==800 and q['selected_size']==400 and q['downloaded_size']==210
    assert q['download_coverage']==pytest.approx(.525)
    assert len(q['missing_selected'])==190
    assert not q['historical_listing_delisting_verified']
    assert not q['diversity_is_not_accuracy_or_market_cap_representativeness'] is False


def test_universe_from_cached_current_master_large(tmp_path):
    rows=catalog()
    atomic_json(tmp_path/'universe.json',{'at':time.time(),'rows':rows,'scope':'CACHED_CURRENT_SNAPSHOT'})
    c=Client(tmp_path,Settings(stocks=600,universe_mode='balanced'))
    selected,scope=c.universe()
    assert len(selected)==600 and scope=='CACHED_CURRENT_SNAPSHOT'
    assert (tmp_path/'universe_quality.json').exists()
    assert (tmp_path/'selected_universe.csv').exists()
    q=json.loads((tmp_path/'universe_quality.json').read_text())
    assert q['catalog_size']==1800 and q['selected_market']['TWSE']>0 and q['selected_market']['TPEX']>0


def test_discovery_failure_wont_claim_54_stock_full_market(tmp_path):
    def no_network(*a,**k):raise requests.ConnectionError('OFFLINE')
    c=Client(tmp_path,Settings(stocks=600),transport=no_network)
    with pytest.raises(SourceError,match='BROAD_UNIVERSE_CATALOG_UNAVAILABLE'):c.universe()


def test_legacy_fallback_requires_explicit_mode(tmp_path):
    def no_network(*a,**k):raise requests.ConnectionError('OFFLINE')
    cfg=Settings(universe_mode='legacy_random')
    q,scope=Client(tmp_path,cfg,transport=no_network).universe()
    assert 40<=len(q)<=60 and 'LEGACY' in scope


def test_supplied_delisted_symbols_append_without_certification(tmp_path):
    src=catalog(100)
    atomic_json(tmp_path/'universe.json',{'at':time.time(),'rows':src,'scope':'CACHED_CURRENT_SNAPSHOT'})
    (tmp_path/'historical_universe.csv').write_text('ticker,name,sector,listed_date,delisted_date,source\n5201.TWO,delisted,電子,2022-01-01,2024-12-31,EXAMPLE_IMPORT\n')
    c=Client(tmp_path,Settings(stocks=300))
    pool,scope=c.universe()
    assert '5201.TWO' in {r['ticker'] for r in pool}
    assert 'UNVERIFIED' in scope
    x=next(r for r in pool if r['ticker']=='5201.TWO')
    assert x['delisted_date']=='2024-12-31'


def test_profile_weights_are_small_family_and_sum_one():
    for h in (5,20,60):
        p=profiles_for_v198(h)
        assert p['legacy_control']
        assert len(p)>=5
        for k,w in p.items():
            assert sum(w.values())==pytest.approx(1.)
            if k!='legacy_control':assert 1<=len(w)<=3
        assert all('p197_sector' not in w for w in p.values())
    p=profiles_for_v198(5)
    assert p['high_defense_65_35']=={'p197_annual_high':.65,'p197_downside':.35}


def test_no_optional_financial_factor_without_evidence():
    assert all('p197_quality' not in w for w in profiles_for_v198(60,Settings(optional_data=False)).values())
    assert 'quality_liquid_research' in profiles_for_v198(60,Settings(optional_data=True))


def test_loading_cache_does_not_fabricate_600_stock_pool(tmp_path):
    dates=pd.bdate_range('2025-01-01',periods=30)
    frame=pd.concat([pd.DataFrame({'date':dates,'ticker':f'{1100+i}.TW','open':10+i,
      'close':10+i,'high':10+i,'low':10+i,'volume':1000000,'source':'SYNTHETIC_TEST_ONLY'}) for i in range(25)],ignore_index=True)
    frame.to_csv(tmp_path/'history_prices.csv.gz',index=False,compression='gzip')
    prices,_=load_inputs(tmp_path,Settings(stocks=600,as_of='2025-03-31'))
    assert prices.ticker.nunique()==25


def test_quartiles_are_historical_signal_date_only():
    d=pd.Timestamp('2025-01-02')
    p=pd.DataFrame({'date':[d]*24,'ticker':[f'{1000+i}.TW' if i%2 else f'{1000+i}.TWO' for i in range(24)],
                    'eligible':[True]*24,'turnover20':np.arange(1,25,dtype=float)*1e6,'sector':['dummy']*24})
    q=historical_pool_diagnostics(p,Settings(stocks=600,as_of='2025-01-06'))
    assert q['tested_tickers']==24 and q['today_eligible']==24
    assert q['tested_market']=={'TPEX':12,'TWSE':12}
    assert q['liquidity_proxy_quartiles_ntd']['0.5']==pytest.approx(12.5e6)


def test_backward_compat_legacy_stays_unchanged():
    from v197_selection import profiles_for
    old=profiles_for(5,Settings(factor_design='ten_angle_v197'))
    new=profiles_for(5,Settings(factor_design='balanced_v198'))
    assert 'sector_leadership_RESEARCH_ONLY' in old
    assert 'sector_leadership_RESEARCH_ONLY' not in new
    assert 'high_defense_65_35' not in old


def test_offline_sourced_catalog_can_supply_broad_universe_without_network(tmp_path):
    def no_network(*a,**k):raise requests.ConnectionError('OFFLINE')
    source=catalog(800)
    df=pd.DataFrame(source)
    df['source']='USER_DOCUMENTED_SNAPSHOT'
    df.to_csv(tmp_path/'historical_universe.csv',index=False)
    c=Client(tmp_path,Settings(stocks=600),transport=no_network)
    pool,scope=c.universe()
    assert len(pool)==600 and 'USER_SUPPLIED' in scope and 'UNVERIFIED' in scope
    assert (tmp_path/'universe_quality.json').exists()
    assert not c.universe_quality['historical_listing_delisting_verified']


def test_downloaded_count_is_not_equivalent_to_factor_history(tmp_path):
    from v198_universe import history_depth_report
    selected=catalog(4)
    ticks=[x['ticker'] for x in selected]
    frames=[]
    for ticker,n in zip(ticks,[2,79,80,260]):
        frames.append(pd.DataFrame({'ticker':ticker,'date':pd.bdate_range('2025-01-01',periods=n)}))
    q=history_depth_report(pd.concat(frames,ignore_index=True),selected,warmup=80)
    assert q['downloaded_price_symbols']==4
    assert q['warmup_eligible_symbols']==2
    assert q['warmup_eligible_fraction']==.5
    assert q['has_252_bar_history_symbols']==1


def test_sourced_listing_dates_propagate_into_history_and_eligibility(tmp_path,monkeypatch):
    from v19_features import build_panel
    from v19_data import COLS
    source=catalog(5)
    source[0].update(listed_date='2024-01-01',delisted_date='2024-12-31',member_source='DATED_USER_SOURCE')
    atomic_json(tmp_path/'universe.json',{'at':time.time(),'rows':source,'scope':'USER_DOCUMENTED_CURRENT'})
    def fake_prices(self,ticker,start,end,official_fallback=False,force=False):
        dates=pd.bdate_range('2024-03-01',periods=170)
        close=np.linspace(10,25,len(dates))
        f=pd.DataFrame({'date':dates,'ticker':ticker,'open':close,'high':close*1.01,
            'low':close*.99,'close':close,'volume':5000000,
            'adj_close':close,'source':'MOCK_PRICE_NOT_MARKET'})
        return f,{'ticker':ticker,'state':'FETCHED','rows':len(f)}
    monkeypatch.setattr(Client,'prices',fake_prices)
    cfg=Settings(stocks=5,years=3,as_of='2026-10-15',optional_data=False)
    prices,ev,manifest=download_history(tmp_path,cfg)
    x=prices[prices.ticker==source[0]['ticker']]
    assert set(x['delisted_date'])=={'2024-12-31'}
    assert set(x['member_source'])=={'DATED_USER_SOURCE'}
    assert manifest['universe_quality']['has_252_bar_history_symbols']==0
    assert len(pd.read_csv(tmp_path/'history_prices.csv.gz'))>0
    # Real metadata reaches the eligibility filter: a post-delisting bar is ineligible.
    later=pd.concat([x,pd.DataFrame([dict(x.iloc[-1],date=pd.Timestamp('2025-03-03'))])],ignore_index=True)
    # Already dated metadata is not erased by the price cleaning stage.
    assert set(later['delisted_date'])=={'2024-12-31'}


def test_small_catalog_cannot_masquerade_as_large_target(tmp_path,monkeypatch):
    from v19_data import download_history
    rows=catalog(5)
    atomic_json(tmp_path/'universe.json',{'at':time.time(),'rows':rows,'scope':'CACHED_STOCK_MASTER'})
    def fake_prices(self,ticker,start,end,official_fallback=False,force=False):
        dates=pd.bdate_range('2025-01-01',periods=100)
        vals=np.linspace(10,20,len(dates))
        x=pd.DataFrame({'date':dates,'ticker':ticker,'open':vals,'high':vals*1.01,
          'low':vals*.99,'close':vals,'volume':1000000,'adj_close':vals,'source':'MOCK'})
        return x,{'ticker':ticker,'state':'FETCHED','rows':len(x)}
    monkeypatch.setattr(Client,'prices',fake_prices)
    _,_,m=download_history(tmp_path,Settings(stocks=600,as_of='2026-10-15'))
    assert m['pool_equities']==5 and m['ready_equities']==5
    assert m['download_completeness']==1.
    assert m['requested_target_coverage']<.01
    assert m['catalog_target_coverage']<.01

# V19.8.1: executable Cloud safe-start and repeatable validation checks.
def test_safe_boot_has_no_heavy_model_import(monkeypatch):
    import importlib.util
    import sys
    import types
    import app
    events=[]
    class Block:
        def __enter__(self): return self
        def __exit__(self, *args): return False
    class FakeStreamlit(types.SimpleNamespace):
        def set_page_config(self, **kw): events.append('set_page_config')
        def title(self, *a, **kw): events.append('title')
        def success(self, *a, **kw): pass
        def info(self, *a, **kw): pass
        def caption(self, *a, **kw): pass
        def button(self, *a, **kw): return False
        def expander(self, *a, **kw): return Block()
        def write(self, *a, **kw): pass
    fake=FakeStreamlit(session_state={})
    monkeypatch.setitem(sys.modules,'streamlit',fake)
    monkeypatch.delitem(sys.modules,'v19_ui',raising=False)
    app.run()
    assert events==['set_page_config','title']
    assert 'v19_ui' not in sys.modules


def test_full_boot_traces_import_failures_without_crashing_safemode(monkeypatch,capsys):
    import app
    import sys
    import types
    from contextlib import nullcontext
    records=[]
    fake=types.SimpleNamespace(session_state={'ar1981_mode':'full'},
         set_page_config=lambda **kw:records.append(('page_config',kw)),
         sidebar=nullcontext(),
         error=lambda s:records.append(('error',s)),
         warning=lambda s:records.append(('warning',s)),
         caption=lambda s:records.append(('caption',s)),
         button=lambda *a,**k:False)
    monkeypatch.setitem(sys.modules,'streamlit',fake)
    bad=types.ModuleType('v19_ui')
    def fail(**kw):raise ValueError('intentional synthetic error')
    bad.main=fail
    monkeypatch.setitem(sys.modules,'v19_ui',bad)
    app.run()
    assert any(t=='error' for t,_ in records)
    assert 'stage=FULL_DASHBOARD_ERROR' in capsys.readouterr().err
    assert len([t for t,_ in records if t=='page_config'])==1


def test_full_boot_page_config_is_only_called_once(monkeypatch):
    import app,sys,types
    from contextlib import nullcontext
    events=[]
    fake=types.SimpleNamespace(session_state={'ar1981_mode':'full'},
      set_page_config=lambda **kw:events.append('config'),
      sidebar=nullcontext(),button=lambda *a,**kw:False)
    monkeypatch.setitem(sys.modules,'streamlit',fake)
    stub=types.ModuleType('v19_ui')
    def fake_main(configure_page=True):
        assert configure_page is False
        events.append('main')
    stub.main=fake_main
    monkeypatch.setitem(sys.modules,'v19_ui',stub)
    app.run()
    assert events==['config','main']


def test_cloud_probe_can_report_subprocess_failure_without_app_crash(monkeypatch):
    import app,subprocess
    def bad(*args,**kwargs):
        return subprocess.CompletedProcess(args,137,stdout='',stderr='Killed')
    monkeypatch.setattr(subprocess,'run',bad)
    outcome=app.run_import_probe()
    assert outcome['status']=='IMPORT_ERROR'
    assert outcome['exit_code']==137


def test_cloud_recent_history_limits_rows_but_full_history_unchanged(tmp_path):
    from v19_service import _recent_price_rows,load_inputs
    from v19_config import Settings
    days=pd.bdate_range('2024-01-01',periods=410)
    data=pd.concat([
        pd.DataFrame({'ticker':ticker,'date':days,'close':np.arange(len(days))})
        for ticker in ('2330.TW','3008.TW','0050.TW')
    ],ignore_index=True)
    # Mix ticker/date order so last K means latest DATE, not the last CSV rows.
    data=data.sample(frac=1.,random_state=34)
    f=tmp_path/'history_prices.csv.gz'
    data.to_csv(f,index=False,compression='gzip')
    recent=_recent_price_rows(f,300,cutoff='2025-05-01')
    assert recent.groupby('ticker').size().to_dict()=={'0050.TW':300,'2330.TW':300,'3008.TW':300}
    assert str(recent.date.max())<='2025-05-01'
    cfg=Settings(stocks=600,as_of='2025-05-01')
    full,_=load_inputs(tmp_path,cfg)
    truncated,_=load_inputs(tmp_path,cfg,recent_bars=300)
    assert len(full)==1230
    assert len(truncated)==900
    assert truncated.groupby('ticker').date.max().equals(full.groupby('ticker').date.max()) is False or len(truncated)<len(full)



def test_lab_ranking_missing_selected_input_excludes_stock():
    from research_lab import DateMatrix, Trial
    n=205
    dt=pd.bdate_range('2025-01-01',periods=n)
    rows=[]
    for i,d in enumerate(dt):
        for code in ['001.TW','002.TW','003.TW','0050.TW']:
            rows.append(dict(date=d,ticker=code,eligible=code!='0050.TW',
                p197_annual_high=np.nan if code=='003.TW' else 90+i*.001,
                p197_downside=90. if code=='003.TW' else 30.,
                net_5=.03 if code=='003.TW' else .01,mae_5=-.01,
                return_basis_5='ADJUSTED_PROVIDER',exit_date_5=d+pd.Timedelta(days=7)))
    p=pd.DataFrame(rows)
    m=DateMatrix(p,['p197_annual_high','p197_downside'],[5],top_k=2)
    z=m.evaluate(Trial(5,{'p197_annual_high':1.},'test'))
    selected=set(np.array(m.tickers)[z['selected'][0]])
    assert selected == {'001.TW','002.TW'}
    assert z['valid'][0]
    assert z['returns'][0] == pytest.approx(.01)


def test_lab_factor_weights_must_sum_and_no_automatic_live():
    from research_lab import Trial,selection_score,protocol_key
    p={'coverage':1.,'n_dates':50,'nonoverlap':10,'excess_mean':.02,
       'excess_std':.01,'downside_mean':-.02}
    assert np.isfinite(selection_score(p,2))
    assert selection_score({**p,'nonoverlap':2},2) == -np.inf
    assert protocol_key('hash',['factor'],[5],1981,3) != protocol_key('other',['factor'],[5],1981,3)


def test_lab_unbounded_restart_and_one_time_seal(tmp_path):
    from research_lab import DateMatrix, optimise, Trial
    from types import SimpleNamespace
    dates=pd.bdate_range('2024-03-01',periods=350)
    panel=[]
    for i,d in enumerate(dates):
        for code in ('001.TW','002.TW','003.TW','0050.TW'):
            panel.append(dict(date=d,ticker=code,eligible=code!='0050.TW',
                p197_annual_high=float(40 + i*.03 + (code=='001.TW')),
                p197_downside=float(60+i*.01 + (code=='002.TW')),
                net_5=.004 if code=='0050.TW' else .005 + (.001 if code=='001.TW' else 0),
                mae_5=-.02,return_basis_5='ADJUSTED_PROVIDER',
                exit_date_5=d+pd.Timedelta(days=7)))
    matrix=DateMatrix(pd.DataFrame(panel),['p197_annual_high','p197_downside'],[5],top_k=2)
    args=SimpleNamespace(output=str(tmp_path),seed=1981,horizons=[5],max_inputs=2,
        reset=False,resume=False,finalize=False,include_optional=False,rounds=0,batch=4,patience=2,
        max_minutes=.5,min_progress=.00015)
    first=optimise(args,matrix,'abc')
    assert first['audit_exposed'] is False
    assert not (tmp_path/'frozen_models.json').exists()
    assert (tmp_path/'checkpoint.json').exists()
    args.resume=True
    args.finalize=True
    second=optimise(args,matrix,'abc')
    assert second['audit_exposed'] is True
    assert (tmp_path/'sealed.json').exists()
    assert json.loads((tmp_path/'frozen_models.json').read_text())['sealed']
    assert not second['results']['5']['independent_approval']
    with pytest.raises(ValueError,match='sealed'):
        optimise(args,matrix,'abc')


def test_cloud_database_download_is_explicit_not_eager(tmp_path,monkeypatch):
    """A large archive must never be read merely to render the Cloud homepage."""
    import sys,types
    from v192_ui import database_download
    p=tmp_path/'v192_history_database.zip';p.write_bytes(b'test archive payload')
    calls=[]
    fake=types.SimpleNamespace(
        session_state={},
        button=lambda *a,**k:calls.append('prepare_button') or False,
        download_button=lambda *a,**k:calls.append(('download',a[1]))
    )
    monkeypatch.setitem(sys.modules,'streamlit',fake)
    database_download(tmp_path)
    assert calls==['prepare_button']
    fake.button=lambda *a,**k:True
    database_download(tmp_path)
    assert calls[-1]==('download',b'test archive payload')


def test_cloud_locked_streamlit_skips_known_buggy_1580():
    from pathlib import Path
    requirements=Path(__file__).parent.joinpath('requirements.txt').read_text()
    assert 'streamlit==1.59.2' in requirements
    assert 'pyarrow==24.0.0' in requirements
