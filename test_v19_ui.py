"""Headless UI contract tests. Not a browser/Streamlit Cloud acceptance test."""
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import importlib
import sys
import types
import numpy as np
import pandas as pd
import pytest
from test_v19_core import prices,cfg,panel

class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__('streamlit');self.calls=[];self.session_state={};self.secrets={};self.sidebar=nullcontext();self.view=None
    def cache_data(self,*args,**kwargs):
        def deco(fn):fn.clear=lambda:None;return fn
        return deco(args[0]) if args and callable(args[0]) else deco
    def fragment(self,*args,**kwargs):
        return args[0] if args and callable(args[0]) else lambda f:f
    def __getattr__(self,name):
        def call(*args,**kwargs):
            self.calls.append((name,args,kwargs))
            if name in ('expander','container','spinner'):return nullcontext()
            if name=='columns':return [nullcontext() for _ in range(args[0] if isinstance(args[0],int) else len(args[0]))]
            if name in ('radio','selectbox'):
                values=args[1]
                if kwargs.get('key')=='v19_view' and self.view in values:return self.view
                return values[kwargs.get('index',0)]
            if name in ('button','download_button','link_button'):return False
            if name=='checkbox':return kwargs.get('value',False)
            if name in ('number_input','date_input'):return kwargs.get('value')
            if name=='slider':return args[3] if len(args)>3 else kwargs.get('value')
            return None
        return call

@pytest.fixture
def ui(monkeypatch,tmp_path):
    fake=FakeStreamlit();monkeypatch.setitem(sys.modules,'streamlit',fake)
    sys.modules.pop('v19_ui',None)
    sys.modules.pop('v191_ui',None)
    import v19_ui
    monkeypatch.setattr(v19_ui,'DATA',tmp_path/'v19')
    monkeypatch.setattr(v19_ui,'BASE',tmp_path)
    return v19_ui,fake


def test_decision_page_nonempty(ui,panel,cfg):
    m,f=ui;m.page_decisions(panel,{},cfg)
    cards=[a for name,a,k in f.calls if name=='markdown' and a and 'v19-card' in str(a[0])]
    assert len(cards)==5
    assert any(name=='dataframe' for name,_,_ in f.calls)


def test_decision_page_empty(ui,cfg):
    m,f=ui;m.page_decisions(pd.DataFrame(),{},cfg)
    assert any(name=='info' for name,_,_ in f.calls)


def test_validation_no_report_visible(ui,panel,cfg):
    m,f=ui;m.page_validation(panel,{},cfg)
    assert any(name=='info' for name,_,_ in f.calls)


def test_doctor_page_fields_and_chart(ui,panel,cfg):
    m,f=ui;m.page_doctor(panel,{},cfg)
    assert any(name=='plotly_chart' for name,_,_ in f.calls)
    tables=[args[0] for name,args,_ in f.calls if name=='dataframe']
    assert any(len(t)==6 for t in tables)


def test_doctor_no_prices(ui,cfg):
    m,f=ui;m.page_doctor(pd.DataFrame(),{},cfg)
    assert any(name=='info' for name,_,_ in f.calls)


def test_initial_cloud_main_does_not_load_full_historical_panel(ui, cfg, monkeypatch):
    m, f = ui
    def forbidden(*args, **kwargs):
        raise AssertionError('Eager panel load would risk Cloud OOM')
    monkeypatch.setattr(m, '_display_data', forbidden)
    m.main(configure_page=False)
    assert any(name=='info' for name,_,_ in f.calls)
    assert not any(name=='set_page_config' for name,_,_ in f.calls)


@pytest.mark.parametrize('view_index',[0,1,2])
def test_main_three_destinations(ui,panel,cfg,monkeypatch,view_index):
    m,f=ui
    f.view=['\u9078\u80a1\u8207\u898f\u5283','\u6b77\u53f2\u9a57\u8b49','\u500b\u80a1\u8a3a\u65b7'][view_index]
    f.session_state['ar_cloud_allow_panel']=True
    monkeypatch.setattr(m,'_display_data',lambda *a,**k:(panel,{}))
    monkeypatch.setattr(m,'news_block',lambda:None)
    m.main()
    assert sum(name=='set_page_config' for name,_,_ in f.calls)==1
    assert not any(name=='error' for name,_,_ in f.calls)


@pytest.mark.parametrize('value',[None,np.nan,np.inf,'bad'])
def test_missing_formatter_is_not_zero(ui,value):
    m,_=ui
    assert m.number(value)=='\u2014' and m.percent(value)=='\u2014'


def test_percent_negative_preserved(ui):
    m,_=ui;assert m.percent(-.03)=='-3.00%'


def test_zero_is_not_missing(ui):
    m,_=ui;assert m.number(0)=='0.00' and m.percent(0)=='+0.00%'


def test_html_escape(ui):
    m,_=ui;assert '<script>' not in m.esc('<script>')


def test_quote_stale_and_no_trade_signal(monkeypatch):
    import v19_live
    from datetime import datetime,timedelta
    from v19_config import TZ
    now=datetime(2025,1,2,12,0,tzinfo=TZ);ticker='2330.TW'
    with v19_live._LOCK:
        v19_live._CACHE[ticker]={'day':now.date().isoformat(),'at':10**20,'quote':{
            'ticker':ticker,'price':100,'timestamp':(now-timedelta(minutes=20)).isoformat(),
            'can_buy':False,'can_execute':True,'is_buy_signal':False}}
        v19_live._PENDING.pop(ticker,None)
    q=v19_live.observation(ticker,now=now)
    assert q['price']==100 and not q['can_execute'] and not q['can_buy'] and q['age_seconds']==1200
