from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from v19_config import Settings, VERSION
from v1961_evidence import monthly_revenue_records, sanitize_revenue_evidence, LEGACY_TIMING, EXPLICIT_TIMING, ASSUMED_TIMING, MISSING_TIMING
from v19_data import Client
from v196_factors import scorecard_weights


def raw_rows(**extra):
    rows=[dict(date='2023-05-01',revenue_year=2023,revenue_month=4,revenue=100),
          dict(date='2024-05-01',revenue_year=2024,revenue_month=4,revenue=120)]
    rows[-1].update(extra)
    return pd.DataFrame(rows)


def legacy_row():
    return pd.DataFrame([dict(ticker='2330.TW',date='2024-05-01',available_at='2024-05-02',source='TaiwanStockMonthRevenue',timing=LEGACY_TIMING,revenue_yoy=20.)])


def test_month_marker_is_not_disclosure():
    r=monthly_revenue_records('2330.TW',raw_rows(),Settings())
    assert len(r)==1 and pd.isna(r.available_at.iloc[0])
    assert r.timing.iloc[0]==MISSING_TIMING


def test_create_time_alone_does_not_prove_disclosure():
    r=monthly_revenue_records('2330.TW',raw_rows(create_time='2024-05-09'),Settings())
    assert pd.isna(r.available_at.iloc[0])


def test_explicit_release_observed_next_day():
    r=monthly_revenue_records('2330.TW',raw_rows(published_at='2024-05-10'),Settings())
    assert pd.Timestamp(r.available_at.iloc[0])==pd.Timestamp('2024-05-11')
    assert r.timing.iloc[0]==EXPLICIT_TIMING


def test_unknown_old_cache_does_not_bypass_strict_mode():
    r=sanitize_revenue_evidence(legacy_row(),Settings())
    assert r.revenue_yoy.isna().all() and r.timing.iloc[0]==MISSING_TIMING


def test_assumed_scenario_explicitly_labeled():
    cfg=replace(Settings(),financial_timing='lagged_research')
    r=monthly_revenue_records('2330.TW',raw_rows(),cfg)
    assert pd.Timestamp(r.available_at.iloc[0])==pd.Timestamp('2024-05-15')
    assert r.timing.iloc[0]==ASSUMED_TIMING


def test_old_cache_can_use_declared_scenario_not_old_date():
    cfg=replace(Settings(),financial_timing='lagged_research')
    r=sanitize_revenue_evidence(legacy_row(),cfg)
    assert pd.Timestamp(r.available_at.iloc[0])==pd.Timestamp('2024-05-15')
    assert r.revenue_yoy.iloc[0]==20.


def test_assumed_cache_is_excluded_when_switch_back_to_strict():
    r=sanitize_revenue_evidence(legacy_row(),replace(Settings(),financial_timing='lagged_research'))
    assert sanitize_revenue_evidence(r,Settings()).revenue_yoy.isna().all()


def test_known_future_release_never_pulled_earlier():
    r=monthly_revenue_records('2330.TW',raw_rows(published_at='2024-06-01'),Settings())
    assert pd.Timestamp(r.available_at.iloc[0])==pd.Timestamp('2024-06-02')


def test_negative_prior_revenue_no_invented_yoy():
    raw=raw_rows();raw.loc[0,'revenue']=-100
    assert monthly_revenue_records('2330.TW',raw,Settings()).empty


def test_flows_unchanged_by_revenue_migration():
    r=legacy_row();r.loc[0,'source']='TaiwanStockInstitutionalInvestorsBuySell';r.loc[0,'revenue_yoy']=np.nan;r['foreign_net']=123
    pd.testing.assert_frame_equal(sanitize_revenue_evidence(r,Settings()),r)


def test_user_explicit_ledger_retains_actual_availability():
    r=legacy_row();r['timing']=EXPLICIT_TIMING;r['available_at']='2024-05-11'
    pd.testing.assert_frame_equal(sanitize_revenue_evidence(r,Settings()),r)


@pytest.mark.parametrize('design',['legacy_v192','expanded_v193','sparse_v194','sparse_v195','tournament_v196'])
def test_financial_downloader_not_bound_to_retired_engine(tmp_path,monkeypatch,design):
    import v194_evidence
    called=[]
    monkeypatch.setattr(v194_evidence,'additional_evidence',lambda client,ticker,start,end:(called.append(ticker) or pd.DataFrame()))
    cfg=replace(Settings(),optional_data=True,financial_history=True,factor_design=design)
    c=Client(tmp_path,cfg);c.finmind=lambda *args,**kwargs:{'status':200,'data':[]}
    c.evidence('2330.TW','2023-01-01','2024-01-01')
    assert called==['2330.TW']


@pytest.mark.parametrize('flag', ['optional_data','financial_history'])
def test_opt_out_respected(tmp_path,monkeypatch,flag):
    import v194_evidence
    called=[]
    monkeypatch.setattr(v194_evidence,'additional_evidence',lambda *a:(called.append(1) or pd.DataFrame()))
    cfg=replace(Settings(),optional_data=True,financial_history=True);cfg=replace(cfg,**{flag:False})
    c=Client(tmp_path,cfg);c.finmind=lambda *args,**kwargs:{'status':200,'data':[]}
    c.evidence('2330.TW','2023-01-01','2024-01-01')
    assert not called


def test_revenue_profile_does_not_require_unavailable_earnings():
    p=scorecard_weights(5,replace(Settings(),optional_data=True))
    assert p['revenue_only']=={'r196_revenue':1.}
    assert set(p['revenue_turnover'])=={'r196_revenue','r196_dollar_turnover'}


def test_no_revenue_policy_if_optional_disabled():
    assert 'revenue_only' not in scorecard_weights(5,Settings())


def test_release_identity():
    assert VERSION=='19.8'


def test_migration_preserves_later_assumed_ingestion_timestamp():
    cfg=replace(Settings(),financial_timing='lagged_research')
    r=monthly_revenue_records('2330.TW',raw_rows(create_time='2024-06-20'),cfg)
    migrated=sanitize_revenue_evidence(r,cfg)
    assert pd.Timestamp(migrated.available_at.iloc[0])==pd.Timestamp('2024-06-21')


def test_legacy_later_availability_is_not_pulled_earlier():
    cfg=replace(Settings(),financial_timing='lagged_research')
    r=legacy_row();r['available_at']='2024-06-20'
    migrated=sanitize_revenue_evidence(r,cfg)
    assert pd.Timestamp(migrated.available_at.iloc[0])==pd.Timestamp('2024-06-20')


def test_revenue_profile_has_human_readable_label():
    from v196_factors import profile_label
    assert profile_label('simple_control_revenue_only') != 'revenue_only'
    assert profile_label('simple_control_revenue_turnover') != 'revenue_turnover'
