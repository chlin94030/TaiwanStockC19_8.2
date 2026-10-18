"""Strict prospective, point-in-time out-of-sample limit-up model comparison.

Input rows are archived as-of completed-day decisions, scored BEFORE future
outcomes.  The evaluator does not download or manufacture prices. It refuses
unverified IPO/limit-price regimes, incomplete outcomes, leakage or missing
proof of forward entry/fees. Never tune weights on the holdout.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {
    'decision_date','ticker','baseline_score','shadow_score','future_limit_touch5',
    'forward_net_return5','forward_max_adverse_pct','entry_tradeable',
    'outcome_end_date','real_market_data','pit_verified','historical_universe_verified',
    'official_limits_verified','raw_ohlcv_verified','corporate_action_verified',
    'future_5_sessions_complete','prospective_decision_logged','fees_tax_slippage_verified','journal_id',
}
PROOF_FLAGS = (
    'real_market_data','pit_verified','historical_universe_verified',
    'official_limits_verified','raw_ohlcv_verified','corporate_action_verified',
    'future_5_sessions_complete','prospective_decision_logged','fees_tax_slippage_verified',
)


def _flag_true(value):
    return value is True or str(value).strip().lower() in ('true','1','yes')


def _top_metrics(frame:pd.DataFrame,rank_key:str, top_n:int) -> tuple[dict,pd.DataFrame]:
    if frame.empty:return {},pd.DataFrame()
    picks=(frame.sort_values(['decision_date',rank_key,'ticker'],ascending=[True,False,True])
           .groupby('decision_date',sort=True).head(top_n).copy())
    # Without a plausible entry, the putative limit-up was not a realizable
    # return. Treat the unfilled allocation as cash (0), never fake fills.
    picks['realizable_net']=np.where(picks.entry_tradeable.eq(1),picks.forward_net_return5,0.0)
    picks['realizable_adverse']=np.where(picks.entry_tradeable.eq(1),picks.forward_max_adverse_pct,0.0)
    picks['tradeable_hit']=(picks.future_limit_touch5.eq(1) & picks.entry_tradeable.eq(1)).astype(int)
    daily=(picks.groupby('decision_date')
           .agg(precision5=('future_limit_touch5','mean'),
                executable_hit_rate=('tradeable_hit','mean'),
                any_limit_hit=('future_limit_touch5','max'),
                mean_net=('realizable_net','mean'),
                mean_adverse=('realizable_adverse','mean'),
                executable_ratio=('entry_tradeable','mean'),
                count=('ticker','size')))
    metrics={
        'dates':int(len(daily)), 'signals':int(len(picks)),
        'limit_touch_precision_at_n':round(float(picks.future_limit_touch5.mean()),4),
        'realizable_limit_touch_precision_at_n':round(float(picks.tradeable_hit.mean()),4),
        'share_of_days_any_limit':round(float(daily.any_limit_hit.mean()),4),
        'avg_5d_net_return':round(float(picks.realizable_net.mean()),5),
        'avg_tradeable_only_net_return':round(float(picks.loc[picks.entry_tradeable.eq(1),'forward_net_return5'].mean()),5) if picks.entry_tradeable.any() else None,
        'p10_5d_net_return':round(float(picks.realizable_net.quantile(.10)),5),
        'avg_max_adverse_pct':round(float(picks.realizable_adverse.mean()),3),
        'can_enter_fraction':round(float(picks.entry_tradeable.mean()),4),
        'untradeable_positive_signals':int(((picks.future_limit_touch5.eq(1)) & (picks.entry_tradeable.eq(0))).sum()),
    }
    return metrics,daily


def validate_live_journal(data:pd.DataFrame | None, *, top_n=5,min_days=120,min_distinct_stocks=300,seed=1820) -> dict:
    if data is None or data.empty:
        return {'status':'NOT_RUN','reason':'尚無足夠事先封存的真實選股決策與官方五日結果；不能用合成資料宣稱實戰勝率'}
    missing=sorted(REQUIRED-set(data.columns))
    if missing:
        return {'status':'NOT_VERIFIED','reason':'缺少必要欄位或來源證據','missing_columns':missing}
    x=data.copy()
    x['decision_date']=pd.to_datetime(x['decision_date'],errors='coerce')
    x['outcome_end_date']=pd.to_datetime(x['outcome_end_date'],errors='coerce')
    numeric=['baseline_score','shadow_score','forward_net_return5','forward_max_adverse_pct']
    for col in numeric:x[col]=pd.to_numeric(x[col],errors='coerce')
    if x[['decision_date','outcome_end_date','ticker','journal_id','future_limit_touch5',*numeric]].isna().any().any():
        return {'status':'NOT_VERIFIED','reason':'關鍵日期、日誌、分數或實際結果含缺值'}
    if not all(x[f].map(_flag_true).all() for f in PROOF_FLAGS):
        failed=[f for f in PROOF_FLAGS if not x[f].map(_flag_true).all()]
        return {'status':'NOT_VERIFIED','reason':'部分樣本未通過PIT／官方真實資料完整核對','flags':failed}
    if not (x['future_limit_touch5'].isin([0,1,True,False])).all():
        return {'status':'NOT_VERIFIED','reason':'漲停標籤必須是經官方每日漲停價核對的0/1'}
    if not (x['entry_tradeable'].map(lambda y: str(y).lower() in ('true','false','1','0')).all()):
        return {'status':'NOT_VERIFIED','reason':'缺少進場可交易性核對（不等於實際成交）'}
    if x.duplicated(['decision_date','ticker']).any():
        return {'status':'NOT_VERIFIED','reason':'同日同檔重複樣本；不可重複計權'}
    if (x['outcome_end_date']<=x['decision_date']).any():
        return {'status':'NOT_VERIFIED','reason':'未來觀察截止日期不晚於決策日期'}
    if (x['journal_id'].astype(str).str.len()<8).any():
        return {'status':'NOT_VERIFIED','reason':'缺少有效事先封存識別碼'}
    if (x[['baseline_score','shadow_score']]<0).any().any() or (x[['baseline_score','shadow_score']]>100).any().any():
        return {'status':'NOT_VERIFIED','reason':'分數不在0~100範圍'}
    if not np.isfinite(x[numeric].to_numpy(dtype=float)).all():
        return {'status':'NOT_VERIFIED','reason':'數值包含無限或NaN'}
    x['future_limit_touch5']=x['future_limit_touch5'].astype(int)
    x['entry_tradeable']=x['entry_tradeable'].map(_flag_true).astype(int)
    x['ticker']=x['ticker'].astype(str)
    dates=sorted(x['decision_date'].unique())
    if len(dates)<min_days or x.ticker.nunique()<min_distinct_stocks:
        return {'status':'INSUFFICIENT','days':len(dates),'stocks':int(x.ticker.nunique()),
                'reason':'兩週資料不足；至少需多期、足量股票之真實決策及五日後驗證'}
    # Frozen 2/3 history / 1/3 test, purge 5 fully observed decision dates.
    split=int(len(dates)*.67)
    test_dates=dates[min(len(dates),split+5):]
    if len(test_dates)<25:
        return {'status':'INSUFFICIENT','reason':'排除五交易日事件重疊後，樣本外測試日期不足25日'}
    test=x[x.decision_date.isin(test_dates)].copy()
    base,base_daily=_top_metrics(test,'baseline_score',top_n)
    shadow,shadow_daily=_top_metrics(test,'shadow_score',top_n)
    common=base_daily.join(shadow_daily,lsuffix='_base',rsuffix='_shadow',how='inner')
    if len(common)<25 or int(common[['count_base','count_shadow']].min().min())<top_n:
        return {'status':'INSUFFICIENT','reason':'OOS日期／日內可排名樣本不足'}
    p_delta=(common['precision5_shadow']-common['precision5_base']).to_numpy()
    r_delta=(common['mean_net_shadow']-common['mean_net_base']).to_numpy()
    # Five-session overlapping labels create serial correlation; do not use
    # naive independent-day bootstrap.  Moving-block resampling preserves
    # clusters of consecutive days (and repeated stock events) conservatively.
    rng=np.random.default_rng(seed)
    n_days=len(common)
    block_length=min(5,n_days)
    n_blocks=int(np.ceil(n_days/block_length))
    starts=rng.integers(0,max(1,n_days-block_length+1),(2500,n_blocks))
    draws=np.concatenate([starts+k for k in range(block_length)],axis=1)[:,:n_days]
    precision_ci=np.quantile(p_delta[draws].mean(axis=1),[.025,.975])
    ret_ci=np.quantile(r_delta[draws].mean(axis=1),[.025,.975])
    # A second report uses non-overlapping five-session anchors.  Promotion
    # requires >=20 distinct such anchors, rather than counting the same
    # stock's single limit event five times as independent successes.
    anchor_idx=np.arange(0,n_days,5)
    anchor_precision_delta=p_delta[anchor_idx]
    anchor_tradeable_delta=(common['executable_hit_rate_shadow']-common['executable_hit_rate_base']).to_numpy()[anchor_idx]
    n_anchors=len(anchor_idx)
    # Trial cannot replace official list unless it improves rare-event hit rate
    # with statistical support and without sacrificing net returns/downside.
    ready=bool(n_anchors>=20 and precision_ci[0]>0 and ret_ci[0]>=-.002 and
               float(np.mean(anchor_tradeable_delta))>0 and
               shadow['p10_5d_net_return']>=base['p10_5d_net_return']-.01 and
               shadow['can_enter_fraction']>=base['can_enter_fraction']-.02)
    return {'status':'OOS_EVALUATED','production_promotion_approved':ready,
            'decision_dates':len(dates),'holdout_dates':len(common),
            'nonoverlapping_5d_anchor_dates':n_anchors,
            'anchor_precision_delta':round(float(anchor_precision_delta.mean()),4),
            'anchor_realizable_limit_hit_delta':round(float(anchor_tradeable_delta.mean()),4),
            'test_start':str(pd.Timestamp(test_dates[0]).date()),
            'test_end':str(pd.Timestamp(test_dates[-1]).date()),
            'sample_event_base_rate':round(float(test.future_limit_touch5.mean()),4),
            'baseline':base,'experimental':shadow,
            'delta_precision_at_n':round(float(p_delta.mean()),4),
            'delta_net_return':round(float(r_delta.mean()),5),
            'precision_delta_95ci':[round(float(v),4) for v in precision_ci],
            'confidence_method':'moving_block_bootstrap_5_sessions',
            'net_delta_95ci':[round(float(v),5) for v in ret_ci],
            'note':'僅評估事先封存的決策與官方驗證上限；不是未來保證，入場可交易性亦非實際成交證明。'}


def main():
    p=argparse.ArgumentParser(description='V18.20 real historical PIT OOS test')
    p.add_argument('--csv',default='')
    p.add_argument('--output',default='data/v1820_oos_status.json')
    a=p.parse_args()
    obs=pd.read_csv(a.csv) if a.csv else None
    report=validate_live_journal(obs)
    output=Path(a.output);output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':report.get('status'),'reason':report.get('reason'),'output':str(output)},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
