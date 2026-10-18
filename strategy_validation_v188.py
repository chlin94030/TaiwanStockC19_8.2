"""Auditable V18.8 selection weights and *fail-closed* OOS evaluation.

This module never changes production weights. A true OOS result requires a
separate, recorded point-in-time observation table with historical universe,
actual announcement versions, completed forward outcomes and execution costs.
Synthetic and present-day backfilled samples are not valid performance proof.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from strategy_config import SELECTION_WEIGHTS, V18_DIMENSION_WEIGHTS, V18_OVERLAY_BLEND
from v18_confidence_engine import weights_for

from strategy_config import HOLDING_SESSIONS
DAYS = dict(HOLDING_SESSIONS)
WEIGHT_VERSION = 'v18.11-one-week-provisional / mid-long-frozen'
REQUIRED = ('decision_date','ticker','horizon','base_score','candidate_score',
            'net_forward_return','market_forward_return','pit_verified',
            'historical_universe_verified','corporate_action_verified',
            'execution_verified','real_market_data','evidence_version_id',
            'universe_observation_id')


def weight_audit() -> dict:
    weights = {}
    checked = True
    for h in DAYS:
        core = dict(SELECTION_WEIGHTS[h]); dimensions = dict(V18_DIMENSION_WEIGHTS[h]); factors = weights_for(h)
        sums = {'baseline': round(sum(core.values()), 10),
                'dimension': round(sum(dimensions.values()),10),
                'thirteen_factor':round(sum(factors.values()),10)}
        ok = all(abs(v - 1) < 1e-9 for v in sums.values()) and all(
            np.isfinite(v) and v >= 0 for v in [*core.values(), *dimensions.values(), *factors.values()])
        checked = checked and ok
        weights[h] = {'core':core, 'dimension':dimensions, 'thirteen_factor':factors,
                      'maximum_new_factor_blend':V18_OVERLAY_BLEND[h],
                      'weight_sums':sums,'arithmetic_ok':ok}
    return {
        'weight_version':WEIGHT_VERSION,
        'weight_arithmetic_verified':checked,
        'rank_core_frozen':False,
        'short_core_recalibrated_provisional':True,
        'profitability_oos_verified':False,
        'weight_calibration_status':'UNVERIFIED_REAL_MARKET_OOS',
        'explanation':'權重可重現、權重和與非負性已核對；缺乏完整歷史 PIT、下市股票與已實現報酬證據，不能稱已實證最優。',
        'horizons':weights,
    }


def _bool_series(s:pd.Series) -> pd.Series:
    return s.map(lambda x: x is True or str(x).strip().lower() in ('true','1','yes'))


def _metrics(obs:pd.DataFrame, score_col:str, top_n:int) -> dict:
    frames=[]; daily=[]
    for day, day_df in obs.groupby('decision_date',sort=True):
        d=day_df.sort_values([score_col,'ticker'],ascending=[False,True]).head(top_n)
        if d.empty: continue
        frames.append(d)
        daily.append({'date':str(day), 'mean_alpha':float((d.net_forward_return-d.market_forward_return).mean()),
                      'median_return':float(d.net_forward_return.median()),
                      'positive_rate':float((d.net_forward_return>0).mean())})
    if not frames:
        return {'available':False,'n_dates':0}
    chosen=pd.concat(frames)
    alpha=chosen.net_forward_return-chosen.market_forward_return
    return {'available':True,'n_dates':len(daily),'sample_n':len(chosen),
            'mean_net_return':round(float(chosen.net_forward_return.mean()),5),
            'mean_excess':round(float(alpha.mean()),5),
            'positive_rate':round(float((chosen.net_forward_return>0).mean()),4),
            'p10_net_return':round(float(chosen.net_forward_return.quantile(.10)),5),
            'date_metrics':daily}


def evaluate_oos(observations:pd.DataFrame | None,*, min_dates=24,min_stocks=100,top_n=5) -> dict:
    """Check real-source evidence and evaluate frozen weights on untouched dates.

    No hyperparameter tuning; no synthetic fallback; purges horizon look-forward
    from calibration. Actual market holidays should be provided by the data
    producer; a 1.5x calendar-day embargo is conservative for a quick QA gate.
    """
    if observations is None or observations.empty:
        return {'status':'NOT_RUN','reason':'沒有可回放的真實 PIT 歷史決策與後續成交資料'}
    missing=[key for key in REQUIRED if key not in observations.columns]
    if missing:
        return {'status':'NOT_VERIFIED','reason':'缺少驗效欄位','missing_columns':missing}
    o=observations.copy()
    o['decision_date']=pd.to_datetime(o.decision_date,errors='coerce')
    o=o.dropna(subset=['decision_date','ticker','horizon','base_score','candidate_score',
                       'net_forward_return','market_forward_return'])
    o=o[o.horizon.isin(DAYS)]
    for key in ('pit_verified','historical_universe_verified','corporate_action_verified','execution_verified','real_market_data'):
        if not _bool_series(o[key]).all():
            return {'status':'NOT_VERIFIED', 'reason':f'{key} 含有未經證明／非真實資料',
                    'records_checked':len(o)}
    if (o['evidence_version_id'].fillna('').astype(str).str.len()<5).any() or (o['universe_observation_id'].fillna('').astype(str).str.len()<5).any():
        return {'status':'NOT_VERIFIED','reason':'缺少不可回填的歷史公告版本／上市母體封存識別'}
    if (o[['base_score','candidate_score']] < 0).any().any() or (o[['base_score','candidate_score']]>100).any().any():
        return {'status':'NOT_VERIFIED','reason':'選股分數超出可比較尺度'}
    if o.duplicated(['decision_date','ticker','horizon']).any():
        return {'status':'NOT_VERIFIED','reason':'日期、股票及週期不唯一（重複樣本）'}
    groups={}
    for h, hd in o.groupby('horizon'):
        days=sorted(hd.decision_date.unique())
        if len(days)<min_dates or hd.ticker.nunique()<min_stocks:
            groups[h]={'status':'INSUFFICIENT_OOS','dates':len(days),'stocks':int(hd.ticker.nunique())}
            continue
        split=int(len(days)*.67)
        last_train=pd.Timestamp(days[split-1]); test_after=last_train+pd.Timedelta(days=int(DAYS[h]*1.6)+3)
        test=hd[hd.decision_date>=test_after]
        test_days=int(test.decision_date.nunique())
        if test_days<8:
            groups[h]={'status':'INSUFFICIENT_OOS','dates':len(days),'test_dates':test_days,
                       'reason':'Purged OOS 測試日期少於 8；不可回頭縮小禁運期硬湊驗效'}
            continue
        base=_metrics(test,'base_score',top_n)
        candidate=_metrics(test,'candidate_score',top_n)
        paired={x['date']:x['mean_alpha'] for x in base['date_metrics']}
        delta=[x['mean_alpha']-paired[x['date']] for x in candidate['date_metrics'] if x['date'] in paired]
        # Conservative per-date bootstrap, not individual-stock IID assumptions.
        rng=np.random.default_rng(188)
        boots=[np.mean(rng.choice(delta,size=len(delta),replace=True)) for _ in range(800)] if delta else []
        ci=np.quantile(boots,[.025,.975]).tolist() if boots else [None,None]
        candidate_beats=(ci[0] is not None and ci[0] > 0 and
                         candidate['p10_net_return'] >= base['p10_net_return'] - .01)
        groups[h]={'status':'OOS_EVALUATED','train_last_date':str(last_train.date()),
                   'test_start_date':str(test_after.date()),'test_dates':test_days,
                   'base':base,'candidate':candidate,'mean_alpha_delta':round(float(np.mean(delta)),5),
                   'bootstrap_ci_95':[round(float(v),5) for v in ci],
                   'candidate_beats_baseline':bool(candidate_beats)}
    completed=all(groups.get(h,{}).get('status')=='OOS_EVALUATED' for h in DAYS)
    return {'status':'OOS_COMPARISON_COMPLETE' if completed else 'NOT_VERIFIED_ALL_HORIZONS',
            'tradeable_profitability_proven':False,
            'note':'此報告僅依提交的真實 PIT 樣本進行固定權重樣本外比較；不能單憑此證明未來績效或權重全域最優。',
            'horizons':groups}


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--observations',default='')
    ap.add_argument('--output',default='data/strategy_weight_validation_v188.json')
    args=ap.parse_args()
    report={'audit':weight_audit(),
            'oos':evaluate_oos(pd.read_csv(args.observations) if args.observations else None)}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    print(json.dumps({'audit_ok':report['audit']['weight_arithmetic_verified'],
                      'oos_status':report['oos']['status'],'output':str(path)},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
