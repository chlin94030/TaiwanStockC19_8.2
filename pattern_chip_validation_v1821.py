"""Prospective factor audit for V18.21, not a fitted price prediction model.

Accepts ONLY pre-logged, externally verified market outcomes. Fails closed on
unverified price limits, tradability, PIT, listings, or missing feature vintages.
All comparisons are on a held-out tail with five-decision-day embargo.  One
explanation score is shown per factor; no automatic optimizer touches weights.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

from limitup_validation_v1820 import PROOF_FLAGS, _flag_true
from pattern_chip_lab_v1821 import REVERSAL_WEIGHTS, BREAKOUT_WEIGHTS

FACTOR_FIELDS=tuple(dict.fromkeys((*REVERSAL_WEIGHTS.keys(),*BREAKOUT_WEIGHTS.keys())))
REQUIRED_FIELDS=set(('decision_date','outcome_end_date','ticker','journal_id','baseline_score',
                     'forward_net_return5','forward_max_adverse_pct','future_limit_touch5',
                     'entry_tradeable','historical_feature_pit_verified',*FACTOR_FIELDS,*PROOF_FLAGS))


def validate_factor_evidence(data: pd.DataFrame | None, *, min_days=120, min_stocks=300, top_n=5) -> dict:
    if data is None or data.empty:
        return {'status':'NOT_RUN','reason':'尚無事先封存並有官方後驗結果的實際樣本，無法推算最佳因子權重'}
    missing=sorted(REQUIRED_FIELDS-set(data.columns))
    if missing:
        return {'status':'NOT_VERIFIED','reason':'缺少逐日證據、因子或真實結果','missing_columns':missing}
    x=data.copy()
    for key in ('decision_date','outcome_end_date'):
        x[key]=pd.to_datetime(x[key],errors='coerce')
    if x[['decision_date','outcome_end_date','ticker','journal_id']].isna().any().any():
        return {'status':'NOT_VERIFIED','reason':'決策日／到期日／股票／封存ID有缺失'}
    if x.duplicated(['decision_date','ticker']).any():
        return {'status':'NOT_VERIFIED','reason':'同日同股票重複推薦不可重複計權'}
    if (x['outcome_end_date']<=x['decision_date']).any():
        return {'status':'NOT_VERIFIED','reason':'後驗期間須晚於封存決策日'}
    for key in (*PROOF_FLAGS,'historical_feature_pit_verified'):
        if not x[key].map(_flag_true).all():
            return {'status':'NOT_VERIFIED','reason':f'{key} 未通過真實歷史公告／價格證據'}
    for field in ('baseline_score','forward_net_return5','forward_max_adverse_pct',*FACTOR_FIELDS):
        x[field]=pd.to_numeric(x[field],errors='coerce')
    if x[['baseline_score','forward_net_return5','forward_max_adverse_pct']].isna().any().any():
        return {'status':'NOT_VERIFIED','reason':'基準分數或持有期報酬／回撤缺漏'}
    if not all(x[c].dropna().between(0,100).all() for c in (*FACTOR_FIELDS,'baseline_score')):
        return {'status':'NOT_VERIFIED','reason':'因子分數不在0～100'}
    if not x['future_limit_touch5'].isin([0,1,True,False]).all():
        return {'status':'NOT_VERIFIED','reason':'官方漲停標籤非0/1'}
    if not x['entry_tradeable'].map(lambda y: str(y).lower() in {'0','1','true','false'}).all():
        return {'status':'NOT_VERIFIED','reason':'買得到／買不到標籤缺失'}
    x['future_limit_touch5']=x['future_limit_touch5'].astype(int)
    x['entry_tradeable']=x['entry_tradeable'].map(_flag_true).astype(int)
    dates=sorted(x.decision_date.unique())
    if len(dates)<min_days or x.ticker.nunique()<min_stocks:
        return {'status':'INSUFFICIENT','dates':len(dates),'distinct_stocks':int(x.ticker.nunique()),
                'reason':'樣本不足，至少120個決策日、300檔股票，且來源與後驗結果均須可追溯'}
    split=int(len(dates)*.67)
    held_dates=dates[split+5:]
    if len(held_dates)<25:
        return {'status':'INSUFFICIENT','reason':'樣本外保留5日事件禁運後，決策日不足25日'}
    held=x[x.decision_date.isin(held_dates)].copy()
    held['net_realizable']=np.where(held.entry_tradeable.eq(1),held.forward_net_return5,0.)
    held['touch_tradeable']=held.future_limit_touch5*held.entry_tradeable
    # Explicitly avoid using the sample outcomes to re-choose factor weights.
    def metrics(y,pick_key='baseline_score'):
        ranked=(y.sort_values(['decision_date',pick_key,'ticker'],ascending=[True,False,True])
                .groupby('decision_date').head(top_n))
        if ranked.empty:return None
        return {'picked':len(ranked),'dates':ranked.decision_date.nunique(),
                'limit_touch_rate':round(float(ranked.future_limit_touch5.mean()),4),
                'tradeable_limit_rate':round(float(ranked.touch_tradeable.mean()),4),
                'mean_realizable_5d_net':round(float(ranked.net_realizable.mean()),5),
                'average_max_adverse_pct':round(float(ranked.forward_max_adverse_pct.mean()),3),
                'fill_rate':round(float(ranked.entry_tradeable.mean()),4)}
    report={'status':'RESEARCH_ONLY','holdout_dates':len(held_dates),'holdout_stocks':held.ticker.nunique(),
            'baseline_top':metrics(held),'factors':{},
            'caution':'單因子相關非因果；本報告不推導正式最佳權重，也不宣稱可預測每檔漲停'}
    for name in FACTOR_FIELDS:
        sub=held.dropna(subset=[name]).copy()
        available=float(len(sub)/len(held)) if len(held) else 0
        if available<.75 or sub.decision_date.nunique()<25:
            report['factors'][name]={'status':'INSUFFICIENT_FACTOR_COVERAGE','coverage':round(available,3)}
            continue
        corr=sub[name].corr(sub['future_limit_touch5'],method='spearman')
        # Correlation may be NaN if outcome class is constant: honestly label.
        if pd.isna(corr):
            report['factors'][name]={'status':'NO_EVENT_VARIANCE','coverage':round(available,3)}
            continue
        score_metrics=metrics(sub,name)
        report['factors'][name]={'status':'HOLDOUT_DESCRIPTIVE_ONLY','coverage':round(available,3),
            'spearman_limit_hit':round(float(corr),4),'factor_top':score_metrics}
    return report


def main():
    parser=argparse.ArgumentParser(description='Read pre-verified PIT factor data with real post-event outcomes')
    parser.add_argument('--csv',type=Path)
    parser.add_argument('--output',type=Path,default=Path('v1821_factor_validation.json'))
    args=parser.parse_args()
    if args.csv is None or not args.csv.exists():
        result=validate_factor_evidence(None)
    else:
        try:result=validate_factor_evidence(pd.read_csv(args.csv))
        except (ValueError, OSError) as exc:result={'status':'NOT_VERIFIED','reason':type(exc).__name__}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps({'status':result['status'],'output':str(args.output)},ensure_ascii=False))
    return result['status']

if __name__=='__main__':main()
