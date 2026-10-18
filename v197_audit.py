"""Auditable matched-case summaries; never feeds retrospective winners into fits.

Usage: python v197_audit.py results_v197/v19_results.json --output result_review
Optional --prices input_prices.csv.gz determines non-overlapping calendar positions.
Outputs are DESCRIPTIVE, not a fresh independent holdout or optimized weights.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from v197_ten_pillars import LENSES, PILLARS, NAMES


def summarize(report, prices=None):
    records=pd.DataFrame(report.get('results',[]))
    if records.empty:raise ValueError('No historical cases to audit')
    active=records[records.strategy.str.startswith('ten_shadow_',na=False)]
    base=records.loc[records.strategy=='baseline', ['date','horizon','mean_net','price_basis','complete']]
    benchmark=records.loc[records.strategy=='benchmark_0050',['date','horizon','mean_net','price_basis','complete']]
    merged=active.merge(base,on=['date','horizon'],suffixes=('','_baseline')).merge(benchmark,on=['date','horizon'],suffixes=('','_benchmark'))
    valid=(merged.complete & merged.complete_baseline & merged.complete_benchmark &
           merged[['mean_net','mean_net_baseline','mean_net_benchmark']].notna().all(axis=1) &
           (merged.price_basis==merged.price_basis_baseline) & (merged.price_basis==merged.price_basis_benchmark))
    merged=merged.loc[valid].copy()
    merged['strategy']=merged.strategy.str.removeprefix('ten_shadow_')
    merged['relative_0050']=merged.mean_net-merged.mean_net_benchmark
    merged['relative_old']=merged.mean_net-merged.mean_net_baseline
    merged['year']=pd.to_datetime(merged.date).dt.year
    if prices is not None and not prices.empty:
        px=prices.copy()
        px.date=pd.to_datetime(px.date)
        # Use the actual observed trading calendar, not day-count approximations.
        cnt=px.groupby('date').ticker.nunique()
        cal=pd.DatetimeIndex(cnt[cnt>=max(2,int(px.ticker.nunique()*.5))].index).sort_values()
        stamp_to_ix={str(x.date()):i for i,x in enumerate(cal)}
    else:stamp_to_ix={}
    out=[]
    for (h,strategy),g in merged.groupby(['horizon','strategy'],sort=True):
        g=g.sort_values('date');h=int(h)
        disjoint=[];last_idx=None
        for row in g.itertuples():
            ix=stamp_to_ix.get(str(row.date))
            if ix is None:continue
            if last_idx is None or ix-last_idx>=h:
                disjoint.append(row);last_idx=ix
        years=g.groupby('year').relative_0050.mean().to_dict()
        dif=g.relative_0050.to_numpy(float)
        out.append({'horizon':h,'strategy':strategy,'valid_paired_dates':len(g),
                    'net_mean':float(g.mean_net.mean()),'legacy_mean':float(g.mean_net_baseline.mean()),
                    'benchmark_mean':float(g.mean_net_benchmark.mean()),
                    'excess_0050_mean':float(dif.mean()),
                    'excess_legacy_mean':float(g.relative_old.mean()),
                    'beats_0050_dates_fraction':float((dif>0).mean()),
                    'worst_paired_excess':float(np.min(dif)),
                    'disjoint_cases':len(disjoint),
                    'disjoint_excess_0050_mean':float(np.mean([x.relative_0050 for x in disjoint])) if disjoint else None,
                    'year_excess_vs_0050':{str(k):float(v) for k,v in years.items()},
                    'development_not_blind_holdout':True})
    features=[]
    for h in sorted(set(x['horizon'] for x in out)):
        rows=[x for x in report.get('factor_summary',[]) if x.get('horizon')==h and x.get('factor') in PILLARS]
        for x in rows:
            features.append({'horizon':h,'factor':x['factor'],'name':NAMES.get(x['factor'],x['factor']),
                             'coverage':x.get('mean_coverage'), 'rank_ic_mean':x.get('rank_ic_mean'),
                             'rank_ic_cases':x.get('ic_valid_cases'),
                             'ablation_delta':x.get('full_minus_removed_mean'),
                             'note':'DESCRIPTIVE_NOT_HOLDOUT_OR_CAUSAL'})
    return out,features,merged


def render_review(report, profiles, features):
    lines=['# Alpha Radar V19.7｜十面向驗證報告','',
           '**重要：以下均為已多次檢視的開發樣本。100日期不等於100次獨立實驗，更不是全球最佳權重。**','',
           '## 十個思考面向','', '| 面向 | 用途／資料界線 |','|---|---|']
    descriptions={
       'market_context':'0050趨勢與股票池廣度，只當市場風險背景，不當個股排序',
       'industry_rotation':'同業強弱排除本股；目前使用現行產業標籤，歷史分類未核，不准正式採用',
       'year_high':'近一年高點位置；用截至當日的歷史資料',
       'trend_quality':'趨勢行進效率、持續相對強度',
       'compression':'先波動收斂，再觀察靠近過去突破價',
       'institutional':'需有發布時間的外資／投信連續買賣資料',
       'revenue':'必須核對原始公告時間；月份標記不當公告日',
       'profit_value':'EPS／獲利品質／現金流與估值；目前資料不足即留空',
       'downside':'過去回撤與下跌波動，不等於未來停損保證',
       'liquidity':'成交額與交易能力；交易跳空、手續費另外扣算'}
    for k,title in LENSES.items():lines.append(f'| {title} | {descriptions[k]} |')
    lines+=['','## 相同日期且同價格基礎配對（固定假說）','',
            '| 日數 | 固定方案 | 組數 | 淨報酬 | 舊模型 | 0050 | 相對0050 | 不重疊數 |',
            '|---:|---|---:|---:|---:|---:|---:|---:|']
    for x in profiles:
        def pct(a):return '—' if a is None else f'{a*100:+.2f}%'
        lines.append(f"| {x['horizon']} | {x['strategy']} | {x['valid_paired_dates']} | {pct(x['net_mean'])} | {pct(x['legacy_mean'])} | {pct(x['benchmark_mean'])} | {pct(x['excess_0050_mean'])} | {x['disjoint_cases']} |")
    lines+=['','## 技術因子的歷史排序訊號','',
            'IC為同一天跨股票的排序關聯，不是上漲機率；刪除因子效果只是描述性診斷，不能事後直接改權重。','',
            '| 日數 | 因子 | 覆蓋 | IC | 有效案件 |', '|---:|---|---:|---:|---:|']
    for x in features:
        ic='—' if x['rank_ic_mean'] is None else f"{x['rank_ic_mean']:+.3f}"
        cov=x['coverage']
        lines.append(f"| {x['horizon']} | {x['name']} | {cov:.0%} | {ic} | {x['rank_ic_cases']} |")
    lines+=['','## 模型採用（不能與固定假說混淆）','']
    for h,b in report.get('weights',{}).items():
        f=b.get('fit',{})
        lines.append(f"- {h}日：最新研究候選 **{f.get('selected_profile','—')}**；歷史檢查：**{f.get('status','—')}**；已採用：{'舊對照' if f.get('adopted_is_legacy_control',True) else '過去區塊已通過的研究候選'}。")
    lines+=['','**尚未建立的證據：**全歷史上市櫃母體、原始財報與營收公告／修訂紀錄、歷史產業分類、即時成交路徑，以及真正未參與設計的新樣本。',
            '原始價格由先前使用者提供之供應商資料重播，尚未逐筆獨立核對官方資料。',
            '若不能勝過被動持有0050，系統應顯示「個股策略優勢尚未確立」，而不是強迫推薦。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('report');p.add_argument('--prices');p.add_argument('--output',default='v197_review')
    a=p.parse_args()
    report=json.loads(Path(a.report).read_text(encoding='utf-8'))
    prices=pd.read_csv(a.prices) if a.prices else None
    summaries,features,pairs=summarize(report,prices)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summaries).to_csv(out/'ten_profile_comparisons.csv',index=False)
    pd.DataFrame(features).to_csv(out/'ten_factor_diagnostics.csv',index=False)
    pairs.to_csv(out/'ten_matched_cases.csv',index=False)
    (out/'V197_Research_Review.md').write_text(render_review(report,summaries,features),encoding='utf-8')
    (out/'ten_angle_summary.json').write_text(json.dumps({'version':report.get('version'),'profiles':summaries,'factors':features,
         'research_only':True,'new_blind_holdout_proven':False},ensure_ascii=False,indent=2),encoding='utf-8')
    print('SAVED',out)
    return 0

if __name__=='__main__':raise SystemExit(main())
