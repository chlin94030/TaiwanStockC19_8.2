"""Read a supplied markdown report without inventing its missing trade ledger.

Means are arithmetic across explicitly matched cases, NOT a compounded equity
curve. Zeros are observations. Missing values are never converted into zeros.
No function in this module fits or changes model coefficients.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd


def parse_markdown(text: str) -> dict:
    if len(text) > 5_000_000:
        raise ValueError('Summary report too large')
    records = []
    for line in text.splitlines():
        cells = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(cells) < 6 or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', cells[0]):
            continue
        pd.Timestamp(cells[0])
        try:
            horizon = int(cells[1])
        except ValueError:
            continue
        def value(x):
            if x in ('--', '\u2014', '', 'None', 'nan'): return None
            if not re.fullmatch(r'[+-]?\d+(?:\.\d+)?%', x):
                raise ValueError('Unexpected return cell: '+x)
            v = float(x[:-1]) / 100
            if not np.isfinite(v): raise ValueError('Nonfinite return')
            return v
        if (cells[0], horizon) in {(r['date'],r['horizon']) for r in records}:
            raise ValueError('Duplicate date/horizon in report')
        records.append(dict(date=cells[0], horizon=horizon,
                            baseline=value(cells[2]), adaptive=value(cells[3]),
                            benchmark=value(cells[4]), complete=cells[5].lower()=='true'))
    if not records: raise ValueError('No cutoff return table found')
    settings = {}
    blocks=re.findall(r'```json\s*(.*?)```',text,re.S)
    for block in blocks:
        obj=json.loads(block)
        if isinstance(obj,dict) and 'horizons' in obj: settings=obj
    return {'records':records,'settings':settings,
            'cutoffs': sorted({r['date'] for r in records}),
            'source_precision':'rounded_report_cells_not_raw_trade_returns',
            'cannot_identify_best_factors_without_trade_and_factor_ledger':True}


def paired_summary(records: list[dict]) -> list[dict]:
    f=pd.DataFrame(records)
    if f.empty:return []
    out=[]
    for h,g in f.groupby('horizon',sort=True):
        cols=['baseline','adaptive','benchmark']
        for c in cols:g[c]=pd.to_numeric(g[c],errors='coerce')
        paired=g[g.complete].dropna(subset=['baseline','adaptive'])
        if 'model_price_comparable' in paired:
            paired=paired[paired.model_price_comparable.fillna(False).astype(bool)]
        market=paired.dropna(subset=cols)
        if 'benchmark_comparable' in market:
            market=market[market.benchmark_comparable.fillna(False).astype(bool)]
        def mean(df,col):return float(df[col].mean()) if len(df) else None
        d=(paired.adaptive-paired.baseline)
        out.append({'horizon':int(h),'requested_cases':len(g),'paired_model_cases':len(paired),
            'baseline_mean':mean(paired,'baseline'),'adaptive_mean':mean(paired,'adaptive'),
            'weight_change_mean':float(d.mean()) if len(d) else None,
            'improved':int((d>1e-9).sum()),'worsened':int((d<-1e-9).sum()),
            'unchanged':int((d.abs()<=1e-9).sum()),
            'positive_cases':int((paired.adaptive>0).sum()),
            'matched_benchmark_cases':len(market),'matched_adaptive_mean':mean(market,'adaptive'),
            'matched_baseline_mean':mean(market,'baseline'),'matched_benchmark_mean':mean(market,'benchmark'),
            'mean_excess':float((market.adaptive-market.benchmark).mean()) if len(market) else None,
            'benchmark_beats':int((market.adaptive>market.benchmark).sum()),
            'incomplete_dates':g.loc[~g.complete,'date'].tolist(),
            'missing_benchmark_dates':g.loc[g.benchmark.isna(),'date'].tolist()})
    return out


def audit_markdown(text):
    result=parse_markdown(text)
    result['summary']=paired_summary(result['records'])
    result['evaluation_role']='OBSERVED_DEVELOPMENT_DATA_NOT_NEW_HOLDOUT'
    result['findings']=[
        'A summary return table cannot attribute failure to KD, RSI, W-bottom, or any other individual factor.',
        'Compare model returns only on identical complete dates; compare to benchmark on their intersection.',
        'A report marked optional_data=false does not validate revenue or institutional factors.',
        'These exposed cutoff dates are development diagnostics for future revisions.',
        'Overlapping 60-session cases are not independent trials.',
        'Request trade ledger, fit traces and price source coverage before claiming any factor is optimal.'
    ]
    return result


def markdown(result):
    def pct(v):return '--' if v is None else f'{v*100:+.2f}%'
    lines=['# V19.0 supplied report audit', '',
        'Computed from the user-supplied rounded markdown cells. Not a fresh market backtest.',
        'Means across cases are NOT annualized or compounded portfolio returns.', '',
        '| Sessions | Model pairs | Fixed mean | Adaptive mean | Delta | Benchmark pairs | Adaptive matched | 0050 matched | Excess | Beats |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['summary']:
        lines.append(f"| {r['horizon']} | {r['paired_model_cases']} | {pct(r['baseline_mean'])} | {pct(r['adaptive_mean'])} | {pct(r['weight_change_mean'])} | {r['matched_benchmark_cases']} | {pct(r['matched_adaptive_mean'])} | {pct(r['matched_benchmark_mean'])} | {pct(r['mean_excess'])} | {r['benchmark_beats']} |")
    lines += ['', '## Findings']+['- '+x for x in result['findings']]
    lines += ['', '## Missing case ledger']
    for r in result['summary']:
        lines.append(f"- {r['horizon']} sessions: incomplete={r['incomplete_dates']}; benchmark missing={r['missing_benchmark_dates']}")
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('report');p.add_argument('--output',default='v191_report_audit')
    a=p.parse_args();r=audit_markdown(Path(a.report).read_text(encoding='utf-8-sig'))
    Path(a.output+'.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    Path(a.output+'.md').write_text(markdown(r),encoding='utf-8')
    print(markdown(r))

if __name__=='__main__':main()
