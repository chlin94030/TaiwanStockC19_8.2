"""Paired replay diagnostics for the wider research design."""
from __future__ import annotations
import numpy as np
import pandas as pd
from v192_sampling import disjoint_dates


def expanded_summary(records, calendar, cfg):
    df=pd.DataFrame(records);out=[]
    names=('expanded_fixed_research','training_challenger_research','adaptive','guarded')
    for h in cfg.horizons:
        data=df[df.horizon==h]
        base=data[data.strategy=='baseline'].set_index('date')
        bench=data[data.strategy=='benchmark_0050'].set_index('date')
        for label in names:
            q=data[data.strategy==label].set_index('date')
            if q.empty:continue
            idx=base.index.intersection(q.index)
            b=pd.to_numeric(base.reindex(idx).mean_net,errors='coerce')
            c=pd.to_numeric(q.reindex(idx).mean_net,errors='coerce')
            bm=pd.to_numeric(bench.reindex(idx).mean_net,errors='coerce')
            valid=b.notna()&c.notna()
            qbasis=q.reindex(idx).price_basis;basebasis=base.reindex(idx).price_basis
            valid &= ((qbasis==basebasis)|(qbasis=='CASH')|(basebasis=='CASH'))
            valid &= qbasis.isin(['RAW_PRICE_ONLY','ADJUSTED_PROVIDER','CASH'])&basebasis.isin(['RAW_PRICE_ONLY','ADJUSTED_PROVIDER','CASH'])
            # Calendar-select the independent-looking subset BEFORE filtering
            # missing outcomes, so absent losing observations do not reshuffle it.
            frozen=disjoint_dates([pd.Timestamp(d) for d in idx],calendar,h)
            dis=valid&idx.isin([str(d.date()) for d in frozen])
            bm_base=bench.reindex(idx).price_basis
            paired=valid&bm.notna()&((qbasis==bm_base)|(qbasis=='CASH'))
            paired &= bm_base.isin(['RAW_PRICE_ONLY','ADJUSTED_PROVIDER','CASH'])
            entry={'horizon':h,'strategy':label,'paired_cases':int(valid.sum()),
                   'legacy_mean':float(b[valid].mean()) if valid.any() else None,
                   'candidate_mean':float(c[valid].mean()) if valid.any() else None,
                   'mean_gain':float((c-b)[valid].mean()) if valid.any() else None,
                   'improved_cases':int(((c-b)[valid]>1e-12).sum()),
                   'changed_cases':int(((c-b)[valid].abs()>1e-12).sum()),
                   'median_gain':float((c-b)[valid].median()) if valid.any() else None,
                   'benchmark_pairs':int(paired.sum()),
                   'benchmark_mean':float(bm[paired].mean()) if paired.any() else None,
                   'matched_candidate':float(c[paired].mean()) if paired.any() else None,
                   'mean_excess':float((c-bm)[paired].mean()) if paired.any() else None,
                   'benchmark_wins':int((c[paired]>bm[paired]).sum()),
                   'disjoint_cases':int(dis.sum()),
                   'disjoint_mean_gain':float((c-b)[dis].mean()) if dis.any() else None,
                   'research_only':True,'new_holdout':False}
            out.append(entry)
    return out


def expanded_markdown(report):
    def pct(x):return '--' if x is None or pd.isna(x) else f'{x*100:+.2f}%'
    out=['## Research design comparison','',
         'All supplied evaluation dates are exposed development replay, not a new holdout.',
         'Expanded fixed is a new research hypothesis; training challenger is chosen BEFORE each cutoff.',
         'Adaptive uses the legacy baseline if the unchanged inner gate rejects the challenger.',
         'Guarded is a separate cash policy, never an improvement claim about stock ranking.',
         ('Factor ablations use the pre-cutoff sparse candidate when available; unselected factors are not declared ineffective.' if report.get('settings',{}).get('factor_design') in ('sparse_v194','sparse_v195') else 'Factor ablations use the expanded fixed research weights, not zero-weight legacy factors.'),'',
         '| Sessions | Strategy | Paired cases | Legacy | Candidate | Gain | Benchmark pairs | Excess | Disjoint cases | Disjoint gain |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x in report.get('expanded_summary',[]):
        out.append(f"| {x['horizon']} | {x['strategy']} | {x['paired_cases']} | {pct(x['legacy_mean'])} | {pct(x['candidate_mean'])} | {pct(x['mean_gain'])} | {x['benchmark_pairs']} | {pct(x['mean_excess'])} | {x['disjoint_cases']} | {pct(x['disjoint_mean_gain'])} |")
    return '\n'.join(out)
