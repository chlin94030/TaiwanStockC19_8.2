"""Clear case denominators and a reproducible, credential-free diagnostic bundle."""
from __future__ import annotations
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from v191_audit import paired_summary


def case_rows(results, horizons):
    rows=[]
    dates=sorted({r['date'] for r in results})
    for date in dates:
        for h in horizons:
            parts={r['strategy']:r for r in results if r['date']==date and r['horizon']==h}
            if not parts:continue
            a=parts.get('adaptive',{});b=parts.get('baseline',{});etf=parts.get('benchmark_0050',{})
            def valid(r):
                x=r.get('mean_net')
                return bool(r.get('complete') and x is not None and np.isfinite(x))
            model=valid(a) and valid(b)
            def basis_match(left,right):
                lb=left.get('price_basis','UNKNOWN');rb=right.get('price_basis','UNKNOWN')
                supported={'RAW_PRICE_ONLY','ADJUSTED_PROVIDER','CASH'}
                return lb in supported and rb in supported and (lb==rb or 'CASH' in (lb,rb))
            # Legacy summary reports have no coordinate metadata. Their numbers
            # remain a descriptive audit, never a strict weight-adoption test.
            legacy_basis=all('price_basis' not in r for r in (a,b,etf))
            model_basis=legacy_basis or basis_match(a,b)
            bench=model and model_basis and valid(etf) and (legacy_basis or basis_match(a,etf))
            rows.append({'date':date,'horizon':h,'baseline':b.get('mean_net'),
                'adaptive':a.get('mean_net'),'benchmark':etf.get('mean_net'),
                'guarded':parts.get('guarded',{}).get('mean_net'),
                'complete':model,'model_price_comparable':model_basis,'benchmark_comparable':bench,
                'basis_verified':not legacy_basis,
                'model_reason':a.get('case_reason','UNKNOWN'),
                'benchmark_reason':etf.get('case_reason','UNKNOWN'),
                'adaptive_selected':a.get('selected',0),'adaptive_filled':a.get('filled',0),
                'adaptive_path_complete':a.get('path_complete',False),
                'adaptive_price_basis':a.get('price_basis','UNKNOWN'),
                'benchmark_price_basis':etf.get('price_basis','UNKNOWN'),
                'same_price_basis':basis_match(a,etf),
                'guard_decision':parts.get('guarded',{}).get('decision')})
    return rows


def summarize_results(results,horizons):
    cases=case_rows(results,horizons)
    summary=paired_summary(cases)
    for row in summary:
        cohort=[r for r in cases if r['horizon']==row['horizon'] and r['benchmark_comparable'] and r['same_price_basis']]
        row['strict_price_basis_cases']=len(cohort)
        row['strict_price_basis_excess']=float(np.mean([r['adaptive']-r['benchmark'] for r in cohort])) if cohort else None
        row['not_compounded_or_annualized']=True
    return summary,cases


def missing_ledger(panel,dates,horizons,cfg):
    """Diagnose source gaps across the whole pool; no selection using these flags."""
    out=[]
    for date in dates:
        frame=panel[panel.date==pd.Timestamp(date)]
        for h in horizons:
            col=f'outcome_state_{h}'
            for row in frame.itertuples():
                state=getattr(row,col,'NO_OUTCOME')
                if state not in ('ASSUMED_OPEN_FILL','NOT_MATURED'):
                    out.append({'date':str(pd.Timestamp(date).date()),'ticker':row.ticker,
                                'horizon':h,'state':state,'is_benchmark':row.ticker==cfg.benchmark,
                                'source':row.source,'eligible_on_signal_date':bool(row.eligible)})
        if frame[frame.ticker==cfg.benchmark].empty:
            out.append({'date':str(pd.Timestamp(date).date()),'ticker':cfg.benchmark,'horizon':None,
                        'state':'NO_BENCHMARK_SIGNAL_BAR','is_benchmark':True})
    return out


def bundle(directory,filenames):
    directory=Path(directory)
    dest=directory/'v191_diagnostic_bundle.zip'
    with ZipFile(dest,'w',compression=ZIP_DEFLATED) as z:
        for name in filenames:
            file=directory/name
            if file.is_file():z.write(file,name)
    return dest


def read_bundle(path,max_uncompressed=200_000_000):
    """Read only documented CSV inputs; no extraction, pickle, or executable files."""
    from io import BytesIO
    import gzip
    allowed=('input_prices.csv.gz','input_evidence.csv.gz','input_manifest.json')
    output={}
    with ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive entries')
        if sum(i.file_size for i in z.infolist())>max_uncompressed:raise ValueError('Archive too large')
        for name in allowed:
            if name not in names:continue
            raw=z.read(name)
            if name.endswith('.gz'):
                # Bound the second compression layer too.
                with gzip.GzipFile(fileobj=BytesIO(raw)) as f:
                    raw=f.read(max_uncompressed+1)
                if len(raw)>max_uncompressed:raise ValueError('CSV too large')
                output[name]=pd.read_csv(BytesIO(raw))
            else:output[name]=json.loads(raw)
    if 'input_prices.csv.gz' not in output:raise ValueError('Bundle does not include replay prices')
    return output['input_prices.csv.gz'],output.get('input_evidence.csv.gz',pd.DataFrame()),output.get('input_manifest.json',{})
