"""Same-date paired development comparison; never converts missing returns to zero.

Usage: python compare_v195.py --old OLD_REPLAY.zip --new NEW_REPLAY.zip --output out
The archives must contain v19_results.json; or pass that JSON file directly.
"""
from __future__ import annotations
import argparse
import json
import io
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd


def load_report(path):
    path=Path(path)
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as z:
            members=[n for n in z.namelist() if Path(n).name=='v19_results.json']
            if len(members)!=1:raise ValueError('Exactly one v19_results.json is required')
            if z.getinfo(members[0]).file_size>200_000_000:raise ValueError('Report too large')
            report=json.loads(z.read(members[0]))
            if 'input_prices.csv.gz' in z.namelist():
                report['_comparison_prices']=pd.read_csv(io.BytesIO(z.read('input_prices.csv.gz')),compression='gzip')
            return report
    report=json.loads(path.read_text(encoding='utf-8'))
    prices=path.parent/'input_prices.csv.gz'
    if prices.exists():report['_comparison_prices']=pd.read_csv(prices)
    return report


def equivalent_prices(a,b):
    """Allow CSV floating representation only, not different dates or prices."""
    if a is None or b is None or set(a.columns)!=set(b.columns):return False
    keys=['ticker','date']
    if a.duplicated(keys).any() or b.duplicated(keys).any():return False
    a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
    if len(a)!=len(b):return False
    numeric={'open','high','low','close','volume','adj_close'}
    for col in a:
        if col in numeric:
            if not np.allclose(pd.to_numeric(a[col],errors='coerce'),pd.to_numeric(b[col],errors='coerce'),rtol=1e-12,atol=1e-10,equal_nan=True):return False
        elif col=='date':
            if not pd.to_datetime(a[col]).equals(pd.to_datetime(b[col])):return False
        elif not a[col].fillna('').astype(str).equals(b[col].fillna('').astype(str)):return False
    return True


def compare(old,new):
    if old.get('cutoffs')!=new.get('cutoffs'):raise ValueError('Cutoff lists differ; compare deliberately before pairing')
    essential=['horizons','stocks','top_k','min_universe','min_turnover','buy_fee','sell_fee','sell_tax',
               'benchmark_sell_tax','slippage','max_entry_gap','universe_seed','prefer_adjusted','as_of']
    differences={k:[old.get('settings',{}).get(k),new.get('settings',{}).get(k)] for k in essential
                 if old.get('settings',{}).get(k)!=new.get('settings',{}).get(k)}
    if differences:raise ValueError('Incompatible comparison settings: '+str(differences))
    if old.get('data_hash') and new.get('data_hash') and old['data_hash']!=new['data_hash']:
        if not equivalent_prices(old.get('_comparison_prices'),new.get('_comparison_prices')):
            raise ValueError('Input prices differ; no paired comparison permitted')
    a=pd.DataFrame(old['results']);b=pd.DataFrame(new['results'])
    rows=[]
    for strategy in ['baseline','training_challenger_research','adaptive','guarded']:
        left=a[a.strategy==strategy];right=b[b.strategy==strategy]
        x=left.merge(right,on=['date','horizon','strategy'],suffixes=('_old','_new'),validate='one_to_one')
        valid=x.complete_old.astype(bool)&x.complete_new.astype(bool)
        valid &= np.isfinite(pd.to_numeric(x.mean_net_old,errors='coerce'))&np.isfinite(pd.to_numeric(x.mean_net_new,errors='coerce'))
        valid &= (x.price_basis_old==x.price_basis_new)|x.price_basis_old.eq('CASH')|x.price_basis_new.eq('CASH')
        x=x[valid]
        for h,g in x.groupby('horizon'):
            delta=g.mean_net_new-g.mean_net_old
            rows.append({'horizon':int(h),'strategy':strategy,'paired_cases':len(g),
                'old_mean':float(g.mean_net_old.mean()),'new_mean':float(g.mean_net_new.mean()),
                'delta_percentage_points':float(delta.mean()*100),
                'improved':int((delta>1e-10).sum()),'worsened':int((delta< -1e-10).sum()),
                'same':int((delta.abs()<=1e-10).sum()),'not_compounded':True})
    counts=[]
    for label,r in [('old',old),('new',new)]:
        for h in r['settings']['horizons']:
            fits=[f for f in r.get('fits',[]) if f['horizon']==h]
            counts.append({'version':label,'horizon':h,'cases':len(fits),
                'candidate_fitted':sum(bool(f.get('challenger_weights')) for f in fits),
                'adopted':sum(f.get('status')=='SPARSE_INNER_VALIDATION_ACCEPTED' for f in fits),
                'input_count_distribution':dict(pd.Series([len([k for k,v in f.get('challenger_weights',{}).items() if v>0])
                    for f in fits if f.get('challenger_weights')]).value_counts().items())})
    return rows,counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();old=load_report(args.old);new=load_report(args.new)
    rows,counts=compare(old,new);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'v194_v195_paired_returns.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'v194_v195_fit_counts.csv',index=False)
    data={'comparison':rows,'fit_counts':counts,'same_exposed_dates_not_new_blind_holdout':True,
          'stored_frame_hashes_equal':old.get('data_hash')==new.get('data_hash'),
          'exported_price_snapshots_equivalent':equivalent_prices(old.get('_comparison_prices'),new.get('_comparison_prices')),
          'csv_numeric_tolerance':{'rtol':1e-12,'atol':1e-10}}
    # Pandas counts may be numpy integers; normalization is explicit.
    (out/'v195_comparison.json').write_text(json.dumps(data,indent=2,default=lambda x:int(x)),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(counts).to_string(index=False))


if __name__=='__main__':main()
