"""V19.8 universe coverage and board/industry sampling.

Current exchange snapshots are NOT a historical constituent or delisting ledger.
Sampling stratification is based on CURRENT board/industry labels, never future
returns, future OHLC availability or a hindsight liquidity percentile.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd

KNOWN_BOARDS = ('TWSE','TPEX')
EXCLUDED_SECTORS = ('ETF','ETN','權證','受益證券','指數投資證券','債券')


def market(ticker):
    return 'TWSE' if str(ticker).upper().endswith('.TW') else 'TPEX' if str(ticker).upper().endswith('.TWO') else 'OTHER'


def normalize_catalog(rows):
    """Ignore non-common-stock codes and suspicious provider catalog entries."""
    out={}
    for r in rows:
        if not isinstance(r,dict):continue
        ticker=str(r.get('ticker','')).upper().strip()
        code=ticker.split('.')[0]
        if len(code)!=4 or not code.isdigit() or code.startswith('0') or market(ticker)=='OTHER':continue
        sector=str(r.get('sector','') or '').strip() or 'UNKNOWN'
        if any(x.lower() in sector.lower() for x in EXCLUDED_SECTORS):continue
        name=str(r.get('name') or code).strip()
        row=dict(r,ticker=ticker,board=market(ticker),sector=sector,name=name)
        # Do not assert historical membership: date from stock-info snapshots is not IPO date.
        out[ticker]=row
    return [out[k] for k in sorted(out)]


def balanced_sample(rows, limit, seed=1900, mode='balanced'):
    """Sample without replacement; deterministic independently of row ordering.

    balanced: 70% proportionate board-sector mass + 30% square-root mass.
    proportional: source-catalog proportions. full_current: all rows.
    legacy_random: uniform across codes. Selection never sees a stock's returns.
    """
    catalog=normalize_catalog(rows)
    if mode not in ('balanced','proportional','full_current','legacy_random'):
        raise ValueError('UNKNOWN_UNIVERSE_SAMPLING_MODE')
    n=len(catalog)
    if not n:return []
    if mode=='full_current' or limit>=n:return catalog
    if limit<=0:raise ValueError('limit must be positive')
    rng=np.random.default_rng(int(seed))
    if mode=='legacy_random':
        ids=sorted(int(x) for x in rng.choice(n,min(limit,n),replace=False))
        return [catalog[i] for i in ids]
    groups={}
    for row in catalog:
        key=(row['board'],row['sector'])
        groups.setdefault(key,[]).append(row)
    keys=sorted(groups)
    sizes=np.array([len(groups[k]) for k in keys],dtype=int)
    prop=sizes/sizes.sum()
    sqrt=np.sqrt(sizes);sqrt=sqrt/sqrt.sum()
    mass=prop if mode=='proportional' else .70*prop+.30*sqrt
    allocation=np.zeros(len(keys),dtype=int)
    # Reserve a first slot in each category when the requested size permits.
    # Proportional sampling does not force rare categories that are not representative.
    if mode=='balanced' and limit>=len(keys): allocation[:]=1
    remaining=int(limit-allocation.sum())
    for _ in range(remaining):
        available=allocation<sizes
        if not available.any():break
        target=mass*limit
        # Largest remaining deficit with deterministic seeded tie-breaking.
        deficit=np.where(available,target-allocation,-np.inf)
        max_deficit=float(deficit.max())
        contenders=np.flatnonzero(np.isclose(deficit,max_deficit))
        index=int(rng.choice(contenders))
        allocation[index]+=1
    selected=[]
    for i,key in enumerate(keys):
        group=groups[key]
        take=int(allocation[i])
        if take>=len(group):selected.extend(group)
        elif take:
            idx=sorted(int(j) for j in rng.choice(len(group),size=take,replace=False))
            selected.extend(group[j] for j in idx)
    assert len(selected)==min(limit,n)
    return sorted(selected,key=lambda r:r['ticker'])


def summarize_catalog(catalog, selected, downloaded=(),min_bars=180):
    catalog=normalize_catalog(catalog);selected=normalize_catalog(selected)
    available=set(str(x) for x in downloaded)
    def count(rows, field):return dict(sorted(Counter(str(r.get(field,'UNKNOWN')) for r in rows).items()))
    c={r['ticker'] for r in catalog};s={r['ticker'] for r in selected}
    kept=s&available
    sector_count=count(catalog,'sector');sector_selected=count(selected,'sector')
    boards_count=count(catalog,'board');boards_selected=count(selected,'board')
    result={
        'catalog_size':len(c), 'selected_size':len(s), 'downloaded_size':len(kept),
        'catalog_coverage':round(len(s)/len(c),6) if c else 0.,
        'download_coverage':round(len(kept)/len(s),6) if s else 0.,
        'catalog_market':boards_count,'selected_market':boards_selected,
        'catalog_sector':sector_count,'selected_sector':sector_selected,
        'catalog_sector_groups':len(sector_count),'selected_sector_groups':sum(bool(x) for x in sector_selected.values()),
        'missing_selected':sorted(s-available),
        'historical_listing_delisting_verified':False,
        'snapshot_industry_labels_not_point_in_time':True,
        'unknown_market_cap_buckets':True,
        'diversity_is_not_accuracy_or_market_cap_representativeness':True,
    }
    result['catalog_fingerprint']=hashlib.sha256('|'.join(sorted(c)).encode()).hexdigest()
    result['selected_fingerprint']=hashlib.sha256('|'.join(sorted(s)).encode()).hexdigest()
    result['sector_comparison']=[{'board':board,'sector':sector,'catalog':n,
        'selected':sum(r['sector']==sector and r['board']==board for r in selected),
        'downloaded':sum(r['ticker'] in kept and r['sector']==sector and r['board']==board for r in selected)}
        for (board,sector),n in sorted(Counter((r['board'],r['sector']) for r in catalog).items())]
    return result


def historical_pool_diagnostics(panel, cfg, manifest=None):
    """Time-T only coverage, never choose cutoffs on future outcome completeness."""
    manifest=manifest or {}
    valid=panel[panel.ticker!=cfg.benchmark].copy()
    if valid.empty:return {'status':'NO_STOCK_BARS','tested_tickers':0}
    last=valid.date.max()
    p=valid.loc[valid.date.eq(last)].copy()
    n=int(valid.ticker.nunique())
    boards=valid[['ticker']].drop_duplicates().assign(board=lambda d:d.ticker.map(market))
    out={'status':'POPULATION_INCOMPLETE_OR_SURVIVOR_BIASED',
         'tested_tickers':n,'asof_bar_date':str(last.date()),
         'requested_tickers':int(getattr(cfg,'stocks',n)),
         'today_with_bars':int(p.ticker.nunique()),
         'today_eligible':int(p.eligible.sum()) if 'eligible' in p else None,
         'tested_market':boards.board.value_counts().to_dict(),
         'membership_history_verified':False,
         'price_pool_must_not_be_treated_as_market_cap_weighted':True}
    if 'sector' in valid:
        sectors=valid.sort_values('date').drop_duplicates('ticker',keep='last').sector.fillna('UNKNOWN').astype(str)
        out['tested_sectors']=sectors.value_counts().to_dict()
    if 'turnover20' in p:
        z=pd.to_numeric(p.loc[p.eligible,'turnover20'],errors='coerce') if 'eligible' in p else pd.to_numeric(p.turnover20,errors='coerce')
        z=z[np.isfinite(z)&(z>0)]
        if len(z)>=20:
            # Turnover is NOT capitalization. Bin at historical signal date, not after returns.
            out['liquidity_proxy_quartiles_ntd']={str(k):float(v) for k,v in z.quantile([.25,.5,.75]).items()}
    if isinstance(manifest.get('universe_quality'),dict):out['universe_quality']=manifest['universe_quality']
    return out


def history_depth_report(prices, selected, warmup=80):
    """Separate a downloaded symbol from one having adequate historical bars.

    A count of bars is NOT a continuous, corporate-action-correct price history.
    The 252-session indicator and longer label windows need more observations.
    This is a coverage diagnostic, not retrospective stock exclusion.
    """
    names={r['ticker'] for r in normalize_catalog(selected)}
    if prices is None or prices.empty or not names:
        return {'downloaded_price_symbols':0,'warmup_eligible_symbols':0,
                'warmup_eligible_fraction':0.0,'has_252_bar_history_symbols':0,
                'has_252_bar_history_fraction':0.0}
    p=prices.loc[prices.ticker.isin(names)]
    n=p.groupby('ticker').date.nunique() if len(p) else pd.Series(dtype=int)
    enough=int((n>=int(warmup)).sum())
    annual=int((n>=252).sum())
    return {'downloaded_price_symbols':int(len(n)),
            'warmup_eligible_symbols':enough,'warmup_eligible_fraction':round(enough/len(names),6),
            'has_252_bar_history_symbols':annual,
            'has_252_bar_history_fraction':round(annual/len(names),6),
            'price_bars_median':float(n.median()) if len(n) else 0.0,
            'bar_count_is_not_proof_of_complete_price_series':True}
