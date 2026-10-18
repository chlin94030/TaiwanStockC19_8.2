"""Additional financial downloads with explicit availability and revision limits.

Statement `date` is PERIOD END, never silently treated as publication date.
Strict mode caches undated statements but does not admit them to a backtest.
The user may explicitly enable a 120-day-lag research scenario; that is not PIT.
Current vendor snapshots cannot establish original, unrevised historical values.
"""
from __future__ import annotations
import hashlib
import time
import numpy as np
import pandas as pd

DATASETS = ('TaiwanStockFinancialStatements','TaiwanStockBalanceSheet',
            'TaiwanStockCashFlowsStatement','TaiwanStockPER')


def finite(value):
    try:
        x=float(value)
        return x if np.isfinite(x) else None
    except (ValueError,TypeError):return None


def financial_records(ticker, datasets, cfg):
    """Convert explicitly identified statement items; no fuzzy label guesses."""
    out=[]; omitted=[]
    statements={}
    for dataset in DATASETS[:3]:
        df=pd.DataFrame(datasets.get(dataset,[]))
        if not {'date','type','value'}.issubset(df):continue
        for period,g in df.groupby('date',sort=True):
            end=pd.Timestamp(period)
            values={str(r.type):finite(r.value) for r in g.itertuples()}
            announced=[]
            for col in ('announcement_date','published_at','available_at'):
                if col in g:
                    announced.extend(pd.to_datetime(g[col],errors='coerce').dropna().tolist())
            stamps=[pd.Timestamp(x).tz_localize(None) if pd.Timestamp(x).tzinfo is not None else pd.Timestamp(x) for x in announced]
            announced=[x for x in stamps if x>=end]
            if announced:
                available=max(announced)+pd.Timedelta(days=1)
                kind='VENDOR_REPORTED_PUBLICATION_NOT_REVISION_LEDGER'
            elif cfg.financial_timing=='lagged_research':
                available=end+pd.Timedelta(days=cfg.statement_lag_days)
                kind=f'ASSUMED_{cfg.statement_lag_days}D_PERIOD_LAG_NOT_PIT'
            else:
                omitted.append({'ticker':ticker,'dataset':dataset,'period':str(end.date()),'reason':'PUBLICATION_TIME_MISSING_CACHED_NOT_SCORED'})
                continue
            statements[(dataset,end.to_period('Q'))]=(values,available,kind)
    financial=DATASETS[0];balance=DATASETS[1];cashflow=DATASETS[2]
    def get(values,keys):
        for k in keys:
            if values.get(k) is not None:return values[k]
        return None
    for (dataset,quarter),(v,stamp,kind) in sorted(statements.items(),key=lambda x:(str(x[0][1]),x[0][0])):
        period=quarter.end_time.normalize()
        r={'ticker':ticker,'date':str(period.date()),'period_end':str(period.date()),
           'available_at':str(stamp),'source':dataset,'timing':kind,'revision_history_verified':False}
        if dataset==financial:
            eps=get(v,('EPS',));old=statements.get((dataset,quarter-4))
            revenue=get(v,('Revenue','OperatingRevenue'));gp=get(v,('GrossProfit',))
            if revenue is not None and revenue>0 and gp is not None:r['gross_margin']=100*gp/revenue
            if eps is not None and old and old[0].get('EPS') is not None and old[0]['EPS']>0:
                r['eps_growth']=100*(eps/old[0]['EPS']-1)
                stamp=max(stamp,old[1])
            qs=[statements.get((dataset,quarter-j)) for j in range(4)]
            if all(q and q[0].get('EPS') is not None for q in qs):
                r['eps_ttm']=sum(q[0]['EPS'] for q in qs);stamp=max([stamp]+[q[1] for q in qs])
        elif dataset==balance:
            assets=get(v,('TotalAssets','Assets'));debt=get(v,('Liabilities','TotalLiabilities'))
            if assets is not None and assets>0 and debt is not None:r['debt_ratio']=debt/assets
        else:
            cfo=get(v,('CashFlowsFromOperatingActivities','NetCashInflowFromOperatingActivities'))
            # Annual comparison ONLY: no mixing YTD cash flow with quarterly income.
            if quarter.quarter==4 and cfo is not None:
                qs=[statements.get((financial,quarter-j)) for j in range(4)]
                if all(q and q[0].get('IncomeAfterTaxes') is not None for q in qs):
                    income=sum(q[0]['IncomeAfterTaxes'] for q in qs)
                    if income>0:
                        r['cash_conversion_ratio']=cfo/income
                        stamp=max([stamp]+[q[1] for q in qs])
                        r['ratio_basis']='annual_cfo_over_sum_four_quarter_income'
        r['available_at']=str(stamp)
        if any(k in r for k in ('eps_growth','eps_ttm','gross_margin','debt_ratio','cash_conversion_ratio')):out.append(r)
    for row in datasets.get('TaiwanStockPER',[]):
        pe=finite(row.get('PER'));d=pd.to_datetime(row.get('date'),errors='coerce')
        if pd.notna(d) and pe is not None and pe>0:
            out.append({'ticker':ticker,'date':str(d.date()),'available_at':str(d+pd.Timedelta(days=1)),
                        'source':'TaiwanStockPER','pe_ratio':pe,'timing':'next_day_after_vendor_date'})
    return pd.DataFrame(out),omitted


def additional_evidence(client,ticker,start,end):
    from v19_data import read_json, atomic_json, SourceError
    payloads={};code=ticker.split('.')[0]
    for dataset in DATASETS:
        begin=(pd.Timestamp(start)-pd.DateOffset(months=24)).date().isoformat() if dataset!='TaiwanStockPER' else start
        key=hashlib.sha256(f'{dataset}|{code}|{begin}|{end}'.encode()).hexdigest()[:24]
        cache=client.root/'evidence_raw'/f'{key}.json';saved=read_json(cache)
        try:
            if not client.cfg.refresh_history and saved.get('complete') and (saved.get('payload',{}).get('data') or time.time()-float(saved.get('at',0))<3600):
                obj=saved['payload']
            else:
                obj=client.finmind(dataset,code,begin,end)
                if int(obj.get('status',200))!=200:raise SourceError(dataset+': API_FAILED')
                atomic_json(cache,{'complete':True,'dataset':dataset,'ticker':ticker,'at':time.time(),'payload':obj})
            payloads[dataset]=obj.get('data') or []
        except (SourceError,ValueError,TypeError,KeyError) as exc:
            client.events.append({'stage':dataset,'ticker':ticker,'error':str(exc)})
    frame,omitted=financial_records(ticker,payloads,client.cfg)
    if omitted:
        client.events.append({'stage':'financial_availability','ticker':ticker,
                             'skipped_statement_periods':len(omitted),'reason':'PUBLICATION_TIME_MISSING_CACHED_NOT_SCORED'})
    return frame
