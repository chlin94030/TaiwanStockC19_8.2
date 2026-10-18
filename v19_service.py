"""Persistent research jobs and display contracts for V19.

The app owns its worker, not the assistant. No orders are placed. Successful
reports survive a failed refresh; mobile reconnects do not resubmit the job.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import os
import threading
import time
import numpy as np
import pandas as pd
from v19_config import Settings, TZ, FACTORS, TECH, prior, VERSION
from v19_data import (download_history, import_legacy_sqlite, read_json, atomic_json,
                      atomic_csv, clean_prices, SourceError)
from v19_features import build_panel, current_candidates
from v19_backtest import analyze, add_outcomes, save_analysis

_LOCK=threading.RLock()
_JOBS={}


def paths(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    return {'root':root,'state':root/'v19_job.json','last':root/'last_success.json','runroot':root/'runs'}


def job_status(root):
    p=paths(root);key=str(p['root'].resolve());obj=read_json(p['state'])
    with _LOCK:alive=bool(_JOBS.get(key) and _JOBS[key].is_alive())
    if obj.get('state')=='RUNNING' and not alive:
        pid=obj.get('pid')
        if pid==os.getpid() or time.time()-float(obj.get('updated_epoch',0))>120:
            obj=dict(obj,state='INTERRUPTED',message='Worker stopped; cached prices retained. Resume the job.')
    return obj


def read_last_report(root):
    p=paths(root);last=read_json(p['last'])
    target=last.get('directory')
    if target:
        path=Path(target).resolve()
        if path.is_relative_to(p['runroot'].resolve()):return read_json(path/'v19_results.json')
    return {}


def merge_imported_evidence(root, evidence, cfg):
    if not cfg.optional_data:
        return pd.DataFrame()
    extras=Path(root)/'optional_evidence.csv'
    if extras.exists():
        imported=pd.read_csv(extras)
        if not {'ticker','available_at','source'}.issubset(imported):
            raise SourceError('optional CSV needs ticker, available_at, source')
        evidence=pd.concat([evidence,imported],ignore_index=True).drop_duplicates()
    return evidence


def _recent_price_rows(path, rows_per_ticker, cutoff=None):
    """Read compressed history in bounded chunks for the *interactive UI* only.

    Numerical backtests continue using the full unmodified history through
    load_inputs(..., recent_bars=None). Every chunk is reduced to the latest
    N rows per ticker, avoiding construction of an entire 3-year 600-stock
    DataFrame just to show the current screen.
    """
    if not path.exists():
        return pd.DataFrame()
    recent = pd.DataFrame()
    for chunk in pd.read_csv(path, chunksize=40_000):
        if 'ticker' not in chunk or 'date' not in chunk:
            raise SourceError('PRICE_CACHE_MISSING_TICKER_OR_DATE')
        if cutoff:
            chunk = chunk.loc[chunk['date'].astype(str).le(str(cutoff))]
        if chunk.empty:
            continue
        combined = pd.concat((recent, chunk), ignore_index=True)
        recent = (combined.sort_values(['ticker', 'date'], kind='stable')
                  .groupby('ticker', sort=False, group_keys=False)
                  .tail(rows_per_ticker))
    return recent.sort_values(['ticker','date'], kind='stable').reset_index(drop=True)


def load_inputs(root,cfg=Settings(),recent_bars=None):
    root=Path(root);pricepath=root/'history_prices.csv.gz';evpath=root/'historical_evidence.csv.gz'
    prices=(_recent_price_rows(pricepath, int(recent_bars), cfg.end()) if recent_bars
            else pd.read_csv(pricepath)) if pricepath.exists() else pd.DataFrame()
    if prices.empty:
        for candidate in (root.parent/'daily_prices.sqlite',root/'daily_prices.sqlite'):
            if candidate.exists():
                prices=import_legacy_sqlite(candidate,cfg.start(),cfg.end())
                if not prices.empty:break
    evidence=pd.read_csv(evpath) if cfg.optional_data and evpath.exists() else pd.DataFrame()
    evidence=merge_imported_evidence(root,evidence,cfg)
    if not prices.empty:
        candidates=sorted(set(prices.ticker.astype(str))-{cfg.benchmark})
        if cfg.universe_mode!='full_current' and len(candidates)>cfg.stocks:
            from v198_universe import balanced_sample
            metas=prices.drop_duplicates('ticker').set_index('ticker')
            rows=[{'ticker':x,'sector':str(metas.loc[x,'sector']) if 'sector' in metas else 'UNKNOWN'} for x in candidates]
            selected=balanced_sample(rows,cfg.stocks,cfg.universe_seed,cfg.universe_mode)
            keep={x['ticker'] for x in selected}|{cfg.benchmark}
            prices=prices[prices.ticker.isin(keep)].copy()
            if not evidence.empty:evidence=evidence[evidence.ticker.isin(keep)].copy()
    return prices,evidence


def material_for_display(root,cfg=Settings(),recent_bars=None):
    prices,evidence=load_inputs(root,cfg,recent_bars=recent_bars)
    if prices.empty:return pd.DataFrame(),{}
    panel=build_panel(prices,evidence,cfg);panel,_=add_outcomes(panel,cfg)
    return panel,read_last_report(root)


def start_job(root,cfg=Settings(),mode='download_validate',resume=True):
    cfg=replace(cfg,as_of=cfg.end())  # Freeze the date before a long-running job.
    if mode not in ('download_validate','cache_validate','download_only'):raise ValueError('invalid job mode')
    p=paths(root);key=str(p['root'].resolve())
    with _LOCK:
        if _JOBS.get(key) and _JOBS[key].is_alive():return {'started':False,'state':'RUNNING'}
        lockpath=p['root']/'v19_job.lock'
        if lockpath.exists():
            lock=read_json(lockpath)
            stale=time.time()-float(lock.get('at',0))>5400
            try:
                pid=int(lock.get('pid',0));pid_alive=False
                if pid>0:os.kill(pid,0);pid_alive=True
            except (OSError,ValueError):pid_alive=False
            if pid_alive and not stale and int(lock.get('pid',0))!=os.getpid():return {'started':False,'state':'ANOTHER_PROCESS'}
            lockpath.unlink(missing_ok=True)
        fd=os.open(lockpath,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        with os.fdopen(fd,'w') as f:
            import json
            json.dump({'pid':os.getpid(),'at':time.time()},f)
        runid=datetime.now(TZ).strftime('%Y%m%d_%H%M%S_%f')+'_'+cfg.digest()
        cancel_file=p['root']/'v19_cancel.json'
        cancel_file.unlink(missing_ok=True)
        def cancelled():return read_json(cancel_file).get('run_id')==runid
        directory=p['runroot']/runid
        state={'state':'RUNNING','mode':mode,'run_id':runid,'settings':asdict(cfg),'started_at':datetime.now(TZ).isoformat(),
               'resume_enabled':resume,'started_epoch':time.time(),'pid':os.getpid(),'stage':'prepare','done':0,'total':1,'message':'Preparing data','updated_epoch':time.time()}
        atomic_json(p['state'],state)
        def progress(stage,done,total,message):
            from v192_checkpoint import RunCancelled
            if cancelled():raise RunCancelled('Paused at a safe checkpoint; downloads and completed cases retained')
            state.update(stage=stage,done=done,total=total,message=message,updated_epoch=time.time())
            atomic_json(p['state'],state)
        def worker():
            try:
                if mode=='cache_validate':
                    prices,evidence=load_inputs(root,cfg);manifest=read_json(p['root']/'download_manifest.json')
                else:
                    prices,evidence,manifest=download_history(p['root'],cfg,progress)
                    evidence=merge_imported_evidence(root,evidence,cfg)
                if prices.empty:raise SourceError('No price data obtained. See download_manifest.json; no fabricated results were generated.')
                if mode=='cache_validate' and cfg.factor_design=='balanced_v198' and len(set(prices.ticker)-{cfg.benchmark})<cfg.stocks*cfg.min_download_coverage and cfg.universe_mode!='full_current':
                    raise SourceError('CACHE_POOL_BELOW_REQUESTED_TARGET: currently '+str(len(set(prices.ticker)-{cfg.benchmark}))+
                                      ', target '+str(cfg.stocks)+'; use online download to expand the cache')
                if mode=='download_validate' and (manifest.get('download_completeness',1.)<cfg.min_download_coverage or
                    (cfg.factor_design=='balanced_v198' and cfg.universe_mode!='full_current' and
                     'pool_equities' in manifest and manifest.get('requested_target_coverage',0.)<cfg.min_download_coverage)) :
                    raise SourceError('INCOMPLETE_UNIVERSE: '+str(manifest.get('ready_equities'))+'/'+
                                      str(manifest.get('pool_equities'))+' downloaded; resume download and retain caches')
                if mode=='download_only':
                    state.update(state='DONE',message='History downloaded; choose cached validation next.',updated_epoch=time.time())
                else:
                    progress('features',0,1,'Building point-in-time price factors')
                    report,panel,trades=analyze(prices,evidence,cfg,progress,manifest,checkpoint_dir=p['root']/'checkpoints' if resume else None,cancel_check=cancelled)
                    save_analysis(directory,report,panel,trades,prices,evidence)
                    if report.get('cutoffs'):
                        atomic_json(p['last'],{'directory':str(directory.resolve()),'run_id':runid,'status':report['status']})
                    state.update(state='DONE' if report.get('cutoffs') else 'NEEDS_MORE_DATA',message=report.get('reason') or report['status'],report_directory=str(directory.resolve()),updated_epoch=time.time())
            except Exception as e:
                from v192_checkpoint import RunCancelled
                state.update(state='PAUSED' if isinstance(e,RunCancelled) else 'FAILED',
                             message=type(e).__name__+': '+str(e),updated_epoch=time.time())
            finally:
                atomic_json(p['state'],state)
                lockpath.unlink(missing_ok=True)
        t=threading.Thread(target=worker,daemon=True,name='AlphaRadarV19Research')
        _JOBS[key]=t;t.start()
    return {'started':True,'state':'RUNNING','run_id':runid}


def weights_for(report,horizon,asof):
    block=(report.get('weights') or {}).get(str(horizon)) or {}
    fit=block.get('fit') or {}
    if block.get('weights') and str(fit.get('cutoff','9999'))>str(pd.Timestamp(asof).date()):
        return prior(horizon,TECH),'STARTING_PRIOR_NO_FIT'
    if report and report.get('version')!=VERSION:
        return prior(horizon,TECH),'OLD_REPORT_NOT_APPLIED'
    if block.get('weights') and str(fit.get('cutoff','9999'))<=str(pd.Timestamp(asof).date()):
        return block['weights'],fit.get('status','RESEARCH_FIT')
    return prior(horizon,TECH),'STARTING_PRIOR_NO_FIT'


def analog_summary(panel,row,horizon,weights):
    """Past mature, spaced analogous dates. Historical frequencies, not forecasts."""
    asof=pd.Timestamp(row['date']);col=f'net_{horizon}'
    if col not in panel:return {'state':'NO_OUTCOMES','count':0}
    subset=panel[(panel.date<asof)&(panel[f'exit_date_{horizon}']<asof)&panel.eligible&panel[col].notna()].copy()
    keys=[k for k in weights if k in FACTORS and weights[k]>0]
    missing=[k for k in keys if not np.isfinite(pd.to_numeric(row.get(k),errors='coerce'))]
    if missing:return {'state':'MISSING_SELECTED_INPUTS','count':0,'missing':missing}
    if not keys or subset.empty:return {'state':'INSUFFICIENT','count':0}
    values=subset[keys].to_numpy(dtype=float);query=np.array([row[k] for k in keys]);w=np.array([weights[k] for k in keys])
    w=w/w.sum();mask=np.isfinite(values);coverage=mask@w
    distance=np.nansum((values-query)**2*w,axis=1)/np.maximum(coverage,.01)
    distance[coverage<1.-1e-9]=np.inf
    subset['_distance']=distance
    subset=subset[np.isfinite(subset._distance)].sort_values('_distance').drop_duplicates('date')
    cal=pd.DatetimeIndex(panel.date.unique()).sort_values();chosen=[];locs=[]
    for _,item in subset.iterrows():
        loc=cal.get_loc(item.date)
        if all(abs(loc-q)>=horizon for q in locs):chosen.append(item);locs.append(loc)
        if len(chosen)>=40:break
    minimum=8 if horizon>=60 else 15
    if len(chosen)<minimum:return {'state':'INSUFFICIENT_SPACED_SAMPLES','count':len(chosen),'minimum':minimum}
    samples=pd.DataFrame(chosen);net=samples[col].to_numpy(dtype=float)
    return {'state':'HISTORICAL_ANALOG_NOT_CALIBRATED_FORECAST','count':len(samples),
            'mean_net':float(net.mean()),'median_net':float(np.median(net)),'positive_fraction':float((net>0).mean()),
            'q25_net':float(np.quantile(net,.25)),'worst_mae':float(samples[f'mae_{horizon}'].min()),
            'first':str(samples.date.min().date()),'last':str(samples.date.max().date()),'horizon':horizon}


def field_health(panel,report=None):
    if panel is None or panel.empty:return [{'field':'price history','present':0,'total':0,'state':'MISSING'}]
    current=panel[panel.date==panel.date.max()];out=[]
    for col in ('ticker','name','date','close','source','rsi','k','d','macd_hist','bias20','neckline')+FACTORS:
        if col not in current:
            present=0
        elif col in FACTORS or col in ('close','rsi','k','d','macd_hist','bias20','neckline'):
            present=int(np.isfinite(pd.to_numeric(current[col],errors='coerce')).sum())
        else:
            present=int((current[col].notna() & current[col].astype(str).str.strip().ne('')).sum())
        out.append({'field':col,'present':present,'total':len(current),'coverage':present/len(current),
                    'state':'AVAILABLE' if present==len(current) else 'PARTIAL' if present else 'MISSING_NOT_ZERO'})
    kind=(report or {}).get('data_kind','NOT_RUN')
    out.append({'field':'real_market_validation','present':int(bool(report and report.get('cutoffs') and kind=='PROVIDER_HISTORY')),'total':1,
                'state':kind if kind in ('SYNTHETIC_TEST_ONLY','IMPORTED_UNVERIFIED') else (report or {}).get('status','NOT_RUN')})
    return out


def position_size(capital,entry,stop,risk_fraction=.005,max_weight=.2,cost_buffer=.007):
    if not all(np.isfinite(x) for x in (capital,entry,stop,risk_fraction,max_weight,cost_buffer)):
        return {'shares':0,'state':'INVALID_INPUT'}
    if capital<=0 or not 0<stop<entry or not 0<risk_fraction<=.05 or not 0<max_weight<=1:
        return {'shares':0,'state':'INVALID_INPUT'}
    per_share=entry-stop+entry*cost_buffer
    shares=int(min(capital*risk_fraction/per_share,capital*max_weight/(entry*(1+cost_buffer))))
    return {'shares':shares,'position_value':shares*entry,'planned_max_loss':shares*per_share,
            'risk_budget':capital*risk_fraction,'state':'ILLUSTRATION_GAP_RISK_NOT_CAPPED'}


def request_cancel(root):
    """Ask the app's own worker to stop at the next safe checkpoint."""
    p=paths(root);state=job_status(root)
    if state.get('state')!='RUNNING':return False
    atomic_json(p['root']/'v19_cancel.json',{'run_id':state.get('run_id')})
    return True
