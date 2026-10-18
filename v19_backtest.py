"""Ten seeded cutoff dates, causal nested walk-forward, auditable trade records.

Cutoff T -> entry T+1 open -> exit T+h close on a COMMON benchmark calendar.
Historical outcomes are admitted to training only after their exit < cutoff.
A holdout result never decides that same holdout's weights. No-fill slots stay
cash; selected missing exits invalidate portfolio returns rather than quietly
removing losing/delisted names. Overlapping dates are disclosed, not ten
independent experiments. Results describe this data universe, not global optima.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from v19_config import FACTORS, TECH, OPTIONAL, Settings, prior, VERSION
from v19_features import build_panel, rank_date, score_rows, current_candidates
from v19_data import atomic_json, atomic_csv


def net_return(entry,exit,cfg=Settings()):
    a=np.asarray(entry,dtype=float);b=np.asarray(exit,dtype=float)
    with np.errstate(divide='ignore',invalid='ignore'):
        value=b*(1-cfg.slippage)*(1-cfg.sell_fee-cfg.sell_tax)/(a*(1+cfg.slippage)*(1+cfg.buy_fee))-1
    return np.where((a>0)&(b>0)&np.isfinite(a)&np.isfinite(b),value,np.nan)


def calendar_for(panel,cfg):
    benchmark=panel[panel.ticker==cfg.benchmark].date.drop_duplicates().sort_values()
    if len(benchmark)>=cfg.warmup:
        counts=panel.groupby('date').ticker.nunique()
        dense=pd.DatetimeIndex(counts[counts>=max(2,int(panel.ticker.nunique()*.5))].index)
        return pd.DatetimeIndex(benchmark).union(dense).sort_values()
    # Fallback requires cross-section presence; never insert weekends via bdate_range.
    counts=panel.groupby('date').ticker.nunique()
    n=panel.ticker.nunique()
    return pd.DatetimeIndex(counts[counts>=max(2,int(n*.5))].index).sort_values()


def add_outcomes(panel,cfg=Settings()):
    cal=calendar_for(panel,cfg);pieces=[]
    for ticker,base in panel.groupby('ticker',sort=True):
        f=base.set_index('date').reindex(cal)
        c=f.calc_close.astype(float);o=f.calc_open.astype(float);lo=f.calc_low.astype(float);hi=f.calc_high.astype(float)
        raw=f.close.astype(float);vol=f.share_volume.astype(float);entry=o.shift(-1)
        rawopen=f.open.shift(-1);entryvol=vol.shift(-1)
        gap=entry/c-1  # Comparable coordinates across a verified split.
        raw_gap=rawopen/raw-1
        oneprice=(f.high.shift(-1)==f.low.shift(-1))&(gap.abs()>.085)
        gapcap=(gap>cfg.max_entry_gap) if ticker!=cfg.benchmark else pd.Series(False,index=cal)
        missing_entry=rawopen.isna()|entryvol.isna()
        nofill=(~missing_entry)&((entryvol<=0)|gapcap|oneprice)
        entry_state=np.where(missing_entry,'NO_ENTRY_BAR',np.where(entryvol<=0,'NO_VOLUME',
                    np.where(gapcap,'ABOVE_PREDECLARED_GAP_CAP',np.where(oneprice,'ONE_PRICE_SESSION','ASSUMED_OPEN_FILL'))))
        result=pd.DataFrame({'date':cal,'ticker':ticker,'entry_date':pd.Series(cal).shift(-1).to_numpy(),
                             'entry_open':rawopen.to_numpy(),'entry_gap':gap.to_numpy(),
                             'entry_raw_gap':raw_gap.to_numpy(),'fill_state':entry_state})
        for h in cfg.horizons:
            end=c.shift(-h);exitdate=pd.Series(cal,index=cal).shift(-h)
            def forward_sum(series):
                return series.shift(-1).iloc[::-1].rolling(h,min_periods=h).sum().iloc[::-1]
            minimum=lo.shift(-1).iloc[::-1].rolling(h,min_periods=h).min().iloc[::-1]
            maximum=hi.shift(-1).iloc[::-1].rolling(h,min_periods=h).max().iloc[::-1]
            count=forward_sum(c.notna().astype(int))
            suspicious=forward_sum(f.raw_discontinuity.eq(True).astype(int)).fillna(0)>0
            adjcount=forward_sum(f.adjusted_available.eq(True).astype(int))
            matured=exitdate.notna();exit_volume=vol.shift(-h)
            exit_gap=o.shift(-h)/c.shift(-(h-1))-1
            exit_locked=(f.high.shift(-h)==f.low.shift(-h))&(exit_gap<-.085)
            # Fixed-hold endpoint returns and full-path risk are different
            # contracts. Interior gaps invalidate MAE/MFE/MDD, not known endpoints.
            endpoint_ok=entry.notna()&end.notna()&(~suspicious)&(entryvol>0)&(exit_volume>0)&(~exit_locked)
            path_ok=endpoint_ok&(count==h)
            cost_cfg=replace(cfg,sell_tax=cfg.benchmark_sell_tax) if ticker==cfg.benchmark else cfg
            net=np.asarray(net_return(entry,end,cost_cfg),dtype=float)
            net[~endpoint_ok.to_numpy()]=np.nan
            net[(nofill&matured).to_numpy()]=0.
            net[~matured.to_numpy()]=np.nan
            mae=(minimum/entry-1).clip(upper=0).to_numpy();mfe=(maximum/entry-1).clip(lower=0).to_numpy()
            mae[~path_ok.to_numpy()]=np.nan;mfe[~path_ok.to_numpy()]=np.nan
            mae[(nofill&matured).to_numpy()]=0.;mfe[(nofill&matured).to_numpy()]=0.
            state=np.where(~matured,'NOT_MATURED',np.where(nofill,entry_state,
                     np.where(missing_entry,'NO_ENTRY_BAR',np.where(suspicious,'RAW_CORPORATE_ACTION_DISCONTINUITY',
                     np.where(exit_locked,'EXIT_ONE_PRICE_DOWN_SESSION',np.where(exit_volume<=0,'EXIT_NO_VOLUME',
                     np.where(~endpoint_ok,'MISSING_FORWARD_BAR',np.where(~path_ok,
                              'ASSUMED_OPEN_FILL_PATH_INCOMPLETE','ASSUMED_OPEN_FILL'))))))))
            mdd=np.full(len(cal),np.nan)
            if len(cal)>h:
                windows=np.lib.stride_tricks.sliding_window_view(c.to_numpy()[1:],h)
                starts=entry.to_numpy()[:len(windows)];combined=np.column_stack([starts,windows])
                peaks=np.maximum.accumulate(combined,axis=1)
                with np.errstate(divide='ignore',invalid='ignore'):dd=combined/peaks-1
                mdd[:len(windows)]=np.min(dd,axis=1)
            mdd[~path_ok.to_numpy()]=np.nan;mdd[(nofill&matured).to_numpy()]=0.
            basis=np.where((adjcount==count)&(count>0),'ADJUSTED_PROVIDER',
                           np.where(adjcount==0,'RAW_PRICE_ONLY','MIXED_UNVERIFIED')).astype(object)
            basis[nofill.to_numpy()]='CASH'
            basis[~np.isfinite(net)]='UNKNOWN'
            result[f'net_{h}']=net
            base_return=(end/c-1).where(~suspicious).to_numpy()
            result[f'base_return_{h}']=base_return
            result[f'mae_{h}']=mae;result[f'mfe_{h}']=mfe;result[f'mdd_{h}']=mdd
            result[f'exit_date_{h}']=exitdate.to_numpy();result[f'exit_close_{h}']=f.close.shift(-h).to_numpy()
            result[f'outcome_state_{h}']=state;result[f'return_basis_{h}']=basis
            result[f'path_complete_{h}']=((path_ok|nofill)&matured).to_numpy()
            result[f'missing_path_bars_{h}']=(h-count).where(matured).to_numpy()
        pieces.append(result)
    outcomes=pd.concat(pieces,ignore_index=True)
    return panel.merge(outcomes,on=['ticker','date'],how='left',validate='one_to_one'),cal


def sample_cutoffs(panel,cal,cfg):
    from v192_sampling import cutoff_plan
    return cutoff_plan(panel,cal,cfg)[0]


def weight_candidates(horizon,features,cfg,seed):
    base=prior(horizon,features);w=np.array([base[k] for k in features])
    candidates=[base]
    # Single-group ablations test marginal contribution within TRAINING only.
    for i in range(len(features)):
        q=w.copy();q[i]=0
        if q.sum()>0:candidates.append(dict(zip(features,(q/q.sum()).tolist())))
        if len(candidates)>=cfg.trials:return candidates
    rng=np.random.default_rng(seed)
    while len(candidates)<cfg.trials:
        q=w*np.exp(rng.normal(0,.25,len(w)))
        q=np.maximum(q,0);q=q/q.sum()
        candidates.append(dict(zip(features,q.tolist())))
    return candidates


def group_performance(rows,weights,h,cfg):
    s,coverage=score_rows(rows,weights)
    q=rows.assign(_score=s).dropna(subset=['_score']).sort_values(['_score','ticker'],ascending=[False,True]).head(cfg.top_k)
    if len(q)<cfg.top_k:return np.nan,np.nan
    if q[f'net_{h}'].isna().any():return np.nan,np.nan
    return float(q[f'net_{h}'].mean()),float(q[f'mae_{h}'].mean())


def objective(groups,weights,h,cfg):
    values=[];down=[]
    for g in groups:
        r,m=group_performance(g,weights,h,cfg)
        if np.isfinite(r) and np.isfinite(m):values.append(r);down.append(m)
    if len(values)<max(5,int(len(groups)*.8)):return -np.inf,{'valid_dates':len(values),'total_dates':len(groups)}
    a=np.array(values);b=np.array(down)
    # Same-horizon candidate comparison, net of predefined costs. Penalties are
    # explicit preferences, not fitted constants reported as empirical truth.
    val=float(np.mean(a)+.25*np.mean(b)-.15*np.std(a))
    return val,{'objective':val,'mean_net':float(a.mean()),'mean_mae':float(b.mean()),'valid_dates':len(a),'total_dates':len(groups)}


def batch_objectives(groups,candidates,h,cfg):
    """Vectorized top-k replay. No per-trial pandas sort/groupby loops."""
    if not groups:return [(-np.inf,{'valid_dates':0,'total_dates':0}) for _ in candidates]
    keys=list(candidates[0]);n=len(groups);width=max(len(g) for g in groups)
    a=np.full((n,width,len(keys)),np.nan);y=np.full((n,width),np.nan);down=np.full_like(y,np.nan)
    penalty=np.zeros_like(y)
    for i,g in enumerate(groups):
        g=g.sort_values('ticker');m=len(g)
        a[i,:m,:]=g[keys].to_numpy(dtype=float)
        y[i,:m]=g[f'net_{h}'].to_numpy(dtype=float);down[i,:m]=g[f'mae_{h}'].to_numpy(dtype=float)
        penalty[i,:m]=np.clip(g.bias20.to_numpy(dtype=float)-12,0,15)*.7
    mask=np.isfinite(a);safe=np.nan_to_num(a)
    result=[]
    for weights in candidates:
        w=np.array([weights[k] for k in keys]);w=w/w.sum()
        coverage=mask@w
        raw=safe@w
        score=np.divide(raw,coverage,out=np.full_like(raw,-np.inf),where=coverage>=.75)-penalty
        score[~np.isfinite(score)]=-np.inf
        ids=np.argsort(-score,axis=1,kind='stable')[:,:cfg.top_k]
        selected=np.take_along_axis(y,ids,axis=1);adverse=np.take_along_axis(down,ids,axis=1)
        count=np.isfinite(np.take_along_axis(score,ids,axis=1)).sum(axis=1)
        ok=(count==cfg.top_k)&np.isfinite(selected).all(axis=1)&np.isfinite(adverse).all(axis=1)
        val_dates=int(ok.sum())
        if val_dates<max(5,int(n*.8)):
            result.append((-np.inf,{'valid_dates':val_dates,'total_dates':n}));continue
        means=selected[ok].mean(axis=1);mae=adverse[ok].mean(axis=1)
        obj=float(means.mean()+.25*mae.mean()-.15*means.std())
        result.append((obj,{'objective':obj,'mean_net':float(means.mean()),'mean_mae':float(mae.mean()),
                           'valid_dates':val_dates,'total_dates':n}))
    return result

def fit_before(panel,cutoff,h,cfg=Settings(),seed=1900,context=None):
    if cfg.factor_design == 'balanced_v198':
        from v198_selection import fit_before as balanced_fit
        return balanced_fit(panel,cutoff,h,cfg,seed,context)
    if cfg.factor_design == 'ten_angle_v197':
        from v197_selection import fit_before as ten_fit
        return ten_fit(panel,cutoff,h,cfg,seed,context)
    if cfg.factor_design == 'tournament_v196':
        from v196_selection import fit_before as tournament_fit
        return tournament_fit(panel,cutoff,h,cfg,seed,context)
    if cfg.factor_design == 'sparse_v195':
        from v195_selection import sparse_fit
        return sparse_fit(panel,cutoff,h,cfg,seed,context=context)
    if cfg.factor_design == 'sparse_v194':
        from v194_selection import sparse_fit
        return sparse_fit(panel,cutoff,h,cfg,seed,context=context)
    if cfg.factor_design == 'expanded_v193' and all(k in panel for k in __import__('v193_factors').EXTENDED):
        from v193_selection import expanded_fit
        return expanded_fit(panel,cutoff,h,cfg,seed,context=context)
    from v191_selection import robust_fit
    return robust_fit(panel,cutoff,h,cfg,seed,context=context)


def selected_result(panel,date,weights,h,cfg,strategy):
    q=rank_date(panel,date,weights,cfg).copy()
    record={'date':str(pd.Timestamp(date).date()),'horizon':h,'strategy':strategy,'selected':len(q),'slots':cfg.top_k}
    fields=['ticker','name','date','score','factor_coverage','source','regime','pattern_stage',
            'entry_date','entry_open','entry_gap','fill_state',
            f'exit_date_{h}',f'exit_close_{h}',f'net_{h}',f'base_return_{h}',f'mae_{h}',f'mfe_{h}',f'mdd_{h}',
            f'outcome_state_{h}',f'return_basis_{h}',f'path_complete_{h}',f'missing_path_bars_{h}']+list(FACTORS)
    trades=q.reindex(columns=fields).copy();trades['strategy']=strategy;trades['horizon']=h
    for key,value in weights.items():trades['weight_'+key]=value
    complete=len(q)==cfg.top_k and q[f'net_{h}'].notna().all()
    filled=int(q[f'outcome_state_{h}'].astype(str).str.startswith('ASSUMED_OPEN_FILL').sum()) if len(q) else 0
    path_complete=bool(complete and q[f'path_complete_{h}'].all())
    states=q[f'outcome_state_{h}'].value_counts().to_dict() if len(q) else {}
    basis=set(q[f'return_basis_{h}'])-{'CASH'} if len(q) else set()
    basis_name=next(iter(basis)) if len(basis)==1 else 'CASH' if not basis and len(q) else 'MIXED_UNVERIFIED'
    reason='COMPLETE' if complete else 'TOO_FEW_ELIGIBLE_CANDIDATES' if len(q)<cfg.top_k else 'SELECTED_OUTCOME_MISSING'
    if complete and not path_complete:reason='ENDPOINT_COMPLETE_PATH_INCOMPLETE'
    record.update(complete=bool(complete),filled=filled,missing=int(q[f'net_{h}'].isna().sum()) if len(q) else cfg.top_k,
                  mean_net=float(q[f'net_{h}'].sum()/cfg.top_k) if complete else None,
                  win_fraction=float((q[f'net_{h}']>0).sum()/cfg.top_k) if complete else None,
                  mean_mae=float(q[f'mae_{h}'].mean()) if path_complete else None,
                  mean_position_mdd=float(q[f'mdd_{h}'].mean()) if path_complete else None,
                  path_complete=path_complete,case_reason=reason,outcome_states=states,price_basis=basis_name)
    return record,trades


def _case(panel,day,date,h,cfg,seed,context):
    from v192_factors import case_factors
    w,info,trace=fit_before(panel,date,h,cfg,seed,context=context)
    info['weights']=w
    traces=[dict(t,cutoff=str(date.date()),horizon=h) for t in trace]
    records=[];trades=[]
    for label,weights in {'baseline':(info.get('control_weights') or prior(h,TECH)),'adaptive':w}.items():
        scoring_cfg=replace(cfg,factor_design='legacy_v192') if cfg.factor_design in ('sparse_v194','sparse_v195','tournament_v196','ten_angle_v197','balanced_v198') and (label=='baseline' or info.get('adopted_is_legacy_control')) else cfg
        r,t=selected_result(day,date,weights,h,scoring_cfg,label)
        records.append(r);trades.extend(t.to_dict('records'))
        if label=='adaptive':adaptive_record=r
    from v193_factors import expanded_prior, EXTENDED
    usable=[k for k in FACTORS if k in day and day.loc[day.eligible,k].notna().mean()>=.85 and (cfg.optional_data or k in TECH+EXTENDED)]
    if usable and cfg.factor_design not in ('sparse_v194','sparse_v195','tournament_v196','ten_angle_v197','balanced_v198'):
        expanded=expanded_prior(h,usable)
        er,et=selected_result(day,date,expanded,h,cfg,'expanded_fixed_research')
        er['weights_approved']=False
        records.append(er);trades.extend(et.to_dict('records'))
    if info.get('challenger_weights'):
        cr,ct=selected_result(day,date,info['challenger_weights'],h,cfg,'training_challenger_research')
        cr['weights_approved']=info.get('status') in ('ROBUST_INNER_VALIDATION_ACCEPTED','SPARSE_INNER_VALIDATION_ACCEPTED','SCORECARD_PREQUENTIAL_ACCEPTED')
        records.append(cr);trades.extend(ct.to_dict('records'))
    if cfg.factor_design in ('ten_angle_v197','balanced_v198'):
        from v197_selection import profiles_for
        for name,sw in profiles_for(h,cfg).items():
            if name=='legacy_control':continue
            sr,st=selected_result(day,date,sw,h,cfg,'ten_shadow_'+name)
            sr['fixed_diagnostic_only']=True
            records.append(sr);trades.extend(st.to_dict('records'))
        # Separate market-risk abstention study; NOT an uplift of stock-picking skill.
        # Daily breadth is computed at T. No future intraday reversal claim.
        current=day.loc[day.eligible,'p197_market_gate'] if 'p197_market_gate' in day else pd.Series(dtype='object')
        gate_state=str(current.mode().iat[0]) if not current.empty else 'UNKNOWN'
        if gate_state=='CAUTION':
            records.append({'date':str(date.date()),'horizon':h,'strategy':'ten_market_cash_RESEARCH_ONLY',
                            'selected':0,'slots':cfg.top_k,'mean_net':0.,'complete':True,
                            'filled':0,'missing':0,'mean_mae':0.,'price_basis':'CASH',
                            'case_reason':'PREDECLARED_MARKET_RISK_CASH',
                            'research_only':True,'not_selection_improvement':True})
        else:
            pool_profiles=profiles_for(h,cfg)
            sw=pool_profiles.get('strong_quality_trend',pool_profiles.get('high_52',pool_profiles['legacy_control']))
            mr,mt=selected_result(day,date,sw,h,cfg,'ten_market_cash_RESEARCH_ONLY')
            mr['case_reason']='SUPPORTIVE_OR_NEUTRAL_RESEARCH_RISK_POLICY'
            mr['research_only']=True
            records.append(mr);trades.extend(mt.to_dict('records'))
    if cfg.factor_design=='tournament_v196':
        from v196_factors import scorecard_weights
        for name,sw in scorecard_weights(h,cfg).items():
            if name=='legacy_control':continue
            sr,st=selected_result(day,date,sw,h,cfg,'simple_control_'+name)
            sr['fixed_diagnostic_only']=True
            records.append(sr)
    guard=dict(adaptive_record,strategy='guarded',decision='FOLLOW_RESEARCH_CANDIDATE')
    if not info.get('allocation_allowed',False):
        guard.update(mean_net=0.,complete=True,selected=0,filled=0,missing=0,
                     mean_mae=0.,mean_position_mdd=0.,win_fraction=None,
                     path_complete=True,case_reason='PREDECLARED_CASH_POLICY',price_basis='CASH',
                     decision=info.get('allocation_reason','INSUFFICIENT_EVIDENCE'))
    records.append(guard)
    b=day[day.ticker==cfg.benchmark]
    val=float(b[f'net_{h}'].iloc[0]) if len(b) and pd.notna(b[f'net_{h}'].iloc[0]) else None
    records.append({'date':str(date.date()),'horizon':h,'strategy':'benchmark_0050','mean_net':val,
                    'complete':val is not None,'selected':int(len(b)),
                    'case_reason':str(b[f'outcome_state_{h}'].iloc[0]) if len(b) else 'NO_BENCHMARK_SIGNAL_BAR',
                    'price_basis':str(b[f'return_basis_{h}'].iloc[0]) if len(b) else 'UNKNOWN',
                    'path_complete':bool(b[f'path_complete_{h}'].iloc[0]) if len(b) else False})
    universe=day[day.eligible]
    ok=not universe.empty and universe[f'net_{h}'].notna().all()
    records.append({'date':str(date.date()),'horizon':h,'strategy':'equal_weight_universe',
                    'mean_net':float(universe[f'net_{h}'].mean()) if ok else None,
                    'complete':bool(ok),'selected':len(universe)})
    audit_weights=info.get('challenger_weights',w) if cfg.factor_design in ('sparse_v194','sparse_v195','tournament_v196','ten_angle_v197','balanced_v198') else expanded if usable else w
    audit_label='sparse_training_candidate' if cfg.factor_design in ('sparse_v194','sparse_v195','tournament_v196','ten_angle_v197','balanced_v198') else 'expanded_fixed_research' if usable else 'adaptive'
    return {'records':records,'trades':trades,'fit':info,'trials':traces,
            'factors':[dict(x,audit_strategy=audit_label) for x in case_factors(day,date,h,cfg,audit_weights)] if cfg.factor_audit else []}


def analyze(prices,evidence=None,cfg=Settings(),progress=None,manifest=None,checkpoint_dir=None,cancel_check=None):
    from v192_sampling import cutoff_plan,fit_seed
    from v192_checkpoint import CaseStore,run_identity,RunCancelled
    from v192_factors import attach_signals,summary as factor_summary
    from v192_fast import ReplayContext
    if progress:progress('features',0,1,'Building causal factors once for the entire history')
    panel=build_panel(prices,evidence,cfg)
    panel,cal=add_outcomes(panel,cfg)
    panel=attach_signals(panel)
    dates,plan=cutoff_plan(panel,cal,cfg)
    sources=set(panel.source.astype(str))
    data_kind='SYNTHETIC_TEST_ONLY' if any('SYNTHETIC' in x for x in sources) else ('IMPORTED_UNVERIFIED' if any('unverified' in x for x in sources) else 'PROVIDER_HISTORY')
    identity,metadata=run_identity(prices,evidence,cfg)
    store=CaseStore(checkpoint_dir,identity,dict(metadata,sampling_plan=plan))
    common={'data_kind':data_kind,'version':VERSION,'settings':asdict(cfg),'data_start':str(panel.date.min().date()),
            'data_end':str(panel.date.max().date()),'symbols':int(panel.ticker.nunique()),'bars':len(panel),
            'calendar_sessions':len(cal),'manifest':manifest or {},'as_of':cfg.end(),
            'data_hash':metadata['prices'],'evidence_hash':metadata['evidence'],'run_identity':identity,
            'is_buy_signal':False,'global_optimum_proven':False,'sampling_plan':plan,
            'evaluation_role':cfg.evaluation_role,'new_independent_holdout_proven':False,
            'benchmark_policy':'next_common_session_open_no_stock_gap_cap',
            'research_rule':'nonoverlap_inner_blocks_challenger_selected_on_training_only',
            'caveats':['Current/user universe is not verified historical membership; survivorship bias remains.',
                       'Adjusted-price coverage differs by provider. Raw price returns are not total returns.',
                       'Daily open fills and slippage are assumptions, not executed trades.',
                       f'{cfg.dates} sampled dates can overlap, especially at 60 sessions; they are not independent trials.',
                       'No historical intraday path: daily low/open/close cannot prove a live reversal prediction.',
                       'No official limit-price series: limit-up hit probability is not reported.',
                       'Factor audits are descriptive development diagnostics, not permission to refit on these same outcomes.']}
    cutoffs=[str(x.date()) for x in dates]
    if not dates:
        return dict(common,status='INSUFFICIENT_HISTORY',cutoffs=[],reason=plan.get('reason'),
                    results=[],weights={},fits=[],trials=[],latest=[],factor_cases=[],factor_summary=[]),panel,pd.DataFrame()
    records=[];trades=[];fits=[];trials=[];last_weights={};factor_cases=[]
    total=len(dates)*len(cfg.horizons);step=0;resumed=0
    context=ReplayContext(panel,cfg)
    by_date={d:g for d,g in panel.groupby('date',sort=False)}
    for date in dates:
        for h in cfg.horizons:
            if cancel_check and cancel_check():raise RunCancelled('Paused; completed case checkpoints retained')
            if progress:progress('validate',step,total,str(date.date())+f' / {h} sessions')
            payload=store.load(date,h)
            if payload is None:
                payload=_case(panel,by_date[date],date,h,cfg,fit_seed(cfg.seed,date,h),context)
                store.save(date,h,payload)
            else:resumed+=1
            records.extend(payload['records']);trades.extend(payload['trades']);fits.append(payload['fit'])
            trials.extend(payload['trials']);factor_cases.extend(payload['factors'])
            step+=1
            if progress:progress('validate',step,total,f'{date.date()} / {h} complete; resumed {resumed}')
    latest=[]
    for i,h in enumerate(cfg.horizons):
        if cancel_check and cancel_check():raise RunCancelled('Paused before latest research fit')
        if progress:progress('latest_fit',i,len(cfg.horizons),f'Latest research weights / {h}')
        date=panel.date.max();w,info,trace=fit_before(panel,date,h,cfg,fit_seed(cfg.seed,date,h),context=context)
        last_weights[str(h)]={'weights':w,'fit':info,'not_approved_for_automatic_orders':True}
        q=current_candidates(panel,cfg,w,h)
        for r in q.head(10).to_dict('records'):latest.append(dict(r,horizon=h))
    complete=sum(r.get('complete',False) for r in records if r['strategy']=='adaptive')
    report=dict(common,status='RESEARCH_COMPLETE' if complete==total else 'RESEARCH_PARTIAL',
                cutoffs=cutoffs,complete_adaptive_cases=complete,expected_adaptive_cases=total,
                results=records,weights=last_weights,fits=fits,trials=trials,latest=latest,
                resumed_cases=resumed,factor_cases=factor_cases,
                adjusted_bar_fraction=float(panel.adjusted_available.mean()),
                fit_status_counts=pd.Series([x['status'] for x in fits]).value_counts().to_dict())
    from v191_reporting import summarize_results, missing_ledger
    if progress:progress('factor_report',0,1,'Summarizing descriptive factor and source diagnostics')
    report['summary'],report['case_diagnostics']=summarize_results(records,cfg.horizons)
    report['source_gap_ledger']=missing_ledger(panel,dates,cfg.horizons,cfg)
    report['complete_benchmark_pairs']=sum(r['benchmark_comparable'] for r in report['case_diagnostics'])
    report['optional_factor_coverage']={k:float(panel[k].notna().mean()) for k in OPTIONAL}
    from v198_universe import historical_pool_diagnostics
    report['universe_diagnostics']=historical_pool_diagnostics(panel,cfg,manifest)
    report['overlapping_horizons']={str(h):sum(cal.get_loc(dates[i])-cal.get_loc(dates[i-1])<h for i in range(1,len(dates))) for h in cfg.horizons}
    report['factor_summary']=factor_summary(factor_cases,cal,cfg)
    if cfg.factor_design in ('ten_angle_v197','balanced_v198'):
        from v197_selection import ten_summary
        from v197_ten_pillars import ten_angle_coverage
        report['ten_pillar_shadow_summary']=ten_summary({'results':records})
        report['ten_pillar_coverage']=ten_angle_coverage(panel)
        report['ten_pillar_design']='TEN_LENSES_RESEARCH_NOT_OPTIMAL_OR_PROVEN'
    from v193_reporting import expanded_summary
    report['expanded_summary']=expanded_summary(records,cal,cfg)
    report['expanded_factor_names']=list(__import__('v193_factors').EXTENDED)
    report['factor_audit_reference']='sparse_training_candidate' if cfg.factor_design in ('sparse_v194','sparse_v195','tournament_v196','ten_angle_v197','balanced_v198') else 'expanded_fixed_research'
    report['research_rule']='sparse_nested_subset_then_weights' if cfg.factor_design in ('sparse_v194','sparse_v195','tournament_v196','ten_angle_v197','balanced_v198') else 'expanded_causal_training_with_unchanged_stability_gate'
    from v194_reporting import sparse_summary
    report['sparse_selection_summary']=sparse_summary(fits)
    report['financial_timing_policy']=cfg.financial_timing
    if cfg.factor_design=='sparse_v195':
        report['model_protocol']='support_local_moments_global_disjoint_cv_fixed_final_blocks_equal_or_ridge'
        report['selection_is_not_profit_guarantee']=True
    if cfg.factor_design in ('ten_angle_v197','balanced_v198'):
        report['model_protocol']='point_in_time_ten_lenses_predeclared_profiles_past_issued_policy_validation'
        report['factor_audit_reference']='ten_lens_past_selected_research_policy'
        report['research_rule']='market_sector_financial_chip_price_risk_liquidity_separate'
    if cfg.factor_design=='tournament_v196':
        report['model_protocol']='same_day_percentile_scorecards_past_only_tournament_separate_prequential_gate'
        report['research_rule']='bounded_simple_policy_and_weight_selection_not_exhaustive'
        report['factor_audit_reference']='past_selected_policy'
        report['policy_weight_grid']=[.25,.5,.75]
        report['policy_diagnostic_is_not_adoption']=True
    report['all_factor_coverage']={k:float(panel.loc[panel.eligible,k].notna().mean()) for k in FACTORS}
    report['financial_revision_ledger_verified']=False
    if progress:progress('factor_report',1,1,'Factor diagnostics complete; no test-driven automatic reweighting')
    return report,panel,pd.DataFrame(trades)


def report_markdown(report):
    out=['# Alpha Radar V'+VERSION+' - historical research report','',f"Status: **{report['status']}**",
         f"Data: {report.get('data_start')} to {report.get('data_end')}; {report.get('symbols')} symbols.",
         '','Entry: next common trading-session open. Exit: T+h close. Costs and gap cap are in settings.',
         'All results are research estimates, NOT actual fills or proof of optimal weights.','',
         '## Seeded cutoff dates','',', '.join(report.get('cutoffs') or []) or 'Insufficient dates','',
         '## Equal-weight portfolio net returns','',
         '| Cutoff | Sessions | Baseline | Adaptive | 0050 | Complete |','|---|---:|---:|---:|---:|---|']
    def pct(x):return '--' if x is None or not np.isfinite(x) else f'{x*100:+.2f}%'
    for d in report.get('cutoffs',[]):
        for h in report['settings']['horizons']:
            q={r['strategy']:r for r in report['results'] if r['date']==d and r['horizon']==h}
            out.append(f"| {d} | {h} | {pct(q.get('baseline',{}).get('mean_net'))} | {pct(q.get('adaptive',{}).get('mean_net'))} | {pct(q.get('benchmark_0050',{}).get('mean_net'))} | {q.get('adaptive',{}).get('complete',False)} |")
    out += ['', '## Paired-case summary (arithmetic means, not a compounded curve)', '',
            '| Sessions | Model pairs | Fixed mean | Adaptive mean | Delta | Benchmark pairs | Adaptive matched | Benchmark matched | Excess |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in report.get('summary',[]):
        out.append(f"| {r['horizon']} | {r['paired_model_cases']} | {pct(r['baseline_mean'])} | {pct(r['adaptive_mean'])} | {pct(r['weight_change_mean'])} | {r['matched_benchmark_cases']} | {pct(r['matched_adaptive_mean'])} | {pct(r['matched_benchmark_mean'])} | {pct(r['mean_excess'])} |")
    out += ['', '## Evaluation role', '', report.get('evaluation_role','UNKNOWN'),
            'These exposed historical dates are development diagnostics, not a fresh blind test.',
            'Fixed/Adaptive are study rankings. Guarded is a separate past-evidence stock/cash policy, not an executed portfolio.',
            'Benchmark missing values are not zeros. Interior gaps can permit endpoint returns but never invented path-risk estimates.',
            '', '## Case diagnostics', '', '| Date | Sessions | Model reason | Benchmark reason | Price bases match |',
            '|---|---:|---|---|---|']
    for r in report.get('case_diagnostics',[]):
        if not r['benchmark_comparable'] or not r['adaptive_path_complete'] or not r['same_price_basis']:
            out.append(f"| {r['date']} | {r['horizon']} | {r['model_reason']} | {r['benchmark_reason']} | {r['same_price_basis']} |")
    out += ['', '## Universe representativeness (CURRENT snapshots, not historic full universe)', '',
            json.dumps(report.get('universe_diagnostics',{}),ensure_ascii=False,indent=2),
            'No historical delisting completeness, verified market capitalization, or representative return population is claimed.']
    out += ['', '## Active optional-factor coverage', '', json.dumps(report.get('optional_factor_coverage',{}),ensure_ascii=False)]
    out+=['','## Data classification','',report.get('data_kind','UNCLASSIFIED'),'','## Weight fit statuses','',json.dumps(report.get('fit_status_counts',{}),ensure_ascii=False),'','## Limitations']+['- '+s for s in report.get('caveats',[])]+['','## Configuration','', '```json',json.dumps(report['settings'],indent=2),'```']
    from v192_factors import markdown as factor_markdown
    out += ['',factor_markdown(report)]
    from v193_reporting import expanded_markdown
    out += ['',expanded_markdown(report)]
    from v194_reporting import sparse_markdown
    out += ['',sparse_markdown(report)]
    return '\n'.join(out)


def save_analysis(output,report,panel,trades,prices=None,evidence=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'v19_results.json',report)
    atomic_json(output/'weights_candidate.json',{'version':VERSION,'as_of':report.get('data_end'),
                 'data_hash':report.get('data_hash'),'research_only':True,'weights':report.get('weights',{})})
    atomic_csv(output/'comparison.csv',pd.DataFrame(report.get('results',[])))
    atomic_csv(output/'expanded_comparison.csv',pd.DataFrame(report.get('expanded_summary',[])))
    atomic_csv(output/'selected_trades.csv',trades)
    atomic_csv(output/'factor_panel.csv.gz',panel)
    (output/'v19_validation.md').write_text(report_markdown(report),encoding='utf-8')
    atomic_csv(output/'case_diagnostics.csv',pd.DataFrame(report.get('case_diagnostics',[])))
    atomic_csv(output/'source_gap_ledger.csv',pd.DataFrame(report.get('source_gap_ledger',[])))
    atomic_json(output/'universe_diagnostics.json',report.get('universe_diagnostics',{}))
    q=(report.get('universe_diagnostics',{}).get('universe_quality') or {})
    if q.get('sector_comparison'):atomic_csv(output/'universe_sector_coverage.csv',pd.DataFrame(q['sector_comparison']))
    atomic_json(output/'fit_traces.json',report.get('trials',[]))
    from v194_reporting import export_sparse
    export_sparse(output,report)
    atomic_json(output/'input_manifest.json',report.get('manifest',{}))
    if prices is not None:
        atomic_csv(output/'input_prices.csv.gz',prices)
    else:
        replay=panel.reindex(columns=['date','ticker','open','high','low','close','share_volume','adj_close','source','name','sector','listed_date','delisted_date','member_source']).rename(columns={'share_volume':'volume'})
        atomic_csv(output/'input_prices.csv.gz',replay)
    if evidence is not None and not evidence.empty:
        atomic_csv(output/'input_evidence.csv.gz',evidence)
    else:
        (output/'input_evidence.csv.gz').unlink(missing_ok=True)
    atomic_json(output/'replay_status.json',{'version':VERSION,'price_inputs_present':True,
        'optional_inputs_present':bool(evidence is not None and not evidence.empty),
        'raw_optional_inputs_required':bool(report.get('settings',{}).get('optional_data')),
        'not_real_performance_validation':report.get('data_kind')!='PROVIDER_HISTORY'})
    atomic_json(output/'sampling_plan.json',report.get('sampling_plan',{}))
    atomic_csv(output/'factor_validation_cases.csv',pd.DataFrame(report.get('factor_cases',[])))
    atomic_csv(output/'factor_validation_summary.csv',pd.DataFrame(report.get('factor_summary',[])))
    from v192_factors import markdown as factor_markdown
    (output/'factor_validation.md').write_text(factor_markdown(report),encoding='utf-8')
    from v191_reporting import bundle
    original_bundle=bundle(output,['v19_validation.md','v19_results.json','comparison.csv','selected_trades.csv',
          'case_diagnostics.csv','source_gap_ledger.csv','fit_traces.json','weights_candidate.json',
          'input_prices.csv.gz','input_evidence.csv.gz','input_manifest.json','replay_status.json',
          'sampling_plan.json','universe_diagnostics.json','universe_sector_coverage.csv','factor_validation_cases.csv','factor_validation_summary.csv','factor_validation.md','expanded_comparison.csv','sparse_factor_selection.csv','sparse_latest_weights.csv','sparse_selection_summary.csv','sparse_protocol.json'])
    import shutil
    shutil.copyfile(original_bundle,output/'v198_diagnostic_bundle.zip')
    shutil.copyfile(original_bundle,output/'v192_diagnostic_bundle.zip')
    shutil.copyfile(original_bundle,output/'v193_diagnostic_bundle.zip')
    shutil.copyfile(original_bundle,output/'v195_diagnostic_bundle.zip')
    shutil.copyfile(original_bundle,output/'v196_diagnostic_bundle.zip')
    shutil.copyfile(original_bundle,output/'v197_diagnostic_bundle.zip')
