"""Paired, non-overlapping inner validation for V19.1 research weights.

The attached ten-date report motivated this design, NOT numerical coefficients.
Visible test dates remain development data. Acceptance thresholds below are
predeclared governance preferences; they are NOT claims of statistical power.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from v19_config import Settings, FACTORS, TECH, prior
from v19_features import score_rows


def portfolio_matrix(groups, candidates, h, cfg):
    """Vectorized common-engine replay; selected missing outcomes stay missing."""
    n=len(groups);m=len(candidates)
    result={k:np.full((m,n),np.nan) for k in ('net','mae','fill')}
    bases=np.full((m,n),'UNKNOWN',dtype=object)
    if not n or not m:
        result['basis']=bases;return result
    width=max(len(g) for g in groups);keys=list(candidates[0])
    a=np.full((n,width,len(keys)),np.nan)
    y=np.full((n,width),np.nan);down=np.full_like(y,np.nan);fill=np.zeros_like(y)
    penalty=np.zeros_like(y);basis=np.zeros((n,width),dtype=np.int8)
    codes={'UNKNOWN':0,'RAW_PRICE_ONLY':1,'ADJUSTED_PROVIDER':2,'CASH':3,'MIXED_UNVERIFIED':4}
    labels={v:k for k,v in codes.items()}
    for j,g in enumerate(groups):
        q=g.sort_values('ticker');length=len(q)
        a[j,:length]=q.reindex(columns=keys).to_numpy(float)
        y[j,:length]=q[f'net_{h}'].to_numpy(float);down[j,:length]=q[f'mae_{h}'].to_numpy(float)
        penalty[j,:length]=np.clip(q.bias20.to_numpy(float)-12,0,15)*.7
        fill[j,:length]=q[f'outcome_state_{h}'].astype(str).str.startswith('ASSUMED_OPEN_FILL').to_numpy(float)
        basis[j,:length]=q.get(f'return_basis_{h}',pd.Series('UNKNOWN',index=q.index)).map(codes).fillna(0).to_numpy(np.int8)
    mask=np.isfinite(a);safe=np.nan_to_num(a)
    for i,w in enumerate(candidates):
        v=np.array([w[k] for k in keys],float);v=v/v.sum();coverage=mask@v
        scores=np.divide(safe@v,coverage,out=np.full((n,width),-np.inf),where=coverage>=.75)-penalty
        scores[~np.isfinite(scores)]=-np.inf
        ix=np.argsort(-scores,axis=1,kind='stable')[:,:cfg.top_k]
        ok=np.isfinite(np.take_along_axis(scores,ix,axis=1)).sum(axis=1)==cfg.top_k
        yy=np.take_along_axis(y,ix,axis=1);dd=np.take_along_axis(down,ix,axis=1)
        ff=np.take_along_axis(fill,ix,axis=1);bb=np.take_along_axis(basis,ix,axis=1)
        net_ok=ok&np.isfinite(yy).all(axis=1);risk_ok=ok&np.isfinite(dd).all(axis=1)
        result['net'][i,net_ok]=yy[net_ok].mean(axis=1)
        result['mae'][i,risk_ok]=dd[risk_ok].mean(axis=1)
        result['fill'][i,ok]=ff[ok].mean(axis=1)
        # Cash slots have no conflicting price coordinate; all other slots
        # must share a declared coordinate to enter relative-return validation.
        for j in np.flatnonzero(ok):
            active=np.unique(bb[j][bb[j]!=3])
            if len(active)==0:bases[i,j]='CASH'
            elif len(active)==1:bases[i,j]=labels[int(active[0])]
    result['basis']=bases
    return result


def _benchmark(panel, dates, h, cfg):
    b=panel[panel.ticker==cfg.benchmark].set_index('date').reindex(dates)
    net=b.get(f'net_{h}',pd.Series(np.nan,index=b.index)).to_numpy(float)
    basis=b.get(f'return_basis_{h}',pd.Series('UNKNOWN',index=b.index)).fillna('UNKNOWN').to_numpy(object)
    return net,basis


def paired_cohort(matrix, benchmark, benchmark_basis):
    # Deliberately shared across candidates: no contender picks its own easy dates.
    comparable=((matrix['basis']==benchmark_basis[np.newaxis,:])|(matrix['basis']=='CASH'))
    comparable &= np.isin(matrix['basis'],['RAW_PRICE_ONLY','ADJUSTED_PROVIDER','CASH'])
    comparable &= np.isin(benchmark_basis[np.newaxis,:],['RAW_PRICE_ONLY','ADJUSTED_PROVIDER','CASH'])
    return (np.isfinite(matrix['net']).all(axis=0)&np.isfinite(matrix['mae']).all(axis=0)
            &np.isfinite(benchmark)&comparable.all(axis=0))


def objective_values(matrix, benchmark, common, weights, baseline):
    scores=[]
    for i,w in enumerate(weights):
        if common.sum()<5:
            scores.append(-np.inf);continue
        excess=matrix['net'][i,common]-benchmark[common]
        downside=matrix['mae'][i,common]
        complexity=sum(abs(w.get(k,0)-baseline.get(k,0)) for k in set(w)|set(baseline))
        scores.append(float(excess.mean()+.25*downside.mean()-.15*excess.std()-.01*complexity))
    return scores


def stability_gate(candidate, baseline, benchmark, mae, base_mae, fills, cfg):
    """All inputs are an identical set of non-overlapping validation blocks."""
    c=np.asarray(candidate,float);b=np.asarray(baseline,float);bm=np.asarray(benchmark,float)
    m=np.asarray(mae,float);mb=np.asarray(base_mae,float);f=np.asarray(fills,float)
    if len(c)<cfg.validation_blocks or not all(np.isfinite(x).all() for x in (c,b,bm,m,mb,f)):
        return {'accepted':False,'reason':'INSUFFICIENT_COMPARABLE_BLOCKS','blocks':len(c)}
    d=c-b;leave=(d.sum()-d)/(len(d)-1)
    tests={
        'mean_improvement':bool(d.mean()>=cfg.improvement_floor),
        'not_one_winner':bool(leave.min()>0),
        'majority_blocks_improve':bool((d>0).mean()>=.6),
        'beats_benchmark_on_average':bool((c-bm).mean()>0),
        'positive_net_after_cost':bool(c.mean()>0),
        'tail_risk_not_materially_worse':bool(m.mean()>=mb.mean()-cfg.max_mae_deterioration),
        'sufficient_fill_fraction':bool(f.mean()>=cfg.min_validation_fill),
    }
    return {'accepted':all(tests.values()),'reason':'PASS' if all(tests.values()) else 'STABILITY_GATE_FAILED',
            'blocks':len(c),'tests':tests,'mean_gain':float(d.mean()),'median_gain':float(np.median(d)),
            'worst_leave_one_out_gain':float(leave.min()),'improving_fraction':float((d>0).mean()),
            'mean_excess':float((c-bm).mean()),'mean_net':float(c.mean()),'mean_fill':float(f.mean()),
            'mean_mae':float(m.mean()),'baseline_mae':float(mb.mean())}


def separated_validation(grouped, h, cfg):
    """Take recent disjoint realized-horizon blocks, purging training outcomes."""
    choices=[];next_start=None
    for date,g in reversed(grouped):
        exit_date=g[f'exit_date_{h}'].max()
        if pd.isna(exit_date):continue
        if next_start is None or exit_date < next_start:
            choices.append((date,g));next_start=date
        if len(choices)>=cfg.validation_blocks+2:break
    for n in range(len(choices),cfg.validation_blocks-1,-1):
        valid=list(reversed(choices[:n]));start=valid[0][0]
        train=[]
        for d,g in grouped:
            if d>=start:break
            q=g[g[f'exit_date_{h}']<start]
            if len(q)>=cfg.min_universe:train.append((d,q))
        if len(train)>=cfg.train_dates:return train[-252:],valid
    return [],[]


def factor_diagnostics(train_pairs, features, h):
    values={k:[] for k in features}
    for _,g in train_pairs:
        data=g[features+[f'net_{h}']].replace([np.inf,-np.inf],np.nan)
        ranks=data.rank()
        y=ranks[f'net_{h}']
        # Center pairwise on each finite mask; no scipy requirement.
        a=ranks[features].to_numpy(float);yy=y.to_numpy(float)[:,None]
        mask=np.isfinite(a)&np.isfinite(yy);count=mask.sum(axis=0)
        safe=np.where(mask,a,0);sy=np.where(mask,yy,0)
        denom=np.maximum(count,1)
        xm=safe.sum(axis=0)/denom;ym=sy.sum(axis=0)/denom
        dx=np.where(mask,a-xm,0);dy=np.where(mask,yy-ym,0)
        divisor=np.sqrt((dx*dx).sum(axis=0)*(dy*dy).sum(axis=0))
        ic=np.divide((dx*dy).sum(axis=0),divisor,out=np.full(len(features),np.nan),where=divisor>0)
        for i,k in enumerate(features):
            if count[i]>=10 and np.isfinite(ic[i]):values[k].append(float(ic[i]))
    return {k:{'train_rank_ic_mean':float(np.mean(v)) if v else None,'train_ic_dates':len(v),
               'positive_ic_fraction':float(np.mean(np.array(v)>0)) if v else None,
               'not_causal_or_test_importance':True} for k,v in values.items()}


def robust_fit(panel,cutoff,h,cfg=Settings(),seed=1900,context=None):
    from v19_backtest import weight_candidates
    cutoff=pd.Timestamp(cutoff)
    base=prior(h,TECH)
    info={'cutoff':str(cutoff.date()),'horizon':h,'status':'BASELINE_INSUFFICIENT_INDEPENDENT_BLOCKS',
          'features':list(TECH),'holdout_used_for_selection':False,'train_dates':0,'validation_dates':0,
          'allocation_allowed':False,'allocation_reason':'INSUFFICIENT_COMPARABLE_BLOCKS',
          'validation_design':'purged_nonoverlapping_same_date_same_price_basis',
          'evaluation_role':cfg.evaluation_role}
    if context is not None:
        grouped=context.before(cutoff,h)
    else:
        available=panel[(panel.date<cutoff)&(panel[f'exit_date_{h}']<cutoff)&panel.eligible]
        grouped=[(d,g) for d,g in available.groupby('date',sort=True) if len(g)>=cfg.min_universe]
    train_pairs,valid_pairs=separated_validation(grouped,h,cfg)
    if not train_pairs:return base,info,[]
    train=pd.concat([g for _,g in train_pairs],ignore_index=True)
    coverage={k:float(train[k].notna().mean()) if k in train else 0. for k in FACTORS}
    features=[k for k in FACTORS if coverage[k]>=.85 and (cfg.optional_data or k in TECH)]
    if not all(k in features for k in TECH):return base,dict(info,status='BASELINE_TECH_GAP'),[]
    base=prior(h,features)
    raw=weight_candidates(h,features,cfg,seed)
    # Validate the shrunken weights themselves, not a post-hoc untested blend.
    candidates=[base]
    for w in raw[1:]:
        blend={k:(1-cfg.weight_shrinkage)*base[k]+cfg.weight_shrinkage*w[k] for k in features}
        if not any(all(abs(blend[k]-q[k])<1e-12 for k in features) for q in candidates):candidates.append(blend)
    td=[d for d,_ in train_pairs];vd=[d for d,_ in valid_pairs]
    train_matrix=portfolio_matrix([g for _,g in train_pairs],candidates,h,cfg)
    bm,bb=_benchmark(panel,td,h,cfg);common=paired_cohort(train_matrix,bm,bb)
    info.update(features=features,feature_coverage=coverage,
        excluded={k:'disabled_or_less_than_85_percent_training_coverage' for k in FACTORS if k not in features},
        train_dates=len(td),validation_dates=len(vd),train_first=str(td[0].date()),train_last=str(td[-1].date()),
        inner_validation_first=str(vd[0].date()),inner_validation_last=str(vd[-1].date()),
        validation_cutoffs=[str(x.date()) for x in vd],
        training_cohort_count=int(common.sum()),training_cohort_total=len(td),
        price_basis_mismatch_dates=int((~(train_matrix['basis']==bb[np.newaxis,:]).all(axis=0)).sum()),
        factor_diagnostics=context.diagnostics(train_pairs,features,h) if context is not None else factor_diagnostics(train_pairs,features,h))
    if common.sum()<max(cfg.train_dates,int(np.ceil(.8*len(td)))):
        return base,dict(info,status='BASELINE_TRAIN_COHORT_INCOMPLETE'),[]
    objectives=objective_values(train_matrix,bm,common,candidates,base)
    traces=[{'stage':'train','candidate':i,'weights':w,'objective':objectives[i],
             'common_dates':int(common.sum()),'total_dates':len(td)} for i,w in enumerate(candidates)]
    # Only one challenger is selected on TRAINING. Inner validation is a
    # pass/fail gate, not a second search across six already selected winners.
    challengers=list(range(1,len(candidates)))
    if not challengers:return base,dict(info,status='BASELINE_NO_CHALLENGER'),traces
    challenger=max(challengers,key=lambda i:objectives[i])
    wm=[base,candidates[challenger]]
    val_matrix=portfolio_matrix([g for _,g in valid_pairs],wm,h,cfg)
    vb,vbb=_benchmark(panel,vd,h,cfg);vc=paired_cohort(val_matrix,vb,vbb)
    val_count=int(vc.sum())
    info['validation_common_blocks']=val_count
    info['validation_missing_or_incomparable']=[str(vd[i].date()) for i in range(len(vd)) if not vc[i]]
    if val_count<max(cfg.validation_blocks,int(np.ceil(.8*len(vd)))):
        return base,dict(info,status='BASELINE_VALIDATION_COHORT_INCOMPLETE'),traces
    gate=stability_gate(val_matrix['net'][1,vc],val_matrix['net'][0,vc],vb[vc],
                        val_matrix['mae'][1,vc],val_matrix['mae'][0,vc],val_matrix['fill'][1,vc],cfg)
    # A below-baseline training objective never graduates based on validation luck.
    gate['training_better']=bool(objectives[challenger]>objectives[0])
    accepted=gate['accepted'] and gate['training_better']
    selected=1 if accepted else 0
    nets=val_matrix['net'][selected,vc];excess=nets-vb[vc]
    # Baseline failing does not justify an automatic ETF trade either. This flag
    # only separates stock-study rankings from a predeclared flat/cash policy.
    edge=bool(nets.mean()>0 and excess.mean()>0 and (excess>0).mean()>=.6
              and val_matrix['fill'][selected,vc].mean()>=cfg.min_validation_fill)
    info.update(status='ROBUST_INNER_VALIDATION_ACCEPTED' if accepted else 'BASELINE_STABILITY_NOT_PROVEN',
                candidate=challenger if accepted else 0,challenger=challenger,stability_gate=gate,
                validation_mean_excess=float(excess.mean()),validation_mean_net=float(nets.mean()),
                validation_benchmark_win_fraction=float((excess>0).mean()),
                allocation_allowed=edge,allocation_reason='PAST_BLOCK_EDGE_RESEARCH_ONLY' if edge else 'NO_STABLE_BENCHMARK_EDGE')
    for i in (0,1):
        traces.append({'stage':'inner_validation','candidate':0 if i==0 else challenger,'weights':wm[i],
            'common_dates':val_count,'cutoffs':[str(vd[j].date()) for j in np.flatnonzero(vc)],
            'net_returns':val_matrix['net'][i,vc].tolist(),'benchmark_returns':vb[vc].tolist(),
            'mean_net':float(val_matrix['net'][i,vc].mean()),'mean_excess':float((val_matrix['net'][i,vc]-vb[vc]).mean())})
    return wm[selected],info,traces
