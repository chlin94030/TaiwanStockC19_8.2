"""Wider but bounded causal candidate search; the acceptance gate is unchanged.

Numerical defaults are engineering hypotheses, not coefficients inferred from
outer evaluation. A rejected challenger remains visible as research evidence.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from v19_config import FACTORS, TECH, Settings, prior
from v193_factors import EXTENDED, FAMILIES, cap_families, expanded_prior, normalize
from v191_selection import (portfolio_matrix, _benchmark, paired_cohort,
    objective_values, stability_gate, separated_validation, factor_diagnostics)


def candidate_weights(h, features, cfg, diagnostics, train):
    base=prior(h,features);seed=expanded_prior(h,features)
    proposals=[('legacy',base),('expanded_prior',seed),
               ('half_expansion',{k:.5*base[k]+.5*seed[k] for k in features})]
    # Pre-cutoff IC only. Shrink toward the hypothesis instead of inverting a
    # negative indicator after observing the outer test's returns.
    ic={k:float(diagnostics.get(k,{}).get('train_rank_ic_mean') or 0.) for k in features}
    certainty={k:float(diagnostics.get(k,{}).get('positive_ic_fraction') or .5) for k in features}
    subdates=sorted(train.date.unique())[::5]
    corr=train.loc[train.date.isin(subdates),features].corr(method='spearman',min_periods=20).fillna(0.)
    redundancy={k:1+float((corr[k].abs().drop(k)-.7).clip(lower=0).sum()) for k in features}
    learned={k:seed[k]*float(np.clip(1+6*ic[k]+.5*(certainty[k]-.5),.35,1.65))/redundancy[k] for k in features}
    learned=cap_families(learned)
    proposals.append(('ic_shrunk_nonnegative',{k:.5*seed[k]+.5*learned[k] for k in features}))
    for family,keys in FAMILIES.items():
        if not any(seed.get(k,0)>0 for k in keys):continue
        boost=cap_families({k:seed[k]*(1.5 if k in keys else .85) for k in features})
        proposals.append(('family_'+family,boost))
    # Structural ablations are tested in training only; do not use the outer
    # factor audit to pick a favored single indicator.
    for k in features:
        ablate={j:seed[j] if j!=k else 0. for j in features}
        proposals.append(('remove_'+k,cap_families(ablate)))
    out=[];names=[]
    for name,w in proposals:
        if name!='legacy':
            w={k:(1-cfg.weight_shrinkage)*base[k]+cfg.weight_shrinkage*w[k] for k in features}
        w=normalize(w) if name=='legacy' else cap_families(w)
        if any(all(abs(w[k]-q[k])<1e-11 for k in features) for q in out):continue
        names.append(name);out.append(w)
        if len(out)>=cfg.trials:break
    return out,names


def expanded_fit(panel,cutoff,h,cfg=Settings(),seed=1900,context=None):
    cutoff=pd.Timestamp(cutoff);base=prior(h,TECH)
    info={'cutoff':str(cutoff.date()),'horizon':h,'status':'BASELINE_INSUFFICIENT_INDEPENDENT_BLOCKS',
          'features':list(TECH),'holdout_used_for_selection':False,'train_dates':0,'validation_dates':0,
          'allocation_allowed':False,'allocation_reason':'INSUFFICIENT_COMPARABLE_BLOCKS',
          'validation_design':'purged_nonoverlapping_same_date_same_price_basis',
          'factor_design':'expanded_v193','evaluation_role':cfg.evaluation_role}
    if context is not None:grouped=context.before(cutoff,h)
    else:
        available=panel[(panel.date<cutoff)&(panel[f'exit_date_{h}']<cutoff)&panel.eligible]
        grouped=[(d,g) for d,g in available.groupby('date',sort=True) if len(g)>=cfg.min_universe]
    train_pairs,valid_pairs=separated_validation(grouped,h,cfg)
    if not train_pairs:return base,info,[]
    train=pd.concat([g for _,g in train_pairs],ignore_index=True)
    coverage={k:float(train[k].notna().mean()) if k in train else 0. for k in FACTORS}
    features=[k for k in FACTORS if coverage[k]>=.85 and (cfg.optional_data or k in TECH+EXTENDED)]
    if not all(k in features for k in TECH):return base,dict(info,status='BASELINE_TECH_GAP'),[]
    base=prior(h,features)
    diagnostic=context.diagnostics(train_pairs,features,h) if context is not None else factor_diagnostics(train_pairs,features,h)
    candidates,names=candidate_weights(h,features,cfg,diagnostic,train)
    td=[d for d,_ in train_pairs];vd=[d for d,_ in valid_pairs]
    train_matrix=context.portfolio.matrix(train_pairs,candidates,h) if context is not None else portfolio_matrix([g for _,g in train_pairs],candidates,h,cfg)
    bm,bb=_benchmark(panel,td,h,cfg);common=paired_cohort(train_matrix,bm,bb)
    info.update(features=features,feature_coverage=coverage,
        excluded={k:'disabled_or_less_than_85_percent_training_coverage' for k in FACTORS if k not in features},
        train_dates=len(td),validation_dates=len(vd),train_first=str(td[0].date()),train_last=str(td[-1].date()),
        train_max_exit=str(max(g[f'exit_date_{h}'].max() for _,g in train_pairs).date()),
        inner_validation_first=str(vd[0].date()),inner_validation_last=str(vd[-1].date()),
        validation_cutoffs=[str(x.date()) for x in vd],training_cohort_count=int(common.sum()),
        training_cohort_total=len(td),factor_diagnostics=diagnostic,candidate_names=names)
    if common.sum()<max(cfg.train_dates,int(np.ceil(.8*len(td)))):
        return base,dict(info,status='BASELINE_TRAIN_COHORT_INCOMPLETE'),[]
    objectives=objective_values(train_matrix,bm,common,candidates,base)
    traces=[{'stage':'train','candidate':i,'candidate_name':names[i],'weights':w,'objective':objectives[i],
             'common_dates':int(common.sum()),'total_dates':len(td)} for i,w in enumerate(candidates)]
    challenger=max(range(1,len(candidates)),key=lambda i:objectives[i])
    wm=[base,candidates[challenger]]
    info['challenger_weights']=wm[1];info['challenger_name']=names[challenger]
    val_matrix=context.portfolio.matrix(valid_pairs,wm,h) if context is not None else portfolio_matrix([g for _,g in valid_pairs],wm,h,cfg)
    vb,vbb=_benchmark(panel,vd,h,cfg);vc=paired_cohort(val_matrix,vb,vbb);val_count=int(vc.sum())
    info['validation_common_blocks']=val_count
    info['validation_missing_or_incomparable']=[str(vd[i].date()) for i in range(len(vd)) if not vc[i]]
    if val_count<max(cfg.validation_blocks,int(np.ceil(.8*len(vd)))):
        return base,dict(info,status='BASELINE_VALIDATION_COHORT_INCOMPLETE'),traces
    gate=stability_gate(val_matrix['net'][1,vc],val_matrix['net'][0,vc],vb[vc],
                       val_matrix['mae'][1,vc],val_matrix['mae'][0,vc],val_matrix['fill'][1,vc],cfg)
    gate['training_better']=bool(objectives[challenger]>objectives[0])
    accepted=gate['accepted'] and gate['training_better'];selected=1 if accepted else 0
    nets=val_matrix['net'][selected,vc];excess=nets-vb[vc]
    edge=bool(nets.mean()>0 and excess.mean()>0 and (excess>0).mean()>=.6
              and val_matrix['fill'][selected,vc].mean()>=cfg.min_validation_fill)
    info.update(status='ROBUST_INNER_VALIDATION_ACCEPTED' if accepted else 'BASELINE_STABILITY_NOT_PROVEN',
        candidate=challenger if accepted else 0,challenger=challenger,stability_gate=gate,
        validation_mean_excess=float(excess.mean()),validation_mean_net=float(nets.mean()),
        validation_benchmark_win_fraction=float((excess>0).mean()),allocation_allowed=edge,
        allocation_reason='PAST_BLOCK_EDGE_RESEARCH_ONLY' if edge else 'NO_STABLE_BENCHMARK_EDGE')
    for i in (0,1):
        traces.append({'stage':'inner_validation','candidate':0 if i==0 else challenger,
            'candidate_name':'legacy' if i==0 else names[challenger],'weights':wm[i],
            'common_dates':val_count,'cutoffs':[str(vd[j].date()) for j in np.flatnonzero(vc)],
            'net_returns':val_matrix['net'][i,vc].tolist(),'benchmark_returns':vb[vc].tolist(),
            'mean_net':float(val_matrix['net'][i,vc].mean()),'mean_excess':float((val_matrix['net'][i,vc]-vb[vc]).mean())})
    return wm[selected],info,traces
