"""V19.4 sparse subset search with nested chronological validation.

A legacy control is NOT mixed into candidate weights. Only selected coefficients
are fitted, shrunk toward equal weight on that SAME support, and scored. Missing
required inputs exclude a stock; no per-stock imputation/weight renormalization.

Outer case outcomes never enter this fit. Development folds are purged by label
end date; the final disjoint validation blocks are never used to choose subsets,
weight coefficients, hyperparameters, or to pick a second candidate after a fail.
The finite search is not an exhaustive proof of a globally optimal strategy.
"""
from __future__ import annotations
from dataclasses import replace
from itertools import combinations
import numpy as np
import pandas as pd
from v19_config import FACTORS, TECH, OPTIONAL, prior, Settings
from v194_factors import ATOMIC, BUSINESS, INTERACTIONS, input_support, compatible
from v193_factors import EXTENDED
from v191_selection import separated_validation, _benchmark, stability_gate

TECH_SEARCH = TECH + EXTENDED + ATOMIC + ('rebound_accumulation_pair',)
KNOWN_BASIS = ('RAW_PRICE_ONLY', 'ADJUSTED_PROVIDER')


def support_weights(values, keys, shrink=0.5):
    """Zeros remain exactly zero. Never blend outside the learned support."""
    v = np.maximum(np.asarray(values, dtype=float), 0.)
    if len(v) != len(keys) or not np.isfinite(v).all():
        raise ValueError('Invalid coefficients')
    active = v > 1e-9
    if not active.any():
        return None
    q = np.zeros(len(v)); q[active] = v[active]/v[active].sum()
    q[active] = shrink*q[active] + (1-shrink)/active.sum()
    return {k: float(q[i]) for i,k in enumerate(keys)}


def _solve(gram, target, keys, ridge, shrink):
    """Nonnegative ridge on training date-normalized ranks (coordinate descent)."""
    if not len(keys): return None
    a = np.asarray(gram, float).copy() + np.eye(len(keys))*ridge
    b = np.asarray(target, float)
    coef = np.zeros(len(keys))
    for _ in range(120):
        old = coef.copy()
        for j in range(len(coef)):
            coef[j] = max(0., (b[j]-a[j]@coef+a[j,j]*coef[j])/a[j,j])
        if np.max(np.abs(coef-old)) < 1e-9: break
    return support_weights(coef, keys, shrink)


def _stats(pairs, features, h):
    """Same complete-input coefficient cohort for all subsets in this fold.

    The portfolio test itself keeps all ex-ante eligible stocks; future missing
    outcomes cannot remove a stock or cause replacement of a selected loser.
    """
    p = len(features); gram=np.zeros((p,p)); rhs=np.zeros(p); count=0; rows=0
    for _, g in pairs:
        x = g.reindex(columns=features).to_numpy(float)/100. - .5
        y = pd.to_numeric(g[f'net_{h}'], errors='coerce').to_numpy(float)
        good = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if good.sum() < 8: continue
        xx=x[good]; yy=pd.Series(y[good]).rank(method='average',pct=True).to_numpy()-.5
        xx=xx-xx.mean(axis=0); yy=yy-yy.mean()
        gram += xx.T@xx/len(xx); rhs += xx.T@yy/len(xx)
        count += 1; rows += len(xx)
    if count: gram/=count; rhs/=count
    diag=np.sqrt(np.maximum(np.diag(gram),0))
    den=diag[:,None]*diag[None,:]
    corr=np.divide(gram,den,out=np.zeros_like(gram),where=den>1e-12)
    return {'gram':gram, 'rhs':rhs, 'dates':count, 'rows':rows, 'corr':corr}


def inner_folds(pairs, h, cfg):
    """Expanding past-only fit folds inside the development data."""
    n=len(pairs)
    starts=np.linspace(max(cfg.train_dates, n//2),n,cfg.sparse_cv_folds+1,dtype=int)
    result=[]
    for a,b in zip(starts[:-1],starts[1:]):
        if b<=a: continue
        start=pairs[a][0]
        fit=[(d,g) for d,g in pairs[:a] if g[f'exit_date_{h}'].max()<start]
        if len(fit)<max(12,cfg.train_dates//2): continue
        test=[]; last_exit=None
        for d,g in pairs[a:b]:
            if last_exit is None or d>last_exit:
                test.append((d,g));last_exit=g[f'exit_date_{h}'].max()
        if test: result.append((fit,test))
    return result


class Prepared:
    """Exact Top-K replay arrays. The sparse and reference engines must agree."""
    def __init__(self, pairs, panel, features, h, cfg):
        self.features=tuple(features); self.cfg=cfg; self.dates=[d for d,g in pairs]
        self.n=len(pairs); width=max([len(g) for d,g in pairs]+[cfg.top_k])
        self.x=np.full((self.n,width,len(features)),np.nan)
        self.y=np.full((self.n,width),np.nan);self.mae=self.y.copy()
        self.fill=np.zeros_like(self.y);self.penalty=np.zeros_like(self.y)
        self.basis=np.full((self.n,width),'UNKNOWN',object)
        self.counts=np.zeros(self.n,dtype=int)
        self.pool=np.full(self.n,np.nan);self.pool_basis=np.full(self.n,'UNKNOWN',object)
        for i,(d,g) in enumerate(pairs):
            g=g.sort_values('ticker',kind='stable'); m=len(g);self.counts[i]=m
            self.x[i,:m]=g.reindex(columns=features).to_numpy(float)
            self.y[i,:m]=g[f'net_{h}'];self.mae[i,:m]=g[f'mae_{h}']
            self.fill[i,:m]=g[f'outcome_state_{h}'].astype(str).str.startswith('ASSUMED_OPEN_FILL')
            self.basis[i,:m]=g[f'return_basis_{h}'].astype(str)
            self.penalty[i,:m]=np.clip(g.bias20.to_numpy(float)-12,0,15)*.7
            bases=set(self.basis[i,:m])-{'CASH'}
            if np.isfinite(self.y[i,:m]).all() and len(bases)<=1:
                self.pool[i]=float(self.y[i,:m].mean())
                self.pool_basis[i]=next(iter(bases)) if bases else 'CASH'
        self.bm,self.bm_basis=_benchmark(panel,self.dates,h,cfg)

    def evaluate(self, weights, strict=True):
        w=np.array([weights.get(k,0.) for k in self.features],float)
        if not np.isfinite(w).all() or w.sum()<=0:
            return {k:np.full(self.n,np.nan) for k in ('net','mae','fill')}
        w/=w.sum(); active=np.flatnonzero(w>0)
        x=self.x[:,:,active]; ww=w[active]
        coverage=np.isfinite(x)@ww
        raw=np.nan_to_num(x)@ww
        threshold=1-1e-9 if strict else .75
        scores=np.divide(raw,coverage,out=np.full_like(raw,-np.inf),where=coverage>=threshold)-self.penalty
        scores[~np.isfinite(scores)]=-np.inf
        ids=np.argsort(-scores,axis=1,kind='stable')[:,:self.cfg.top_k]
        take=lambda a:np.take_along_axis(a,ids,axis=1)
        ok=np.isfinite(take(scores)).sum(axis=1)==self.cfg.top_k
        result={k:np.full(self.n,np.nan) for k in ('net','mae','fill')}
        for key,array in [('net',self.y),('mae',self.mae),('fill',self.fill)]:
            a=take(array); good=ok&np.isfinite(a).all(axis=1)
            result[key][good]=a[good].mean(axis=1)
        basis=np.full(self.n,'UNKNOWN',object); selected=take(self.basis)
        for i in np.flatnonzero(ok):
            b=set(selected[i])-{'CASH'}
            if len(b)<=1: basis[i]=next(iter(b)) if b else 'CASH'
        result['basis']=basis
        return result

    def fixed_reference_mask(self, base):
        good=np.isfinite(self.bm)&np.isfinite(self.pool)&np.isfinite(base['net'])&np.isfinite(base['mae'])
        good &= np.isin(self.bm_basis,KNOWN_BASIS)
        good &= ((base['basis']==self.bm_basis)|(base['basis']=='CASH'))
        good &= ((self.pool_basis==self.bm_basis)|(self.pool_basis=='CASH'))
        return good

    def score(self, outcome, mask, weights, cfg):
        # Evaluation dates fixed by the reference, not selected by the candidate.
        if mask.sum()==0:return None
        valid=mask&np.isfinite(outcome['net'])&np.isfinite(outcome['mae'])
        valid &= (outcome['basis']==self.bm_basis)|(outcome['basis']=='CASH')
        if valid.sum()!=mask.sum():return None
        y=outcome['net'][valid];mae=outcome['mae'][valid]
        reference=.5*(self.bm[valid]+self.pool[valid]); excess=y-reference
        complexity=len(input_support(k for k,v in weights.items() if v>1e-9))
        objective=float(excess.mean()+.25*mae.mean()-.15*excess.std()-cfg.sparse_complexity*complexity)
        return {'objective':objective,'net':float(y.mean()),'excess':float(excess.mean()),
                'dates':int(valid.sum()),'active_inputs':complexity}


def _full_weights(weights):
    return {k:float(weights.get(k,0.)) for k in FACTORS}


def sparse_search(train_pairs, panel, features, h, cfg):
    """Bounded forward/backward/swap search, with separate purged fit folds."""
    splits=inner_folds(train_pairs,h,cfg)
    if len(splits)<2:return None,[],{'reason':'INSUFFICIENT_PURGED_SEARCH_FOLDS','folds':len(splits)}
    baseline=prior(h,TECH); folds=[]
    keys=list(features); index={k:i for i,k in enumerate(keys)}
    for fit,test in splits:
        stats=_stats(fit,keys,h)
        if stats['dates']<10:continue
        prepared=Prepared(test,panel,FACTORS,h,cfg)
        mask=prepared.fixed_reference_mask(prepared.evaluate(baseline,strict=False))
        if mask.sum()<max(1,int(np.ceil(len(test)*.8))):continue
        folds.append((stats,prepared,mask))
    if len(folds)<2:return None,[],{'reason':'INSUFFICIENT_SAME_COHORT_SEARCH_FOLDS','folds':len(folds)}
    full=_stats(train_pairs,keys,h)
    considered={}; history=[]

    def evaluate(subset):
        subset=tuple(sorted(subset))
        if subset in considered:return considered[subset]
        if len(considered)>=cfg.sparse_budget:return None
        if not compatible(subset) or len(input_support(subset))>cfg.sparse_max_inputs:return None
        ix=[index[k] for k in subset]
        if len(ix)>1:
            corr=folds[0][0]['corr'][np.ix_(ix,ix)].copy();np.fill_diagonal(corr,0)
            if np.max(np.abs(corr))>cfg.sparse_corr_limit:return None
        scores=[];fold_weights=[]
        for stats,prepared,mask in folds:
            # Singleton is an explicit simple benchmark even when ridge is zero.
            w={subset[0]:1.} if len(subset)==1 else _solve(stats['gram'][np.ix_(ix,ix)],stats['rhs'][ix],subset,cfg.sparse_ridge,cfg.weight_shrinkage)
            if w is None:break
            result=prepared.score(prepared.evaluate(w),mask,w,cfg)
            if result is None:break
            scores.append(result);fold_weights.append(w)
        valid=len(scores)==len(folds)
        mean=float(np.mean([r['objective'] for r in scores])) if valid else -np.inf
        record={'stage':'sparse_search_cv','candidate':len(considered),'candidate_name':'+'.join(subset),
                'requested_subset':list(subset),'active_inputs':len(input_support(subset)),
                'objective':mean if np.isfinite(mean) else None,'folds':len(scores),'required_folds':len(folds),
                'fold_scores':scores,'fold_weights':fold_weights,'valid':valid}
        considered[subset]=record;history.append(record)
        return record

    for k in keys:evaluate((k,))
    valid=lambda:sorted([r for r in considered.values() if r['valid']],key=lambda r:(-r['objective'],r['active_inputs'],r['candidate_name']))
    single=valid()
    if not single:return None,history,{'reason':'NO_COMPLETE_SEARCH_CANDIDATE','folds':len(folds)}
    shortlist=[r['requested_subset'][0] for r in single if len(r['requested_subset'])==1]
    for pair in combinations(shortlist[:6],2):evaluate(pair)
    # Beam exploration, not a requirement to use every feature family.
    for size in range(2,cfg.sparse_max_inputs+1):
        leaders=valid()[:cfg.sparse_beam]
        for leader in leaders:
            active=leader['requested_subset']
            for k in shortlist[:10]:
                if k not in active and len(active)<size:evaluate(tuple(active)+(k,))
        if len(considered)>=cfg.sparse_budget:break
    for leader in valid()[:cfg.sparse_beam]:
        subset=leader['requested_subset']
        for old in subset:
            smaller=tuple(k for k in subset if k!=old)
            if smaller:evaluate(smaller)
            for new in shortlist[:8]:
                if new not in subset:evaluate(smaller+(new,))
    ranked=valid();best=ranked[0]['objective']
    near=[r for r in ranked if r['objective']>=best-cfg.sparse_near_best]
    choice=min(near,key=lambda r:(r['active_inputs'],-r['objective'],r['candidate_name']))
    subset=choice['requested_subset'];ix=[index[k] for k in subset]
    weights={subset[0]:1.} if len(subset)==1 else _solve(full['gram'][np.ix_(ix,ix)],full['rhs'][ix],subset,cfg.sparse_ridge,cfg.weight_shrinkage)
    if weights is None:return None,history,{'reason':'ZERO_NONNEGATIVE_REFIT','folds':len(folds)}
    chosen=_full_weights(weights)
    support=[k for k,v in chosen.items() if v>0]
    summary={'reason':'SEARCH_FINISHED','search_candidates':len(considered),'budget':cfg.sparse_budget,
             'search_exhaustive':False,'cv_objective':choice['objective'],'cv_best_objective':best,
             'folds':len(folds),'refit_dates':full['dates'],'refit_rows':full['rows'],
             'selected_factors':support,'active_inputs':sorted(input_support(support)),
             'requested_subset':subset,'removed_factors':[k for k in FACTORS if k not in support],
             'fold_ranges':[{'fit_first':str(fit[0][0].date()),'fit_last':str(fit[-1][0].date()),
                 'fit_max_exit':str(max(g[f'exit_date_{h}'].max() for _,g in fit).date()),
                 'score_first':str(test[0][0].date()),'score_last':str(test[-1][0].date())} for fit,test in splits],
             'weight_fit':'nonnegative_ridge_on_within_date_return_ranks',
             'shrinkage':'equal_weight_ONLY_within_positive_support',
             'objective':'net_excess_vs_half_benchmark_half_pool_plus_MAE_minus_dispersion_minus_input_count',
             'coefficient_cohort':'complete_candidate_inputs_in_training_only',
             'scoring_missing_policy':'ALL_selected_inputs_required_no_imputation'}
    return chosen,history,summary


def sparse_fit(panel, cutoff, h, cfg=Settings(), seed=1900, context=None):
    cutoff=pd.Timestamp(cutoff);base=_full_weights(prior(h,TECH))
    info={'cutoff':str(cutoff.date()),'horizon':h,'status':'SPARSE_INSUFFICIENT_PAST_BLOCKS',
          'features':list(TECH),'train_dates':0,'validation_dates':0,'allocation_allowed':False,
          'allocation_reason':'INSUFFICIENT_COMPARABLE_BLOCKS','factor_design':'sparse_v194',
          'holdout_used_for_selection':False,'evaluation_role':cfg.evaluation_role,
          'not_global_optimum':True,'adopted_support':[k for k,v in base.items() if v>0],
          'control_weights':base,'adopted_is_legacy_control':True}
    if context is not None:grouped=context.before(cutoff,h)
    else:
        past=panel[(panel.date<cutoff)&(panel[f'exit_date_{h}']<cutoff)&panel.eligible]
        grouped=[(d,g) for d,g in past.groupby('date',sort=True) if len(g)>=cfg.min_universe]
    train_pairs,valid_pairs=separated_validation(grouped,h,cfg)
    if not train_pairs:return base,info,[]
    train=pd.concat([g for d,g in train_pairs],ignore_index=True)
    coverage={k:float(pd.to_numeric(train[k],errors='coerce').replace([np.inf,-np.inf],np.nan).notna().mean()) if k in train else 0. for k in FACTORS}
    eligible=[k for k in FACTORS if not k.startswith('r196_') and coverage[k]>=cfg.sparse_fit_coverage and (cfg.optional_data or k in TECH_SEARCH)]
    eligible=[k for k in eligible if pd.to_numeric(train[k],errors='coerce').nunique()>1]
    excluded={k:('OPTIONAL_DISABLED' if not cfg.optional_data and k not in TECH_SEARCH else
                 'INSUFFICIENT_TRAINING_COVERAGE_OR_CONSTANT') for k in FACTORS if k not in eligible}
    info.update(features=eligible,feature_coverage=coverage,excluded=excluded,train_dates=len(train_pairs),
                train_first=str(train_pairs[0][0].date()),train_last=str(train_pairs[-1][0].date()),
                train_max_exit=str(max(g[f'exit_date_{h}'].max() for d,g in train_pairs).date()),
                inner_validation_first=str(valid_pairs[0][0].date()),inner_validation_last=str(valid_pairs[-1][0].date()),
                validation_dates=len(valid_pairs),validation_cutoffs=[str(d.date()) for d,g in valid_pairs])
    if not eligible:return base,dict(info,status='SPARSE_NO_ELIGIBLE_FACTORS'),[]
    weights,traces,search=sparse_search(train_pairs,panel,eligible,h,cfg)
    info['sparse_search']=search
    if weights is None:return base,dict(info,status='SPARSE_SEARCH_INSUFFICIENT',search_reason=search['reason']),traces
    info.update(challenger_weights=weights,challenger_name='sparse_'+str(search['active_inputs']),
                selected_factors=search['selected_factors'],removed_factors=search['removed_factors'])
    prepared=Prepared(valid_pairs,panel,FACTORS,h,cfg)
    control=prepared.evaluate(base,strict=False);candidate=prepared.evaluate(weights)
    mask=prepared.fixed_reference_mask(control)
    comparable=mask&np.isfinite(candidate['net'])&np.isfinite(candidate['mae'])
    comparable &= (candidate['basis']==prepared.bm_basis)|(candidate['basis']=='CASH')
    n=int(comparable.sum());info['validation_common_blocks']=n
    info['validation_missing_or_incomparable']=[str(prepared.dates[j].date()) for j in range(len(mask)) if not comparable[j]]
    if n<max(cfg.validation_blocks,int(np.ceil(len(mask)*.8))):
        return base,dict(info,status='SPARSE_VALIDATION_COHORT_INCOMPLETE'),traces
    gate=stability_gate(candidate['net'][comparable],control['net'][comparable],prepared.bm[comparable],
                        candidate['mae'][comparable],control['mae'][comparable],candidate['fill'][comparable],cfg)
    gate['tests']['beats_same_pool_equal_weight']=bool((candidate['net'][comparable]-prepared.pool[comparable]).mean()>0)
    gate['accepted']=all(gate['tests'].values());accepted=gate['accepted']
    adopted=weights if accepted else base;out=candidate if accepted else control
    y=out['net'][comparable];bm=prepared.bm[comparable];pool=prepared.pool[comparable]
    edge=bool(y.mean()>0 and (y-bm).mean()>0 and (y-pool).mean()>0 and
              ((y-bm)>0).mean()>=.6 and out['fill'][comparable].mean()>=cfg.min_validation_fill)
    proxy=cfg.optional_data and cfg.financial_timing=='lagged_research' and bool(input_support(k for k,v in weights.items() if v>0).intersection(set(BUSINESS)|{'revenue','quality'}))
    if proxy:edge=False
    info.update(status='SPARSE_INNER_VALIDATION_ACCEPTED' if accepted else 'SPARSE_CANDIDATE_NOT_ADOPTED',
                stability_gate=gate,adopted_is_legacy_control=not accepted,
                adopted_support=[k for k,v in adopted.items() if v>0],allocation_allowed=edge,
                allocation_reason='ASSUMED_FINANCIAL_TIMING_RESEARCH_ONLY' if proxy else 'PAST_BLOCK_EDGE_RESEARCH_ONLY' if edge else 'NO_STABLE_BENCHMARK_EDGE',
                validation_mean_net=float(y.mean()),validation_mean_excess=float((y-bm).mean()),
                validation_mean_pool_excess=float((y-pool).mean()),validation_benchmark_win_fraction=float(((y-bm)>0).mean()))
    for label,result,w in [('legacy_control',control,base),('sparse_candidate',candidate,weights)]:
        traces.append({'stage':'inner_validation','candidate_name':label,'weights':w,
            'selected_factors':[k for k,v in w.items() if v>0],
            'cutoffs':[str(prepared.dates[j].date()) for j in np.flatnonzero(comparable)],
            'net_returns':result['net'][comparable].tolist(),'benchmark_returns':bm.tolist(),
            'pool_returns':pool.tolist(),'common_dates':n})
    return adopted,info,traces
