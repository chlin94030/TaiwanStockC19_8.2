"""V19.5: support-local training, disjoint fold windows and bounded sparse search.

The old engine remains in v194_selection.py. No future test result tunes these
rules. All coefficients, missing-data masks, correlations and support choices
are learned inside dated training folds. Equal weights are an explicit competing
fit method, not a silently imposed prior. This is research, never order routing.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd
from v19_config import FACTORS, TECH, Settings, prior
from v194_factors import ATOMIC, BUSINESS, INTERACTIONS, input_support
from v194_selection import Prepared, _solve, _full_weights, TECH_SEARCH, KNOWN_BASIS
from v191_selection import stability_gate


def compatible(keys):
    """Include semantic parents in overlap checks, not just visible key names."""
    selected = set(keys)
    parents = input_support(selected)
    if 'reversal' in parents and parents.intersection(ATOMIC):
        return False
    if 'quality' in parents and parents.intersection(('earnings_growth','gross_profit_quality')):
        return False
    # Two separate scores cannot both use the same underlying input.
    groups = [set(INTERACTIONS.get(k, (k,))) for k in selected]
    return all(not groups[i].intersection(groups[j])
               for i in range(len(groups)) for j in range(i+1, len(groups)))


def _exit(group, h):
    return pd.to_datetime(group[f'exit_date_{h}'], errors='coerce').max()


def inner_folds(pairs, h, cfg):
    """Globally disjoint test windows, including adjacent fold boundaries."""
    if not pairs:
        return []
    minimum = max(12, cfg.train_dates//2)
    # Leave an initial chronological fit section; require fully matured labels.
    start = max(minimum, len(pairs)//2)
    scoring=[]; last_exit=None
    for i in range(start, len(pairs)):
        d,g=pairs[i];end=_exit(g,h)
        if pd.isna(end) or (last_exit is not None and d<=last_exit):
            continue
        fit_count=sum(_exit(q,h)<d for _,q in pairs[:i])
        if fit_count<minimum:
            continue
        scoring.append((d,g));last_exit=end
    if len(scoring)<2:
        return []
    out=[]
    for ids in np.array_split(np.arange(len(scoring)), min(cfg.sparse_cv_folds,len(scoring))):
        test=[scoring[int(i)] for i in ids]
        first=test[0][0]
        fit=[(d,g) for d,g in pairs if d<first and _exit(g,h)<first]
        if len(fit)>=minimum:
            out.append((fit,test))
    return out


def reserve_validation(grouped, h, cfg):
    """Reserve a fixed, prespecified number; do not consume six blocks by default."""
    choices=[]; next_start=None
    for d,g in reversed(grouped):
        end=_exit(g,h)
        if pd.isna(end):continue
        if next_start is None or end<next_start:
            choices.append((d,g));next_start=d
        if len(choices)==cfg.validation_blocks:break
    if len(choices)<cfg.validation_blocks:return [],[]
    valid=list(reversed(choices));start=valid[0][0];train=[]
    for d,g in grouped:
        if d>=start:break
        q=g[g[f'exit_date_{h}']<start]
        if len(q)>=cfg.min_universe:train.append((d,q))
    window=max(252,8*h+cfg.train_dates)
    if len(train)<cfg.train_dates:return [],[]
    return train[-window:],valid


class TrainingMoments:
    """Cache date-centered moments by each support's complete-row mask.

    Unselected missing columns are numerically zeroed ONLY while computing
    unused matrix entries. A selected missing column always excludes the row;
    a matrix is reused only when that complete-input mask is exactly identical.
    Date weighting is equal. Complete outcomes with unknown price basis are not
    allowed to become training evidence. These masks affect fitting only, never
    which stocks are eligible in the later portfolio evaluation.
    """
    def __init__(self,pairs,features,h):
        self.keys=tuple(features);self.index={k:i for i,k in enumerate(features)}
        self.pairs=pairs;self.h=h
        df=pd.concat([g.assign(_fit_date_id=i) for i,(_,g) in enumerate(pairs)],ignore_index=True)
        self.x=df.reindex(columns=self.keys).to_numpy(float)/100.
        self.y=pd.to_numeric(df[f'net_{h}'],errors='coerce').to_numpy(float)
        self.date_id=df['_fit_date_id'].to_numpy(int)
        self.base=np.isfinite(self.y)
        if f'return_basis_{h}' in df:
            self.base &= df[f'return_basis_{h}'].isin(KNOWN_BASIS+('CASH',)).to_numpy()
        self.finite=np.isfinite(self.x);self.cache={}

    def stats(self,subset):
        ids=[self.index[k] for k in subset]
        good=self.base & self.finite[:,ids].all(axis=1)
        counts=np.bincount(self.date_id,weights=good,minlength=len(self.pairs))
        good &= counts[self.date_id]>=8
        signature=np.packbits(good).tobytes()
        if signature not in self.cache:
            nd=len(self.pairs);p=len(self.keys)
            dates=self.date_id[good];xx=np.where(self.finite[good],self.x[good],0.)
            yy=pd.Series(self.y[good]).groupby(dates,sort=False).rank(pct=True).to_numpy()
            counts=np.bincount(dates,minlength=nd).astype(float);valid=counts>0
            n_dates=int(valid.sum());n_rows=int(good.sum())
            if not n_dates:
                gram=np.zeros((p,p));rhs=np.zeros(p)
            else:
                sx=np.zeros((nd,p));sy=np.zeros(nd)
                np.add.at(sx,dates,xx);np.add.at(sy,dates,yy)
                divisor=np.maximum(counts,1.)
                xx=xx-sx[dates]/divisor[dates,None]
                yy=yy-sy[dates]/divisor[dates]
                weight=1./(divisor[dates]*n_dates)
                gram=xx.T@(xx*weight[:,None]);rhs=xx.T@(yy*weight)
            self.cache[signature]=(gram,rhs,n_dates,n_rows)
        gram,rhs,dates,rows=self.cache[signature]
        gram=gram[np.ix_(ids,ids)];rhs=rhs[ids]
        diag=np.sqrt(np.maximum(np.diag(gram),0.));den=diag[:,None]*diag[None,:]
        corr=np.divide(gram,den,out=np.zeros_like(gram),where=den>1e-12)
        coverage=float((self.finite[:,ids].all(axis=1)).mean())
        return dict(gram=gram,rhs=rhs,corr=corr,dates=dates,rows=rows,coverage=coverage)


def _fit_subset(stats,subset,method,cfg):
    if stats['dates']<10 or stats['coverage']<cfg.sparse_fit_coverage:return None
    if len(subset)>1:
        c=stats['corr'].copy();np.fill_diagonal(c,0)
        if np.max(np.abs(c))>cfg.sparse_corr_limit:return None
    # An everywhere-constant score is not a predictive model; don't rank by ticker.
    if np.max(np.diag(stats['gram']),initial=0.)<1e-10:return None
    if method=='equal':return {k:1./len(subset) for k in subset}
    return _solve(stats['gram'],stats['rhs'],subset,cfg.sparse_ridge,cfg.weight_shrinkage)


def sparse_search(train_pairs,panel,features,h,cfg):
    splits=inner_folds(train_pairs,h,cfg)
    if len(splits)<2:return None,[],{'reason':'INSUFFICIENT_PURGED_SEARCH_FOLDS','folds':len(splits)}
    baseline=prior(h,TECH);folds=[];ranges=[]
    for fit,test in splits:
        prepared=Prepared(test,panel,FACTORS,h,cfg)
        mask=prepared.fixed_reference_mask(prepared.evaluate(baseline,strict=False))
        if mask.sum()<max(1,int(np.ceil(len(test)*.8))):continue
        folds.append((TrainingMoments(fit,features,h),prepared,mask))
        ranges.append({'fit_first':str(fit[0][0].date()),'fit_last':str(fit[-1][0].date()),
            'fit_max_exit':str(max(_exit(g,h) for _,g in fit).date()),
            'score_first':str(test[0][0].date()),'score_last':str(test[-1][0].date()),
            'score_last_exit':str(max(_exit(g,h) for _,g in test).date()),
            'score_cutoffs':[str(d.date()) for d,_ in test]})
    if len(folds)<2:return None,[],{'reason':'INSUFFICIENT_SAME_COHORT_SEARCH_FOLDS','folds':len(folds)}
    considered={};history=[];full=TrainingMoments(train_pairs,features,h)

    def evaluate(subset,method):
        subset=tuple(sorted(subset));key=(subset,method)
        if key in considered:return considered[key]
        if len(considered)>=cfg.sparse_budget or not subset:return None
        if not compatible(subset) or len(input_support(subset))>cfg.sparse_max_inputs:return None
        scores=[];weights=[];cohorts=[];reason=None
        for moment,prepared,mask in folds:
            stats=moment.stats(subset);w=_fit_subset(stats,subset,method,cfg)
            cohorts.append({'dates':stats['dates'],'rows':stats['rows'],'coverage':stats['coverage']})
            if w is None:reason='TRAINING_INPUTS_CONSTANT_CORRELATED_OR_INSUFFICIENT';break
            result=prepared.score(prepared.evaluate(w),mask,w,cfg)
            if result is None:reason='FIXED_SCORE_COHORT_INCOMPLETE';break
            scores.append(result);weights.append(w)
        valid=len(scores)==len(folds)
        objective=float(np.mean([r['objective'] for r in scores])) if valid else None
        record={'stage':'sparse_search_cv','candidate':len(considered),'candidate_name':method+':'+ '+'.join(subset),
            'weight_method':method,'requested_subset':list(subset),'active_inputs':len(input_support(subset)),
            'objective':objective,'folds':len(scores),'required_folds':len(folds),'fold_scores':scores,
            'fold_weights':weights,'training_cohorts':cohorts,'valid':valid,'invalid_reason':reason}
        considered[key]=record;history.append(record);return record

    def visit(subset):
        evaluate(subset,'equal')
        if len(subset)>1:evaluate(subset,'ridge')

    def ranked():
        return sorted([r for r in history if r['valid']],key=lambda r:(-r['objective'],r['active_inputs'],r['candidate_name']))

    for k in features:visit((k,))
    singles=ranked()
    if not singles:return None,history,{'reason':'NO_COMPLETE_SEARCH_CANDIDATE','folds':len(folds)}
    shortlist=[r['requested_subset'][0] for r in singles][:12]
    for pair in combinations(shortlist[:6],2):visit(pair)
    # Exact-depth frontiers prevent a top singleton from blocking every addition.
    for depth in range(2,cfg.sparse_max_inputs+1):
        leaders=[r for r in ranked() if len(r['requested_subset'])==depth-1]
        seen=set();front=[]
        for r in leaders:
            k=tuple(r['requested_subset'])
            if k not in seen:seen.add(k);front.append(r)
            if len(front)>=cfg.sparse_beam:break
        for leader in front:
            for k in shortlist:
                if k not in leader['requested_subset']:visit(tuple(leader['requested_subset'])+(k,))
        if len(considered)>=cfg.sparse_budget:break
    for leader in ranked()[:cfg.sparse_beam]:
        for old in leader['requested_subset']:
            smaller=tuple(k for k in leader['requested_subset'] if k!=old)
            if smaller:visit(smaller)
            for new in shortlist[:8]:
                if new not in leader['requested_subset']:visit(smaller+(new,))
    candidates=ranked();best=candidates[0]['objective']
    near=[r for r in candidates if r['objective']>=best-cfg.sparse_near_best]
    choice=min(near,key=lambda r:(r['active_inputs'],-r['objective'],r['candidate_name']))
    subset=choice['requested_subset'];stats=full.stats(subset)
    w=_fit_subset(stats,subset,choice['weight_method'],cfg)
    if w is None:return None,history,{'reason':'SELECTED_METHOD_REFIT_FAILED_NO_RESELECTION','folds':len(folds)}
    chosen=_full_weights(w);support=[k for k,v in chosen.items() if v>0]
    summary={'reason':'SEARCH_FINISHED','search_candidates':len(history),'budget':cfg.sparse_budget,
        'search_exhaustive':False,'cv_objective':choice['objective'],'cv_best_objective':best,
        'folds':len(folds),'refit_dates':stats['dates'],'refit_rows':stats['rows'],
        'selected_factors':support,'active_inputs':sorted(input_support(support)),
        'requested_subset':subset,'removed_factors':[k for k in FACTORS if k not in support],
        'fold_ranges':ranges,'weight_fit':choice['weight_method'],
        'coefficient_cohort':'SELECTED_SUPPORT_ONLY_COMPLETE_INPUTS_NO_UNUSED_FEATURE_DROPS',
        'test_windows_globally_disjoint':True,'all_coefficients_past_only':True,
        'scoring_missing_policy':'ALL_SELECTED_INPUTS_REQUIRED_NO_IMPUTATION',
        'near_best_preference':'FEWER_UNDERLYING_INPUTS',
        'shrinkage':'EQUAL_ONLY_WITHIN_POSITIVE_RIDGE_SUPPORT',
        'budget_counts':'subset_and_fit_method_candidates',
        'selected_candidate_name':choice['candidate_name']}
    return chosen,history,summary


def sparse_fit(panel,cutoff,h,cfg=Settings(),seed=1900,context=None):
    cutoff=pd.Timestamp(cutoff);base=_full_weights(prior(h,TECH))
    info={'cutoff':str(cutoff.date()),'horizon':h,'status':'SPARSE_INSUFFICIENT_PAST_BLOCKS',
          'features':list(TECH),'train_dates':0,'validation_dates':0,'allocation_allowed':False,
          'allocation_reason':'INSUFFICIENT_COMPARABLE_BLOCKS','factor_design':'sparse_v195',
          'holdout_used_for_selection':False,'evaluation_role':cfg.evaluation_role,
          'not_global_optimum':True,'adopted_support':[k for k,v in base.items() if v>0],
          'control_weights':base,'adopted_is_legacy_control':True,
          'reserved_final_blocks':cfg.validation_blocks,'development_window':max(252,8*h+cfg.train_dates)}
    if context is not None:grouped=context.before(cutoff,h)
    else:
        past=panel[(panel.date<cutoff)&(panel[f'exit_date_{h}']<cutoff)&panel.eligible]
        grouped=[(d,g) for d,g in past.groupby('date',sort=True) if len(g)>=cfg.min_universe]
    train_pairs,valid_pairs=reserve_validation(grouped,h,cfg)
    if not train_pairs:return base,info,[]
    train=pd.concat([g for _,g in train_pairs],ignore_index=True)
    coverage={k:float(pd.to_numeric(train[k],errors='coerce').replace([np.inf,-np.inf],np.nan).notna().mean()) if k in train else 0. for k in FACTORS}
    eligible=[k for k in FACTORS if not k.startswith('r196_') and coverage[k]>=cfg.sparse_fit_coverage and (cfg.optional_data or k in TECH_SEARCH)]
    eligible=[k for k in eligible if pd.to_numeric(train[k],errors='coerce').nunique()>1]
    excluded={k:('OPTIONAL_DISABLED' if not cfg.optional_data and k not in TECH_SEARCH else 'INSUFFICIENT_TRAINING_COVERAGE_OR_CONSTANT') for k in FACTORS if k not in eligible}
    info.update(features=eligible,feature_coverage=coverage,excluded=excluded,train_dates=len(train_pairs),
        train_first=str(train_pairs[0][0].date()),train_last=str(train_pairs[-1][0].date()),
        train_max_exit=str(max(_exit(g,h) for _,g in train_pairs).date()),
        inner_validation_first=str(valid_pairs[0][0].date()),inner_validation_last=str(valid_pairs[-1][0].date()),
        validation_dates=len(valid_pairs),validation_cutoffs=[str(d.date()) for d,_ in valid_pairs])
    if not eligible:return base,dict(info,status='SPARSE_NO_ELIGIBLE_FACTORS'),[]
    weights,traces,search=sparse_search(train_pairs,panel,eligible,h,cfg)
    info['sparse_search']=search
    if weights is None:return base,dict(info,status='SPARSE_SEARCH_INSUFFICIENT',search_reason=search['reason']),traces
    info.update(challenger_weights=weights,challenger_name=search['selected_candidate_name'],
        selected_factors=search['selected_factors'],removed_factors=search['removed_factors'])
    prepared=Prepared(valid_pairs,panel,FACTORS,h,cfg)
    control=prepared.evaluate(base,strict=False);candidate=prepared.evaluate(weights)
    mask=prepared.fixed_reference_mask(control)
    comparable=mask&np.isfinite(candidate['net'])&np.isfinite(candidate['mae'])
    comparable &= (candidate['basis']==prepared.bm_basis)|(candidate['basis']=='CASH')
    n=int(comparable.sum());info['validation_common_blocks']=n
    info['validation_reference_blocks']=int(mask.sum())
    info['validation_missing_or_incomparable']=[str(prepared.dates[j].date()) for j in range(len(mask)) if not comparable[j]]
    if n<cfg.validation_blocks or not np.array_equal(comparable,mask):
        return base,dict(info,status='SPARSE_VALIDATION_COHORT_INCOMPLETE'),traces
    bm=prepared.bm[mask];pool=prepared.pool[mask]
    gate=stability_gate(candidate['net'][mask],control['net'][mask],bm,candidate['mae'][mask],control['mae'][mask],candidate['fill'][mask],cfg)
    gate.setdefault('tests',{})['beats_same_pool_equal_weight']=bool((candidate['net'][mask]-pool).mean()>0)
    gate['accepted']=bool(gate.get('accepted',False) and all(gate['tests'].values()))
    accepted=gate['accepted'];adopted=weights if accepted else base;out=candidate if accepted else control;y=out['net'][mask]
    edge=bool(y.mean()>0 and (y-bm).mean()>0 and (y-pool).mean()>0 and ((y-bm)>0).mean()>=.6 and out['fill'][mask].mean()>=cfg.min_validation_fill)
    inputs=input_support(k for k,v in adopted.items() if v>0)
    proxy=cfg.optional_data and cfg.financial_timing=='lagged_research' and bool(inputs.intersection(BUSINESS+('quality',)))
    if proxy:edge=False
    info.update(status='SPARSE_INNER_VALIDATION_ACCEPTED' if accepted else 'SPARSE_CANDIDATE_NOT_ADOPTED',
        stability_gate=gate,adopted_is_legacy_control=not accepted,
        adopted_support=[k for k,v in adopted.items() if v>0],allocation_allowed=edge,
        allocation_reason='ASSUMED_FINANCIAL_TIMING_RESEARCH_ONLY' if proxy else 'PAST_BLOCK_EDGE_RESEARCH_ONLY' if edge else 'NO_STABLE_BENCHMARK_EDGE',
        validation_mean_net=float(y.mean()),validation_mean_excess=float((y-bm).mean()),
        validation_mean_pool_excess=float((y-pool).mean()),validation_benchmark_win_fraction=float(((y-bm)>0).mean()),
        failed_adoption_checks=[k for k,v in gate.get('tests',{}).items() if not v])
    for label,result,w in [('legacy_control',control,base),('sparse_candidate',candidate,weights)]:
        traces.append({'stage':'inner_validation','candidate_name':label,'weights':w,
            'selected_factors':[k for k,v in w.items() if v>0],
            'cutoffs':[str(prepared.dates[j].date()) for j in np.flatnonzero(mask)],
            'net_returns':result['net'][mask].tolist(),'benchmark_returns':bm.tolist(),
            'pool_returns':pool.tolist(),'common_dates':n})
    return adopted,info,traces
