"""V19.6 simple-control tournament with separate prequential allocation evidence.

Policy weights are chosen on completed PAST blocks, not the outer case. A second
layer tests the algorithm's *then-issued* choices at earlier disjoint dates.
It never treats selection on a block as a new independent validation of itself.
All outcomes, no-fills, corporate-action coordinates and costs use the V19 engine.
This limited library is a development hypothesis, not an exhaustive optimum.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from v19_config import Settings,FACTORS,TECH,prior
from v194_selection import Prepared,KNOWN_BASIS
from v191_selection import stability_gate
from v196_factors import scorecard_weights,OPTIONAL_PARENTS

class ScorecardPrepared(Prepared):
    def evaluate(self,weights,strict=True):
        # A normalized scorecard must not secretly apply another legacy factor.
        saved=self.penalty
        if any(str(k).startswith('r196_') and v>0 for k,v in weights.items()):
            self.penalty=np.zeros_like(saved)
        try:return super().evaluate(weights,strict=strict)
        finally:self.penalty=saved

class Tournament:
    def __init__(self,panel,h,cfg,context=None):
        self.h=h;self.cfg=cfg;self.profiles=scorecard_weights(h,cfg)
        pairs=context.groups if context is not None else [(d,g) for d,g in panel[panel.eligible].groupby('date',sort=True) if len(g)>=cfg.min_universe]
        self.pairs=pairs
        self.dates=[d for d,g in pairs];self.ends=[g[f'exit_date_{h}'].max() for d,g in pairs]
        self.prepared=ScorecardPrepared(pairs,panel,FACTORS,h,cfg)
        self.outcomes={k:self.prepared.evaluate(w,strict=k!='legacy_control') for k,w in self.profiles.items()}
        self.cache={}

    def blocks_before(self,cutoff,limit):
        cutoff=pd.Timestamp(cutoff);ix=[];right=cutoff
        for i in range(len(self.dates)-1,-1,-1):
            if pd.notna(self.ends[i]) and self.ends[i]<right and self.dates[i]<cutoff:
                ix.append(i);right=self.dates[i]
                if len(ix)>=limit:break
        return list(reversed(ix))

    def select(self,cutoff):
        cutoff=pd.Timestamp(cutoff)
        if cutoff in self.cache:return self.cache[cutoff]
        ix=self.blocks_before(cutoff,self.cfg.tournament_blocks);base=self.outcomes['legacy_control']
        traces=[];choice='legacy_control';best=0.
        if len(ix)>=self.cfg.validation_blocks:
            by=base['net'][ix];bb=base['basis'][ix]
            # No replacement date is chosen because a contender has missing data.
            basegood=np.isfinite(by)&np.isin(bb,KNOWN_BASIS+('CASH',))
            for name,w in self.profiles.items():
                if name=='legacy_control':continue
                o=self.outcomes[name];a=o['net'][ix];basis=o['basis'][ix]
                good=basegood&np.isfinite(a)&np.isin(basis,KNOWN_BASIS+('CASH',))&((basis==bb)|(basis=='CASH')|(bb=='CASH'))
                valid=bool(good.all());score=None;mean=None;reason='INCOMPLETE_FIXED_PAST_COHORT'
                if valid:
                    delta=a-by;mean=float(delta.mean());uncertainty=float(delta.std(ddof=1)/np.sqrt(len(delta)))
                    # This is a regularization preference, NOT a confidence bound.
                    score=mean-self.cfg.tournament_uncertainty*uncertainty
                    enough=float((delta>0).mean())>=.6 and score>=self.cfg.improvement_floor
                    reason='ELIGIBLE_RESEARCH_POLICY' if enough else 'PAST_ADVANTAGE_NOT_ROBUST'
                    if enough and score>best+1e-12:best=score;choice=name
                traces.append(dict(stage='policy_selection_past_blocks',candidate_name=name,weights=w,
                    valid=valid,objective=score,mean_gain=mean,reason=reason,
                    block_dates=[str(self.dates[j].date()) for j in ix],
                    block_exits=[str(self.ends[j].date()) for j in ix]))
        info={'selected_profile':choice,'selection_blocks':len(ix),
            'selection_cutoffs':[str(self.dates[j].date()) for j in ix],
            'selection_max_exit':str(max(self.ends[j] for j in ix).date()) if ix else None,
            'selection_score':best,'selection_used_outer_future':False,
            'profiles_tested':len(self.profiles)-1,'model_complexity':len(self.profiles[choice]),
            'not_confidence_bound':True}
        result=(choice,info,traces);self.cache[cutoff]=result;return result

    def fit(self,cutoff):
        cutoff=pd.Timestamp(cutoff);cfg=self.cfg
        choice,selection,traces=self.select(cutoff);candidate=self.profiles[choice];base=self.profiles['legacy_control']
        info=dict(selection,cutoff=str(cutoff.date()),horizon=self.h,
            status='SCORECARD_RESEARCH_ONLY',factor_design='tournament_v196',
            challenger_weights=candidate,challenger_name=choice,control_weights=base,
            selected_factors=list(candidate),removed_factors=[k for k in FACTORS if k not in candidate],
            allocation_allowed=False,allocation_reason='INSUFFICIENT_PREQUENTIAL_EVIDENCE',
            adopted_is_legacy_control=True,adopted_support=list(base),
            holdout_used_for_selection=False,not_global_optimum=True,
            evaluation_role=cfg.evaluation_role,weights_are_portfolio_allocations=False)
        past=self.blocks_before(cutoff,cfg.validation_blocks)
        y=[];mae=[];fill=[];basis=[];issued=[]
        for i in past:
            name,meta,_=self.select(self.dates[i]);o=self.outcomes[name]
            y.append(o['net'][i]);mae.append(o['mae'][i]);fill.append(o['fill'][i]);basis.append(o['basis'][i])
            issued.append({'cutoff':str(self.dates[i].date()),'profile':name,
                           'selection_max_exit':meta['selection_max_exit']})
        info['prequential_choices']=issued;info['validation_dates']=len(past)
        info['train_dates']=selection['selection_blocks']
        if not past:return base,info,traces
        panel_rows=pd.concat([self.pairs[i][1] for i in self.blocks_before(cutoff,cfg.tournament_blocks)],ignore_index=True)
        info['feature_coverage']={k:float(pd.to_numeric(panel_rows[k],errors='coerce').replace([np.inf,-np.inf],np.nan).notna().mean()) for k in FACTORS}
        info['excluded']={k:'NO_USABLE_DATA' for k,v in info['feature_coverage'].items() if v==0}
        b=self.outcomes['legacy_control'];bm=self.prepared.bm[past];pool=self.prepared.pool[past];bb=self.prepared.bm_basis[past]
        valid=np.isfinite(y)&np.isfinite(mae)&np.isfinite(b['net'][past])&np.isfinite(b['mae'][past])&np.isfinite(bm)&np.isfinite(pool)
        valid &= np.isin(bb,KNOWN_BASIS)
        for bs in [np.asarray(basis),b['basis'][past],self.prepared.pool_basis[past]]:
            valid &= (bs==bb)|(bs=='CASH')
        info['validation_common_blocks']=int(valid.sum())
        if len(past)<cfg.validation_blocks or not valid.all():
            info['status']='SCORECARD_PREQUENTIAL_INCOMPLETE';return base,info,traces
        gate=stability_gate(y,b['net'][past],bm,mae,b['mae'][past],fill,cfg)
        gate.setdefault('tests',{})['beats_same_pool_equal_weight']=bool((np.asarray(y)-pool).mean()>0)
        gate['accepted']=bool(gate.get('accepted') and all(gate['tests'].values()))
        accepted=gate['accepted']
        used=candidate if accepted else base
        inputs=set(candidate)
        assumed=cfg.financial_timing=='lagged_research' and bool(inputs.intersection({'r196_revenue','r196_earnings_growth','r196_earnings_yield'}))
        info.update(status='SCORECARD_PREQUENTIAL_ACCEPTED' if accepted else 'SCORECARD_RESEARCH_ONLY',
            stability_gate=gate,failed_adoption_checks=[k for k,v in gate.get('tests',{}).items() if not v],
            adopted_is_legacy_control=not accepted or choice=='legacy_control',adopted_support=list(used),
            allocation_allowed=bool(accepted and not assumed),
            allocation_reason='ASSUMED_FINANCIAL_TIMING_RESEARCH_ONLY' if assumed else 'PREQUENTIAL_EDGE_RESEARCH_ONLY' if accepted else 'NO_STABLE_PREQUENTIAL_EDGE',
            validation_mean_net=float(np.mean(y)),validation_mean_excess=float((np.asarray(y)-bm).mean()),
            validation_cutoffs=[str(self.dates[j].date()) for j in past])
        traces.append({'stage':'prequential_gate','issued_choices':issued,'net_returns':y,'benchmark_returns':bm.tolist(),
                       'policy_selection_and_policy_validation_are_distinct':True})
        return used,info,traces


def fit_before(panel,cutoff,h,cfg=Settings(),seed=1960,context=None):
    # Context is run-specific; never reuse by date alone across different datasets.
    if context is not None:
        if not hasattr(context,'tournaments196'):context.tournaments196={}
        if h not in context.tournaments196:context.tournaments196[h]=Tournament(panel,h,cfg,context)
        return context.tournaments196[h].fit(cutoff)
    return Tournament(panel,h,cfg).fit(cutoff)
