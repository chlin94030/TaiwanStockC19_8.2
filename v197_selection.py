"""Ten-perspective, pre-registered profile tournament with causal policy replay.

Historical profiles remain diagnostics. At cutoff T, a policy can use only cases
whose EXIT is strictly earlier than T. The final gate evaluates the past issued
policy choices, not the same blocks used to select a fixed profile. This is
still development research on a repeatedly examined, survivor-biased universe.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from v19_config import Settings, FACTORS, TECH, prior
from v194_selection import Prepared, KNOWN_BASIS
from v191_selection import stability_gate
from v197_ten_pillars import PILLARS, UNVERIFIED_SECTOR, REQUIRES_DISCLOSURE, NAMES


def profiles_for(h, cfg=Settings()):
    """Fixed hypotheses set BEFORE seeing outer-cutoff returns."""
    if cfg.factor_design=='balanced_v198':
        from v198_selection import profiles_for_v198
        return profiles_for_v198(h,cfg)
    p={
      'legacy_control':prior(h,TECH),
      'high_52':{'p197_annual_high':1.},
      'liquid_capacity':{'p197_liquidity':1.},
      'trend_eff':{'p197_trend_eff':1.},
      'relative_60':{'p197_relative':1.},
      'lower_tail':{'p197_downside':1.},
      'compression':{'p197_compression':1.},
      'high_liquid':{'p197_annual_high':.5,'p197_liquidity':.5},
      'strong_quality_trend':{'p197_annual_high':.4,'p197_trend_eff':.35,'p197_liquidity':.25},
      'risk_balanced':{'p197_annual_high':.4,'p197_downside':.4,'p197_liquidity':.2},
      'squeeze_breakout':{'p197_compression':.45,'p197_annual_high':.35,'p197_liquidity':.2},
      'long_trend_defense':{'p197_relative':.35,'p197_trend_eff':.3,'p197_downside':.35},
      'sector_leadership_RESEARCH_ONLY':{'p197_annual_high':.5,'p197_sector':.5},
    }
    if cfg.optional_data:
        p.update({
          'foreign_flow':{'p197_foreign':1.},
          'trust_flow':{'p197_trust':1.},
          'institutional_leaders':{'p197_annual_high':.45,'p197_foreign':.3,'p197_trust':.25},
          'revenue_growth':{'p197_revenue':1.},
          'value_quality':{'p197_quality':.5,'p197_valuation':.5},
          'profitable_leaders':{'p197_quality':.4,'p197_annual_high':.4,'p197_liquidity':.2},
        })
    return p


def _safe_float_array(a):
    return np.asarray(a,float)


class TenPrepared(Prepared):
    """No hidden legacy bias penalty on new scorecards."""
    def evaluate(self, weights, strict=True):
        if not any(k.startswith('p197_') and w>0 for k,w in weights.items()):
            return super().evaluate(weights,strict=strict)
        previous=self.penalty;self.penalty=np.zeros_like(previous)
        try:return super().evaluate(weights,strict=strict)
        finally:self.penalty=previous


class TenTournament:
    def __init__(self, panel, h, cfg, context=None):
        self.h=h;self.cfg=cfg;self.profiles=profiles_for(h,cfg)
        self.pairs=context.groups if context is not None else [(d,g) for d,g in panel.loc[panel.eligible].groupby('date',sort=True) if len(g)>=cfg.min_universe]
        self.dates=[d for d,g in self.pairs]
        self.ends=[g[f'exit_date_{h}'].max() for d,g in self.pairs]
        # Allocate only used scoring dimensions. A 600-2000 equity universe
        # must not allocate all 50+ research factors on every daily group.
        active=tuple(k for k in FACTORS if any(w.get(k,0)>0 for w in self.profiles.values()))
        self.prepared=TenPrepared(self.pairs,panel,active,h,cfg)
        self.outputs={k:self.prepared.evaluate(w,strict=k!='legacy_control') for k,w in self.profiles.items()}
        self.selection_cache={}

    def past_blocks(self,cutoff,limit):
        cutoff=pd.Timestamp(cutoff);ix=[];right=cutoff
        for i in range(len(self.dates)-1,-1,-1):
            if pd.notna(self.ends[i]) and self.ends[i]<right and self.dates[i]<cutoff:
                ix.append(i);right=self.dates[i]
                if len(ix)>=limit:break
        return list(reversed(ix))

    def select(self,cutoff):
        cutoff=pd.Timestamp(cutoff)
        if cutoff in self.selection_cache:return self.selection_cache[cutoff]
        ix=self.past_blocks(cutoff,self.cfg.tournament_blocks)
        selected='legacy_control';top=0.;evidence=[]
        base=self.outputs['legacy_control']
        if len(ix)>=self.cfg.validation_blocks:
            b=base['net'][ix];bb=base['basis'][ix]
            benchmark=self.prepared.bm[ix];bk=self.prepared.bm_basis[ix]
            for name,weights in self.profiles.items():
                if name=='legacy_control':continue
                a=self.outputs[name]['net'][ix];basis=self.outputs[name]['basis'][ix]
                supported=not bool(set(weights)&UNVERIFIED_SECTOR)
                # A financial value explicitly assumed in a non-PIT research scenario
                # is never authorized as a production signal.
                assumed=bool(set(weights)&REQUIRES_DISCLOSURE) and self.cfg.financial_timing!='disclosure_required'
                valid=(supported and not assumed and np.isfinite(a).all() and np.isfinite(b).all()
                       and np.isfinite(benchmark).all() and np.isin(bk,KNOWN_BASIS).all()
                       and ((basis==bb)|(basis=='CASH')|(bb=='CASH')).all()
                       and ((basis==bk)|(basis=='CASH')).all())
                delta=a-b;mean=float(np.mean(delta)) if valid else None
                uncertainty=float(np.std(delta,ddof=1)/np.sqrt(len(ix))) if valid and len(ix)>1 else 0.
                objective=mean-self.cfg.tournament_uncertainty*uncertainty if valid else None
                beats_benchmark=bool(valid and np.mean(a-benchmark)>0)
                wins=bool(valid and np.mean(delta>0)>=.6)
                # A profile must first beat the simple independent legacy and 0050,
                # rather than merely win the retrospective sample by one outlier.
                eligible=bool(valid and wins and beats_benchmark and objective>=self.cfg.improvement_floor)
                reason='HISTORICAL_SECTOR_MEMBERSHIP_UNVERIFIED' if not supported else ('ASSUMED_RELEASE_TIME' if assumed else 'VALID' if valid else 'INCOMPLETE_PAST_BLOCKS')
                evidence.append({'stage':'fixed_profile_training','candidate_name':name,
                    'weights':weights,'valid':bool(valid),'mean_gain':mean,
                    'objective':objective,'beats_0050_past':beats_benchmark,
                    'eligible':eligible,'reason':reason,
                    'block_dates':[str(self.dates[j].date()) for j in ix]})
                if eligible and objective>top+1e-12:selected=name;top=objective
        info={'selected_profile':selected,'selection_blocks':len(ix),
              'selection_cutoffs':[str(self.dates[j].date()) for j in ix],
              'max_selection_exit':str(max(self.ends[j] for j in ix).date()) if ix else None,
              'objective':top,'selected_before_future_outcome':True}
        result=(selected,info,evidence)
        self.selection_cache[cutoff]=result
        return result

    def fit(self,cutoff):
        cutoff=pd.Timestamp(cutoff)
        profile,selection,trace=self.select(cutoff)
        base=self.profiles['legacy_control'];challenge=self.profiles[profile]
        info=dict(selection,cutoff=str(cutoff.date()),horizon=self.h,
                  factor_design='ten_angle_v197',status='TEN_PILLAR_RESEARCH_ONLY',
                  challenger_name=profile,challenger_weights=challenge,
                  control_weights=base,selected_factors=list(challenge),
                  adopted_is_legacy_control=True,adopted_support=list(base),
                  allocation_allowed=False,allocation_reason='NO_APPROVED_PREQUENTIAL_EVIDENCE',
                  policy_profiles=len(self.profiles),
                  historical_sector_ledger_verified=False,
                  benchmark_price_history_verified=False,
                  rankings_are_probabilities=False)
        validation=self.past_blocks(cutoff,self.cfg.validation_blocks)
        y=[];mae=[];fill=[];basis=[];issued=[]
        for i in validation:
            old,meta,_=self.select(self.dates[i]);o=self.outputs[old]
            y.append(o['net'][i]);mae.append(o['mae'][i]);fill.append(o['fill'][i]);basis.append(o['basis'][i])
            issued.append({'date':str(self.dates[i].date()),'profile':old,'past_selection_exit':meta['max_selection_exit']})
        info['issued_past_policy']=issued;info['validation_blocks']=len(validation)
        if not validation:return base,info,trace
        by=self.outputs['legacy_control'];a=np.asarray(y,float);ma=np.asarray(mae,float)
        bm=self.prepared.bm[validation];pool=self.prepared.pool[validation]
        known=np.isfinite(a)&np.isfinite(ma)&np.isfinite(by['net'][validation])&np.isfinite(by['mae'][validation])&np.isfinite(bm)&np.isfinite(pool)
        basisarr=np.asarray(basis)
        compare_basis=self.prepared.bm_basis[validation]
        known &= np.isin(compare_basis,KNOWN_BASIS) & ((basisarr==compare_basis)|(basisarr=='CASH'))
        known &= ((by['basis'][validation]==compare_basis)|(by['basis'][validation]=='CASH'))
        info['validation_common_blocks']=int(known.sum())
        if len(validation)<self.cfg.validation_blocks or not known.all():
            info['status']='TEN_PILLAR_VALIDATION_INCOMPLETE';return base,info,trace
        gate=stability_gate(a,by['net'][validation],bm,ma,by['mae'][validation],fill,self.cfg)
        gate.setdefault('tests',{})['beats_same_pool']=bool(np.mean(a-pool)>0)
        gate['tests']['not_original_control']=bool(profile!='legacy_control')
        gate['accepted']=bool(gate.get('accepted') and all(gate['tests'].values()))
        approved=bool(gate['accepted'])
        info.update(stability_gate=gate,validation_mean_net=float(a.mean()),
                    validation_mean_excess=float(np.mean(a-bm)),
                    failed_checks=[k for k,v in gate['tests'].items() if not v],
                    validation_dates=[str(self.dates[i].date()) for i in validation],
                    status='TEN_PILLAR_INTERNAL_GATE_PASSED' if approved else 'TEN_PILLAR_RESEARCH_ONLY',
                    allocation_allowed=approved,
                    allocation_reason='INTERNAL_PAST_BLOCK_GATE_ONLY_NOT_LIVE_CERTIFICATION' if approved else 'NO_STABLE_PAST_BLOCK_EDGE',
                    adopted_is_legacy_control=not approved,
                    adopted_support=list(challenge if approved else base))
        trace.append({'stage':'prequential_policy_gate','issued_choices':issued,'benchmark_returns':bm.tolist(),
                      'chosen_returns':a.tolist(),'not_independent_new_holdout':True})
        return challenge if approved else base, info, trace


def fit_before(panel,cutoff,h,cfg=Settings(),seed=1970,context=None):
    if context is not None:
        if not hasattr(context,'ten_tournaments197'):context.ten_tournaments197={}
        if h not in context.ten_tournaments197:
            context.ten_tournaments197[h]=TenTournament(panel,h,cfg,context)
        return context.ten_tournaments197[h].fit(cutoff)
    return TenTournament(panel,h,cfg).fit(cutoff)


def ten_summary(report):
    """Fixed profile vs 0050 on SAME valid cutoff+price coordinate."""
    records=pd.DataFrame(report.get('results',[]));out=[]
    if records.empty:return out
    b=records[records.strategy=='benchmark_0050'][['date','horizon','mean_net','price_basis','complete']]
    base=records[records.strategy=='baseline'][['date','horizon','mean_net','price_basis','complete']]
    shadow=records[records.strategy.str.startswith('ten_shadow_',na=False)]
    for (h,profile),a in shadow.groupby(['horizon','strategy'],sort=True):
        q=a.merge(b,on=['date','horizon'],suffixes=('','_benchmark')).merge(base,on=['date','horizon'],suffixes=('','_baseline'))
        compatible=((q.price_basis==q.price_basis_benchmark) & (q.price_basis==q.price_basis_baseline))
        valid=q.complete & q.complete_benchmark & q.complete_baseline & q[['mean_net','mean_net_benchmark','mean_net_baseline']].notna().all(axis=1) & compatible
        q=q.loc[valid]
        if q.empty:continue
        out.append({'horizon':int(h),'profile':profile.removeprefix('ten_shadow_'),
                    'paired_cases':len(q),'mean_net':float(q.mean_net.mean()),
                    'legacy_mean':float(q.mean_net_baseline.mean()),
                    'benchmark_mean':float(q.mean_net_benchmark.mean()),
                    'vs_legacy':float((q.mean_net-q.mean_net_baseline).mean()),
                    'vs_0050':float((q.mean_net-q.mean_net_benchmark).mean()),
                    'win_vs_0050':float((q.mean_net>q.mean_net_benchmark).mean()),
                    'not_independent_test':True})
    return out
