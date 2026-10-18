"""Array cache for repeated causal fits; outcome access is caller date-bounded."""
from __future__ import annotations
import numpy as np
import pandas as pd
from v19_config import FACTORS

class PortfolioCache:
    def __init__(self,groups,cfg):
        self.cfg=cfg;self.keys=list(FACTORS);self.dates={d:i for i,(d,g) in enumerate(groups)}
        n=len(groups);width=max((len(g) for d,g in groups),default=cfg.top_k)
        self.a=np.full((n,width,len(self.keys)),np.nan);self.penalty=np.zeros((n,width))
        self.y={h:np.full((n,width),np.nan) for h in cfg.horizons}
        self.down={h:np.full((n,width),np.nan) for h in cfg.horizons}
        self.fills={h:np.zeros((n,width)) for h in cfg.horizons}
        self.basis={h:np.zeros((n,width),np.int8) for h in cfg.horizons}
        self.codes={'UNKNOWN':0,'RAW_PRICE_ONLY':1,'ADJUSTED_PROVIDER':2,'CASH':3,'MIXED_UNVERIFIED':4}
        self.labels=np.array(['UNKNOWN','RAW_PRICE_ONLY','ADJUSTED_PROVIDER','CASH','MIXED_UNVERIFIED'],object)
        for i,(d,g) in enumerate(groups):
            q=g.sort_values('ticker');m=len(q)
            self.a[i,:m]=q.reindex(columns=self.keys).to_numpy(float)
            self.penalty[i,:m]=np.clip(q.bias20.to_numpy(float)-12,0,15)*.7
            for h in cfg.horizons:
                self.y[h][i,:m]=q[f'net_{h}'];self.down[h][i,:m]=q[f'mae_{h}']
                self.fills[h][i,:m]=q[f'outcome_state_{h}'].astype(str).str.startswith('ASSUMED_OPEN_FILL')
                self.basis[h][i,:m]=q[f'return_basis_{h}'].map(self.codes).fillna(0)
    def matrix(self,pairs,candidates,h):
        idx=[self.dates[d] for d,g in pairs];n=len(idx);m=len(candidates)
        if not n or not m:
            return {**{k:np.full((m,n),np.nan) for k in ('net','mae','fill')},'basis':np.full((m,n),'UNKNOWN',object)}
        weights=np.array([[w.get(k,0.) for k in self.keys] for w in candidates]);weights/=weights.sum(axis=1,keepdims=True)
        a=self.a[idx];mask=np.isfinite(a)
        scores=np.nan_to_num(a)@weights.T;cov=mask.astype(float)@weights.T
        scores=np.divide(scores,cov,out=np.full_like(scores,-np.inf),where=cov>=.75)
        scores-=self.penalty[idx,:,None]
        scores[~np.isfinite(scores)]=-np.inf
        scores=scores.transpose(2,0,1)
        ix=np.argsort(-scores,axis=2,kind='stable')[:,:,:self.cfg.top_k]
        chosen_score=np.take_along_axis(scores,ix,axis=2);ok=np.isfinite(chosen_score).sum(axis=2)==self.cfg.top_k
        def take(array):return np.take_along_axis(np.broadcast_to(array[idx],(m,n,array.shape[1])),ix,axis=2)
        y=take(self.y[h]);down=take(self.down[h]);fills=take(self.fills[h]);basis=take(self.basis[h])
        result={k:np.full((m,n),np.nan) for k in ('net','mae','fill')}
        good=ok&np.isfinite(y).all(axis=2);result['net'][good]=y.mean(axis=2)[good]
        good=ok&np.isfinite(down).all(axis=2);result['mae'][good]=down.mean(axis=2)[good]
        result['fill'][ok]=fills.mean(axis=2)[ok]
        # Cash is neutral to a known noncash basis; unknown/mixed never passes.
        bmin=np.where(basis==3,99,basis).min(axis=2)
        bmax=np.where(basis==3,-1,basis).max(axis=2)
        b=np.full((m,n),'UNKNOWN',object)
        b[ok&(bmax==-1)]='CASH'
        same=ok&(bmin==bmax)&(bmax>=0)
        b[same]=self.labels[bmax[same]]
        b[ok&(bmin!=bmax)&(bmax!=-1)]='UNKNOWN'
        result['basis']=b
        return result
