"""Causal, comparable scorecards. Percentiles use same-day eligible stocks only.

These are coordinates, NOT additional independent fundamental evidence. The
legacy engine retains its original scales and penalties as an exact control.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PARENTS=('dollar_turnover','medium_return','downside_resilience','revenue',
         'earnings_growth','earnings_yield','trust','foreign','breakout','pattern')
NEW_FACTORS=tuple('r196_'+k for k in PARENTS)
LABELS={
 'r196_dollar_turnover':'成交金額同日排序（不是市值）',
 'r196_medium_return':'60日漲幅同日排序',
 'r196_downside_resilience':'下檔韌性同日排序',
 'r196_revenue':'營收成長同日排序',
 'r196_earnings_growth':'盈餘成長同日排序',
 'r196_earnings_yield':'盈餘殖利率同日排序',
 'r196_trust':'投信買盤同日排序',
 'r196_foreign':'外資買盤同日排序',
 'r196_breakout':'突破結構同日排序',
 'r196_pattern':'W底型態同日排序',
}
OPTIONAL_PARENTS={'revenue','earnings_growth','earnings_yield','trust','foreign'}

def add_scorecards(panel,cfg):
    f=panel.copy()
    for k in PARENTS:
        rawkey={'dollar_turnover':'turnover20','medium_return':'ret60'}.get(k,k)
        raw=pd.to_numeric(f.get(rawkey,pd.Series(np.nan,index=f.index)),errors='coerce').replace([np.inf,-np.inf],np.nan)
        raw=raw.where(f.eligible & f.ticker.ne(cfg.benchmark))
        cnt=raw.groupby(f.date).transform('count')
        score=raw.groupby(f.date).rank(pct=True,method='average')*100.
        f['r196_'+k]=score.where(cnt>=cfg.min_universe)
    return f


def scorecard_weights(h,cfg):
    from v19_config import prior,TECH
    profiles={'legacy_control':prior(h,TECH),'turnover_only':{'r196_dollar_turnover':1.}}
    for other,label in [('downside_resilience','defense'),('medium_return','trend')]:
        for w in (.25,.5,.75):
            profiles[f'turnover_{label}_{int(w*100)}']={'r196_dollar_turnover':w,'r196_'+other:1-w}
    if cfg.optional_data:
        profiles.update({
            'revenue_only':{'r196_revenue':1.},
            'revenue_turnover':{'r196_revenue':.5,'r196_dollar_turnover':.5},
            'growth_valuation':{'r196_revenue':.4,'r196_earnings_growth':.3,'r196_earnings_yield':.3},
            'trust_breakout':{'r196_trust':.5,'r196_breakout':.5},
            'joint_flow_pattern':{'r196_trust':.35,'r196_foreign':.35,'r196_pattern':.3},
        })
    return profiles


def profile_label(name):
    name=str(name).removeprefix('simple_control_')
    if name=='turnover_only':return '\u6210\u4ea4\u91d1\u984d100%'
    if name.startswith('turnover_defense_') or name.startswith('turnover_trend_'):
        kind,weight=name.removeprefix('turnover_').rsplit('_',1)
        other='\u4e0b\u6a94\u97cc\u6027' if kind=='defense' else '60\u65e5\u52d5\u80fd'
        return '\u6210\u4ea4\u91d1\u984d'+weight+'% / '+other+str(100-int(weight))+'%'
    return {'revenue_only':'\u71df\u6536\u6210\u9577100%','revenue_turnover':'\u71df\u6536 / \u6210\u4ea4\u91d1\u984d\u540450%','legacy_control':'\u539f\u6a21\u578b\u5c0d\u7167','growth_valuation':'\u6210\u9577\u8207\u4f30\u503c','trust_breakout':'\u6295\u4fe1\u8207\u7a81\u7834','joint_flow_pattern':'\u6cd5\u4eba\u8207\u578b\u614b'}.get(name,name)


def simple_summary(report):
    rows=pd.DataFrame(report.get('results',[]));out=[]
    if rows.empty:return out
    benchmark=rows[rows.strategy=='benchmark_0050']
    groups=rows[rows.strategy.str.startswith('simple_control_',na=False)]
    for (h,name),g in groups.groupby(['horizon','strategy'],sort=True):
        pairs=g.merge(benchmark[['date','horizon','mean_net','price_basis','complete']],on=['date','horizon'],suffixes=('','_benchmark'))
        pairs=pairs[pairs.complete & pairs.complete_benchmark & pairs.mean_net.notna() & pairs.mean_net_benchmark.notna() & ((pairs.price_basis==pairs.price_basis_benchmark)|(pairs.price_basis=='CASH')|(pairs.price_basis_benchmark=='CASH'))]
        out.append({'horizon':int(h),'profile':name.removeprefix('simple_control_'),'paired_cases':len(pairs),'net':float(pairs.mean_net.mean()) if len(pairs) else None, 'benchmark_net':float(pairs.mean_net_benchmark.mean()) if len(pairs) else None, 'excess':float((pairs.mean_net-pairs.mean_net_benchmark).mean()) if len(pairs) else None})
    return out
