"""Small, append-only prospective ranking archive for *future* OOS evaluation.

This preserves what Alpha Radar actually displayed after each completed scan,
including its full scored universe. It cannot recreate earlier delisted stocks,
retroactively prove vendor publication times, or make ephemeral disk persistent.
"""
from __future__ import annotations
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import uuid


def freeze_completed_ranking(data_dir:Path, snap:dict) -> dict:
    from v174_integrity import assess_snapshot
    audit=assess_snapshot(snap)
    if not audit.get('safe_for_research'):
        return {'status':'SKIPPED','reason':'官方來源或模型日期尚未通過研究快照核對'}
    day=str(snap.get('price_date') or '')
    sid=str(snap.get('snapshot_id') or '')
    stocks=[s for s in (snap.get('stocks') or []) if isinstance(s,dict)]
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',day) or not sid or not stocks:
        return {'status':'SKIPPED','reason':'無有效模型日、快照識別或個股資料'}
    # Never write narrative/PII, news content, or provider secrets into journal.
    rows=[]
    for s in stocks:
        hs={}
        for h in ('short','mid','long'):
            block=(s.get('horizons') or {}).get(h) or {}
            score=block.get('ranking_score')
            if score is None:continue
            r18=block.get('rank_v18') or {}
            hs[h]={'alpha':score, 'v17_preblend':r18.get('score_v17'),
                   'v18_factor':r18.get('score_v18'),
                   'dq':r18.get('dq_score'), 'mc':r18.get('mc_score'),
                   'overlay_weight':r18.get('overlay_weight')}
        if not hs:continue
        shadow=s.get('limitup_shadow') or {}
        setup=shadow.get('setup') or {}
        pattern=s.get('pattern_chip_shadow') or {}
        rows.append({'ticker':s.get('ticker'), 'industry':s.get('fine_industry') or s.get('industry'),
                     'price_date':s.get('price_date'), 'reference_close':s.get('price'),
                     'average_turnover':s.get('avg_turnover_20d'), 'horizons':hs,
                     'pattern_chip_shadow':{
                         'version':pattern.get('version'),'research_rank':pattern.get('research_rank'),
                         'mode':pattern.get('mode'), 'reversal_rank':pattern.get('reversal_rank'),
                         'breakout_rank':pattern.get('breakout_rank'),
                         'w_bottom':pattern.get('w_bottom'),
                         'factor_scores':{k:(v or {}).get('score') for k,v in (pattern.get('factors') or {}).items()},
                         'factor_reasons':{k:(v or {}).get('why') for k,v in (pattern.get('factors') or {}).items()},
                         'chip_scores':{k:(v or {}).get('score') for k,v in ((pattern.get('chip') or {}).get('factors') or {}).items()},
                         'chip_coverage':(pattern.get('chip') or {}).get('evidence_coverage'),
                         'chip_sessions':(pattern.get('chip') or {}).get('continuous_flow_sessions'),
                         'eligible_research':pattern.get('eligible_research'),
                         'blockers':pattern.get('blockers'), 'is_buy_signal':False,
                     },
                     'limitup_shadow':{
                         'version':shadow.get('version'),'research_rank_score':shadow.get('research_rank_score'),
                         'setup_score':setup.get('setup_score'), 'eligible_research':shadow.get('eligible_research'),
                         'evidence_coverage':setup.get('evidence_coverage'),
                         'factors':{k:v.get('score') for k,v in (setup.get('factors') or {}).items()},
                         'is_buy_signal':False,
                     }})
    if not rows:
        return {'status':'SKIPPED','reason':'沒有可封存的選股分數'}
    meta={'snapshot_id':sid,'model_date':day,'generated_at':snap.get('generated_at'),
          'model_version':snap.get('model_version'),
          'limitup_research_contract':snap.get('limitup_shadow_summary') or {},
          'pattern_chip_contract':snap.get('pattern_chip_summary') or {},
          'source_dates':((snap.get('coverage') or {}).get('official_eod') or {}),
          'archive_status':'OBSERVED_AS_GENERATED',
          'limitations':'當天實際記錄的現行股票分數；不能補出建檔之前下市股、原始公佈時點或未來成交',
          'stocks':rows}
    root=Path(data_dir)/'prediction_journal'
    root.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(sid.encode('utf-8')).hexdigest()[:14]
    target=root/(day+'_'+key+'.json.gz')
    if target.exists():
        return {'status':'EXISTING','path':str(target),'stocks':len(rows)}
    temp=root/(target.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with gzip.open(temp,'wt',encoding='utf-8',compresslevel=6) as f:
            json.dump(meta,f,ensure_ascii=False,separators=(',',':'),default=str)
        # There is one filename per unique completed snapshot; do not replace
        # a prior decision timestamp with an updated late-night recalculation.
        if target.exists():
            return {'status':'EXISTING','path':str(target),'stocks':len(rows)}
        os.replace(temp,target)
        return {'status':'RECORDED','path':str(target),'stocks':len(rows)}
    finally:
        temp.unlink(missing_ok=True)
