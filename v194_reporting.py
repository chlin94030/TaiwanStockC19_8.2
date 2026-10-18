"""Selected, excluded, and adopted supports, distinct from performance claims."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import pandas as pd
from v19_config import FACTORS
from v194_factors import input_support


def sparse_summary(fits):
    rows=[]
    for h in sorted({r['horizon'] for r in fits}):
        cases=[r for r in fits if r['horizon']==h]
        counts=Counter(k for r in cases for k in r.get('selected_factors',[]))
        for k in FACTORS:
            weights=[r.get('challenger_weights',{}).get(k,0.) for r in cases if r.get('challenger_weights')]
            rows.append({'horizon':h,'factor':k,'candidate_cases':len(weights),
                         'selected_cases':counts[k], 'mean_candidate_weight':sum(weights)/len(weights) if weights else None,
                         'adopted_cases':sum(r.get('status') in ('SPARSE_INNER_VALIDATION_ACCEPTED','SCORECARD_PREQUENTIAL_ACCEPTED') and r.get('challenger_weights',{}).get(k,0)>0 for r in cases),
                         'not_a_probability':True})
    return rows


def export_sparse(output, report):
    from v19_data import atomic_csv,atomic_json
    output=Path(output);rows=[];latest=[]
    for r in report.get('fits',[]):
        for k in FACTORS:
            w=r.get('challenger_weights',{}).get(k,0.)
            rows.append({'cutoff':r['cutoff'],'horizon':r['horizon'],'factor':k,'candidate_weight':w,
                'adopted_weight':r.get('weights',{}).get(k,0.),'selected':w>0,
                'state':'SELECTED' if w>0 else r.get('excluded',{}).get(k,'NOT_SELECTED'),
                'fit_status':r.get('status'),'training_coverage':r.get('feature_coverage',{}).get(k),
                'coefficient_method':r.get('sparse_search',{}).get('weight_fit') or ('past_only_weight_grid' if r.get('selected_profile') else None),
                'refit_rows':r.get('sparse_search',{}).get('refit_rows'),
                'failed_checks':';'.join(r.get('failed_adoption_checks',[]))})
    for h,b in report.get('weights',{}).items():
        info=b.get('fit',{});candidate=info.get('challenger_weights',{})
        for k in FACTORS:
            latest.append({'horizon':int(h),'factor':k,'candidate_weight':candidate.get(k,0.),
                           'adopted_weight':b.get('weights',{}).get(k,0.),
                           'selected':candidate.get(k,0.)>0,'fit_status':info.get('status')})
    atomic_csv(output/'sparse_factor_selection.csv',pd.DataFrame(rows))
    atomic_csv(output/'sparse_latest_weights.csv',pd.DataFrame(latest))
    atomic_csv(output/'sparse_selection_summary.csv',pd.DataFrame(report.get('sparse_selection_summary',[])))
    atomic_json(output/'sparse_protocol.json',{'version':report.get('version'),'settings':report.get('settings'),
        'subset_search':('bounded_scorecard_weight_grid_with_prequential_policy_validation' if report.get('settings',{}).get('factor_design')=='tournament_v196' else 'nested_purged_forward_backward_swap_bounded'),'global_optimum_proven':False,
        'original_factors_forced_into_candidate':False,'deleted_factors_restored_by_shrinkage':False,
        'financial_revision_ledger_verified':False,'strict_candidate_input_coverage':1.0,
        'development_data_not_new_blind_holdout':True,
        'entry_exit_and_commissions':'see settings; hypothetical not executed',
        'equity_universe_is_not_historical_membership':True,
        'support_local_training':report.get('settings',{}).get('factor_design')=='sparse_v195',
        'globally_disjoint_inner_score_windows':report.get('settings',{}).get('factor_design')=='sparse_v195'})


def sparse_markdown(report):
    if report.get('settings',{}).get('factor_design') not in ('sparse_v194','sparse_v195','tournament_v196','ten_angle_v197','balanced_v198'):return ''
    if report.get('settings',{}).get('factor_design')=='tournament_v196':
        return tournament_markdown(report)
    if report.get('settings',{}).get('factor_design') in ('ten_angle_v197','balanced_v198'):
        return ten_angle_markdown(report)
    out=['## Sparse subset selection (V'+str(report.get('version','unknown'))+')', '',
         'Legacy is an independent control, NOT a mandatory component in a challenger.',
         'Every original factor can be zero. Shrinkage only acts within the positive learned support.',
         'Search, coefficient fitting, and the final adoption test use separate chronological stages.',
         'Search is bounded, not proof of a global optimum. Control and cash results remain separate.',
         'Quarter-end dates are not publication timestamps. Default strict financial mode excludes undated statements.',
         'Optional lagged financial research assumes availability and does not establish original unrevised PIT data.', '',
         '| Horizon | Latest selected research factors | Adopted factors | Status |',
         '|---|---|---|---|']
    for h,b in report.get('weights',{}).items():
        info=b.get('fit',{});candidate=info.get('challenger_weights',{})
        text=', '.join(f'{k}: {100*v:.2f}%' for k,v in candidate.items() if v>0) or 'No fitted candidate'
        adopted=', '.join(f'{k}: {100*v:.2f}%' for k,v in b.get('weights',{}).items() if v>0)
        out.append(f'| {h} | {text} | {adopted} | {info.get("status")} |')
    out += ['', 'Detailed per-date included/excluded masks: sparse_factor_selection.csv.',
            'Selection frequency is not a hit probability. Zero data coverage is NOT evidence of zero predictive value.']
    return '\n'.join(out)


def tournament_markdown(report):
    out=['## Simple-control tournament and prequential evidence', '',
        'A bounded weight grid compares transparent simple models to the independent legacy control.',
        'Policy selection uses the last completed, disjoint blocks. It is not independent validation of itself.',
        'Capital-use checks separately replay the algorithm choices that were issued at earlier dates, each using only data then available.',
        'No claim of a globally optimal weight vector, confidence interval, or future profit is made.',
        'Same-day percentile coordinates are not new independent factors. Dollar turnover is not market capitalization.',
        'Grid values are fixed research assumptions. All exposed-date results are development diagnostics.', '',
        '| Horizon | Latest research profile | Research weights | Gate |',
        '|---|---|---|---|']
    for h,b in report.get('weights',{}).items():
        i=b.get('fit',{});w=i.get('challenger_weights',{})
        out.append('| '+str(h)+' | '+str(i.get('selected_profile'))+' | '+', '.join(k+' '+format(v,'.1%') for k,v in w.items())+' | '+str(i.get('status'))+' |')
    out += ['', 'Simple controls are reported independently; a retrospectively best control is not automatically promoted.',
            'Missing selected outcomes and price-basis conflicts are retained as missing, never converted to cash.']
    from v196_factors import simple_summary
    simple=simple_summary(report)
    if simple:
        out += ['', '| Sessions | Fixed profile | Paired | Net mean | 0050 matched | Excess |', '|---|---|---:|---:|---:|---:|']
        def pct(x):return '--' if x is None else format(x,'+.3%')
        for x in simple:out.append('| '+str(x['horizon'])+' | '+x['profile']+' | '+str(x['paired_cases'])+' | '+pct(x['net'])+' | '+pct(x['benchmark_net'])+' | '+pct(x['excess'])+' |')
    return '\n'.join(out)


def ten_angle_markdown(report):
    out=['## V19.7 ten-perspective causal research','',
         'Ten lenses are evaluated separately; market breadth is a risk context, not a redundant stock score.',
         'Sector composition is not historically verified and sector hypotheses cannot authorize allocation.',
         'The present dates have been repeatedly used for development and are not new blind tests.','',
         '| Horizon | Latest pre-cutoff research profile | Internal gate |', '|---|---|---|']
    for h,b in report.get('weights',{}).items():
        info=b.get('fit',{})
        out.append(f"| {h} | {info.get('selected_profile','--')} | {info.get('status')} |")
    out+=['','| Factor | Data coverage | Evidence condition |','|---|---:|---|']
    for c in report.get('ten_pillar_coverage',[]):
        out.append(f"| {c['name']} | {c['coverage']:.1%} | {c['reason']} |")
    out+=['','### Fixed hypothesis comparison (paired calendar dates and price bases)',
          '| Horizon | Hypothesis | Cases | Net | Legacy | 0050 | Excess 0050 |',
          '|---|---|---:|---:|---:|---:|---:|']
    for x in report.get('ten_pillar_shadow_summary',[]):
        out.append(f"| {x['horizon']} | {x['profile']} | {x['paired_cases']} | {x['mean_net']:+.2%} | {x['legacy_mean']:+.2%} | {x['benchmark_mean']:+.2%} | {x['vs_0050']:+.2%} |")
    return '\n'.join(out)
