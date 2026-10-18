"""User-facing V19.5 validation and deployment-state summaries."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from v19_config import VERSION
from v191_reporting import summarize_results


def pct(x):
    return '\u2014' if x is None or pd.isna(x) else f'{float(x)*100:+.2f}%'


def validation_overview(report):
    st.markdown('### \u540c\u57fa\u6e96\u65e5\u7e3e\u6548\u5c0d\u7167')
    if report.get('version')!=VERSION:
        st.warning('\u9019\u662f\u820a\u7248\u5831\u544a\uff0c\u5c1a\u672a\u7528V19.5\u91cd\u8dd1\uff1b\u820a\u7248\u81ea\u52d5\u8abf\u6574\u6b0a\u91cd\u4e0d\u6703\u5957\u7528\u5230\u65b0\u7248\u6392\u540d\u3002')
    summary=report.get('summary') or summarize_results(report.get('results',[]),report.get('settings',{}).get('horizons',[5,20,60]))[0]
    rows=[]
    for r in summary:
        rows.append({
            '\u6301\u6709\u4ea4\u6613\u65e5':r['horizon'],
            '\u6a21\u578b\u53ef\u6bd4\u7d44\u6578':r['paired_model_cases'],
            '\u56fa\u5b9a\u6b0a\u91cd\u5e73\u5747':pct(r['baseline_mean']),
            '\u8abf\u6b0a\u5e73\u5747':pct(r['adaptive_mean']),
            '\u8abf\u6b0a\u5dee\u7570':pct(r['weight_change_mean']),
            '\u82070050\u540c\u65e5\u7d44\u6578':r['matched_benchmark_cases'],
            '\u8abf\u6b0a\u540c\u65e5\u5e73\u5747':pct(r['matched_adaptive_mean']),
            '0050\u540c\u65e5\u5e73\u5747':pct(r['matched_benchmark_mean']),
            '\u8d85\u984d\u5831\u916c':pct(r['mean_excess']),
            '\u52dd\u904e0050\u7d44\u6578':f"{r['benchmark_beats']}/{r['matched_benchmark_cases']}",
        })
    if rows:st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.caption('\u4e0a\u8868\u662f\u5404\u57fa\u6e96\u65e5\u7b97\u8853\u5e73\u5747\uff0c\u4e0d\u662f\u7d2f\u7a4d\u6216\u5e74\u5316\u5831\u916c\u3002\u7f3a\u503c\u4e0d\u75760\uff1b\u65b0\u7248\u53ea\u6bd4\u8f03\u540c\u65e5\u6709\u6548\u4e14\u50f9\u683c\u57fa\u790e\u76f8\u5bb9\u7684\u7d44\u5225\u3002\u820a\u5831\u544a\u672a\u8a18\u9304\u9084\u539f\u50f9\u57fa\u790e\u6642\uff0c\u53ea\u80fd\u4f5c\u63cf\u8ff0\uff0c\u4e0d\u80fd\u4f5c\u63a1\u7528\u6b0a\u91cd\u7684\u8b49\u64da\u3002')
    if report.get('evaluation_role'):
        st.info('\u672c\u6279\u5df2\u770b\u904e\u7684\u6b77\u53f2\u65e5\u671f\u5df2\u88ab\u7528\u4f86\u6aa2\u8a0e\u8a2d\u8a08\u3002\u91cd\u8dd1\u5c6c\u65bc\u958b\u767c\u91cd\u64ad\uff0c\u4e0d\u53ef\u518d\u7a31\u70ba\u5168\u65b0\u76f2\u6e2c\u6216\u6700\u4f73\u7b56\u7565\u8b49\u660e\u3002')
    cases=report.get('case_diagnostics') or []
    if cases:
        with st.expander('\u7f3a\u503c\u539f\u56e0\uff0f\u9084\u539f\u50f9\u53e3\u5f91\uff0f\u73fe\u91d1\u6c7a\u7b56',expanded=False):
            st.dataframe(pd.DataFrame(cases),hide_index=True,use_container_width=True)
            st.caption('ENDPOINT_COMPLETE_PATH_INCOMPLETE\uff1a\u9032\u51fa\u50f9\u53ef\u7b97\u6301\u6709\u5831\u916c\uff0c\u4f46\u7f3a\u9014\u4e2d\u8cc7\u6599\uff0c\u4e0d\u986f\u793a\u5047\u7684\u6700\u5927\u56de\u64a4\u3002')


def allocation_banner(report,horizon,cfg=None,asof=None):
    block=(report.get('weights') or {}).get(str(horizon),{})
    fit=block.get('fit') or {}
    match=True
    if cfg is not None and cfg.factor_design in ('sparse_v194','sparse_v195'):
        from v194_ui import report_compatible
        match=report_compatible(report,horizon,cfg,asof or fit.get('cutoff','1900-01-01'))
    if match and report.get('version')==VERSION and fit.get('allocation_allowed') and report.get('data_kind')=='PROVIDER_HISTORY':
        st.info('\u904e\u53bb\u4e0d\u91cd\u758a\u5167\u90e8\u5340\u584a\u901a\u904e\u6bd4\u8f03\u9580\u6abb\uff1b\u4ee5\u4e0b\u4ecd\u662f\u7814\u7a76\u5019\u9078\uff0c\u4e0d\u662f\u5373\u6642\u8cb7\u9032\u6307\u4ee4\u3002')
    else:
        st.warning('\u76ee\u524d\u5c1a\u672a\u8b49\u660e\u6b64\u9031\u671f\u7684\u9078\u80a1\u80fd\u7a69\u5b9a\u512a\u65bc\u540c\u671f0050\u3002\u4ee5\u4e0b\u5217\u70ba\u89c0\u5bdf\u6e05\u55ae\uff0c\u4e0d\u6703\u56e0\u70ba\u6709\u6392\u540d\u5c31\u5efa\u8b70\u6eff\u5009\u8cb7\u5165\u3002')
    if fit and match:
        with st.expander('\u672c\u9031\u671f\u7684\u6b0a\u91cd\u63a1\u7528\u4f9d\u64da',expanded=False):
            st.json({'status':fit.get('status'),'train_dates':fit.get('train_dates'),
                'nonoverlap_validation_blocks':fit.get('validation_common_blocks',0),
                'allocation_reason':fit.get('allocation_reason'),
                'gate':fit.get('stability_gate'), 'excluded_factors':fit.get('excluded'),
                'factor_training_diagnostics':fit.get('factor_diagnostics')})
