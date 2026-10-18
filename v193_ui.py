"""Broader-factor labels and side-by-side research presentation."""
from __future__ import annotations
import pandas as pd

LABELS = {
 'trend_efficiency':'\u8da8\u52e2\u884c\u9032\u6548\u7387',
 'momentum_persistence':'\u6392\u9664\u8fd1\u4e94\u65e5\u7684\u52d5\u80fd\u6301\u7e8c\u6027',
 'accumulation':'\u50f9\u91cf\u8cb7\u58d3\u7d2f\u7a4d\uff08\u975e\u6cd5\u4eba\uff09',
 'downside_resilience':'\u4e0b\u6a94\u6ce2\u52d5\u8207\u56de\u64a4\u97cc\u6027',
 'liquidity_quality':'\u6210\u4ea4\u6d41\u52d5\u6027\u54c1\u8cea',
 'orderly_pullback':'\u591a\u982d\u6709\u5e8f\u56de\u6a94\u4fee\u5fa9',
 'squeeze_release':'\u6ce2\u52d5\u6536\u6582\u5f8c\u7684\u50f9\u91cf\u91cb\u653e',
 'market_resilience':'\u5e02\u5834\u4e0b\u8dcc\u65e5\u76f8\u5c0d\u6297\u8dcc',
}
STRATEGIES={
 'expanded_fixed_research':'\u64f4\u5145\u56e0\u5b50\u56fa\u5b9a\u8a66\u9a57\u7d44',
 'training_challenger_research':'\u4e8b\u524d\u8a13\u7df4\u6311\u9078\u7d44',
 'adaptive':'\u63a1\u7528\u5f8c\u7d50\u679c\uff08\u672a\u904e\u5247\u6cbf\u7528\uff09',
 'guarded':'\u53e6\u5217\u80a1\u7968\uff0f\u73fe\u91d1\u7b56\u7565',
}


def broader_results(report):
    import streamlit as st
    rows=report.get('expanded_summary',[])
    if report.get('factor_audit_reference')=='expanded_fixed_research':
        st.caption('\u672c\u6b21\u56e0\u5b50\u79fb\u9664\u6bd4\u8f03\u4ee5\u65b0\u56e0\u5b50\u56fa\u5b9a\u8a66\u9a57\u7d44\u70ba\u57fa\u6e96\uff0c\u4e0d\u628a\u672a\u63a1\u7528\u7684\u96f6\u6b0a\u91cd\u8aa4\u89e3\u70ba\u56e0\u5b50\u7121\u6548\u3002')
    if not rows:return
    st.markdown('### \u65b0\u56e0\u5b50\u5c0d\u820a\u6a21\u578b\uff1a\u4e0d\u53ea\u770b\u6709\u6c92\u6709\u901a\u904e\u8abf\u6b0a')
    def pct(x):return '\u2014' if x is None or pd.isna(x) else f'{x*100:+.2f}%'
    table=[]
    for r in rows:
        table.append({'\u6301\u6709\u65e5':r['horizon'],'\u7b56\u7565':STRATEGIES.get(r['strategy'],r['strategy']),
            '\u540c\u65e5\u7d44\u6578':r['paired_cases'],'\u820a\u6a21\u578b':pct(r['legacy_mean']),
            '\u672c\u7d44':pct(r['candidate_mean']),'\u5831\u916c\u5dee':pct(r['mean_gain']),
            '\u82070050\u5dee':pct(r['mean_excess']),'\u4e0d\u91cd\u758a\u7d44\u6578':r['disjoint_cases'],
            '\u4e0d\u91cd\u758a\u5831\u916c\u5dee':pct(r['disjoint_mean_gain'])})
    st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
    st.caption('\u4e8b\u524d\u8a13\u7df4\u7d44\u4e0d\u7b49\u65bc\u5df2\u901a\u904e\u63a1\u7528\u3002\u4e0a\u8868\u70ba\u540c\u65e5\u7b97\u8853\u5e73\u5747\uff0c\u4e0d\u662f\u8907\u5229\u5831\u916c\u3002\u9019\u6279\u6b77\u53f2\u5df2\u770b\u904e\uff0c\u4ecd\u662f\u958b\u767c\u91cd\u64ad\uff0c\u4e0d\u662f\u65b0\u76f2\u6e2c\u3002')
