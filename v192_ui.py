"""V19.3 factor validation panel; lazy Streamlit import for headless tests."""
from pathlib import Path
import pandas as pd

SIGNAL_LABELS = {
    'kd_low_cross_signal': 'KD\u4f4e\u6a94\u9ec3\u91d1\u4ea4\u53c9\uff08\u4e8b\u4ef6\uff09',
    'macd_hist_flip_signal': 'MACD\u67f1\u9ad4\u7ffb\u6b63\uff08\u4e8b\u4ef6\uff09',
    'rsi_recovery_signal': 'RSI\u8d85\u8ce3\u56de\u5347\uff08\u4e8b\u4ef6\uff09',
    'negative_bias_repair_signal': '\u8ca0\u4e56\u96e2\u4fee\u5fa9\uff08\u4e8b\u4ef6\uff09',
    'w_breakout_signal': 'W\u5e95\u7a81\u7834\uff0f\u56de\u6e2c\uff08\u4e8b\u4ef6\uff09',
    'volume_breakout_signal': '\u653e\u91cf\u7a81\u7834\uff08\u4e8b\u4ef6\uff09',
}


def factor_section(report, labels):
    import streamlit as st
    plan = report.get('sampling_plan', {})
    if plan:
        st.info(f"\u62bd\u6a23\uff1a{plan.get('actual',0)}/{plan.get('requested',0)}\u500b\u4e0d\u91cd\u8907\u65e5\u671f\uff5c"
                f"\u53ef\u62bd\u65e5\u671f {plan.get('eligible_dates',0)}\u500b\uff5c\u7a2e\u5b50 {plan.get('seed')}\uff5c{plan.get('method')}")
        st.caption('\u4e0d\u540c\u57fa\u6e96\u65e5\u7684\u6301\u6709\u5340\u9593\u53ef\u80fd\u91cd\u758a\uff1b100\u500b\u65e5\u671f\u4e0d\u7b49\u65bc100\u6b21\u7368\u7acb\u5be6\u9a57\u3002')
        st.write('\u4f9d\u6642\u9593\u9806\u5e8f\u53d6\u4e0d\u91cd\u758a\u5b50\u96c6\u7684\u7d44\u6578\uff1a'+str(plan.get('disjoint_case_count',{})))
    rows = report.get('factor_summary', [])
    if not rows:
        st.caption('\u672c\u5831\u544a\u5c1a\u7121\u9010\u56e0\u5b50\u9a57\u8b49\u3002\u8acb\u4f7f\u7528V19.3\u91cd\u8dd1\uff1b\u820a\u7d50\u679c\u4e0d\u6703\u88ab\u5047\u88dc\u6210100\u9ede\u3002')
        return
    st.markdown('### \u54ea\u500b\u56e0\u5b50\u6709\u5e6b\u52a9\uff1f')
    hs = sorted(set(r['horizon'] for r in rows))
    h = st.radio('\u56e0\u5b50\u9a57\u8b49\u9031\u671f', hs, horizontal=True, format_func=lambda x:f'{x}\u4ea4\u6613\u65e5', key='v192_factor_h')
    chosen = [r for r in rows if r['horizon']==h]
    table = []
    for r in chosen:
        def fmt(value, percent=False):
            if value is None or pd.isna(value): return '\u2014'
            return f'{100*value:+.2f}%' if percent else f'{value:+.3f}'
        table.append({
            '\u56e0\u5b50\uff0f\u4e8b\u4ef6': SIGNAL_LABELS.get(r['factor'],labels.get(r['factor'],r['factor'])),
            '\u8cc7\u6599\u8986\u84cb':f"{r['mean_coverage']:.0%}",
            '\u6709\u6548IC\u7d44\u6578':r['ic_valid_cases'],
            '\u6392\u5e8f\u95dc\u806fIC':fmt(r['rank_ic_mean']),
            '\u79fb\u9664\u6bd4\u8f03\u7d44\u6578':r['ablation_paired_cases'],
            '\u4fdd\u7559\u6b64\u56e0\u5b50\u7684\u5831\u916c\u5dee':fmt(r['full_minus_removed_mean'],True),
            '\u4e0d\u91cd\u758a\u5b50\u96c6IC':fmt(r['disjoint_rank_ic_mean']),
        })
    st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
    st.caption('IC\u70ba\u6b63\uff1a\u9ad8\u5206\u6a19\u7684\u8207\u8f03\u9ad8\u5f8c\u7e8c\u5831\u916c\u540c\u5411\uff0c\u4e0d\u662f\u52dd\u7387\u3002\u300c\u5831\u916c\u5dee\u300d\u70ba\u5b8c\u6574\u6a21\u578b\u6e1b\u53bb\u79fb\u9664\u8a72\u56e0\u5b50\u5f8c\u7684\u540c\u65e5\u5831\u916c\uff1b\u6b63\u503c\u8868\u793a\u4fdd\u7559\u8a72\u56e0\u5b50\u5728\u9019\u4e9b\u6a23\u672c\u6709\u5e6b\u52a9\u3002')
    st.caption('\u4e8b\u4ef6\u5217\u662f\u6709\uff0f\u7121\u8a0a\u865f\u6bd4\u8f03\uff0c\u672a\u55ae\u7368\u5206\u914d\u6b0a\u91cd\uff0c\u6240\u4ee5\u79fb\u9664\u5dee\u7570\u7559\u7a7a\u3002\u8986\u84cb0%\u8868\u793a\u6c92\u8cc7\u6599\u53ef\u9a57\uff0c\u4e0d\u4ee3\u8868\u56e0\u5b50\u7121\u6548\u3002\u8868\u683c\u4e0d\u6703\u4e8b\u5f8c\u56de\u5beb\u540c\u65e5\u6b0a\u91cd\u3002')


def database_download(root):
    """Prepare archive bytes only after the user asks for a download.

    Streamlit serializes st.download_button(data=...) into RAM.  The previous
    startup eagerly read the entire SQLite+history ZIP, which could kill a
    resource-constrained Cloud worker before the stock dashboard appeared.
    """
    import streamlit as st
    archive=Path(root)/'v192_history_database.zip'
    if not archive.is_file():
        return
    size=archive.stat().st_size
    mb=size/(1024*1024)
    if size>150*1024*1024:
        st.warning(f'研究資料庫約 {mb:.0f} MiB，超過雲端安全下載門檻。請從持久儲存或離線環境取得，以免下載時耗盡記憶體。')
        return
    if st.button(f'準備資料庫下載（{mb:.1f} MiB）', key='v192_prepare_database'):
        st.session_state['v192_download_ready']=True
    if st.session_state.get('v192_download_ready'):
        st.download_button('下載歷史資料庫（SQLite＋原始資料）',archive.read_bytes(),
                           file_name=archive.name,mime='application/zip',key='v192_database_download')
