"""Sparse-model explanations and input-evidence controls, no trading orders."""
from __future__ import annotations
from dataclasses import replace
import io
import json
from pathlib import Path
import pandas as pd
from v19_config import FACTORS, TECH, VERSION, prior
from v194_factors import input_support

STATES={
 'TEN_PILLAR_RESEARCH_ONLY':'十面向研究完成；尚未通過歷史資金使用檢查',
 'TEN_PILLAR_VALIDATION_INCOMPLETE':'十面向過去已完成區塊不足，沿用舊對照',
 'TEN_PILLAR_INTERNAL_GATE_PASSED':'通過內部過去區塊檢查，非實盤保證',
 'SCORECARD_RESEARCH_ONLY':'\u6a21\u578b\u6bd4\u8f03\u5df2\u5b8c\u6210\uff0c\u6295\u8cc7\u8b49\u64da\u5c1a\u4e0d\u8db3',
 'SCORECARD_PREQUENTIAL_INCOMPLETE':'\u904e\u53bb\u4e8b\u524d\u63a8\u85a6\u9a57\u8b49\u8cc7\u6599\u4e0d\u8db3',
 'SCORECARD_PREQUENTIAL_ACCEPTED':'\u901a\u904e\u904e\u53bb\u4e8b\u524d\u6c7a\u7b56\u6aa2\u67e5\uff08\u4ecd\u975e\u4e0b\u55ae\u6307\u4ee4\uff09',
 'SPARSE_INSUFFICIENT_PAST_BLOCKS':'\u904e\u53bb\u4e0d\u91cd\u758a\u9a57\u8b49\u5340\u584a\u4e0d\u8db3',
 'SPARSE_NO_ELIGIBLE_FACTORS':'\u56e0\u5b50\u8cc7\u6599\u8986\u84cb\u4e0d\u8db3',
 'SPARSE_SEARCH_INSUFFICIENT':'\u6642\u9593\u5207\u5206\u641c\u5c0b\u8cc7\u6599\u4e0d\u8db3',
 'SPARSE_VALIDATION_COHORT_INCOMPLETE':'\u6700\u5f8c\u9a57\u8b49\u7684\u540c\u53e3\u5f91\u8cc7\u6599\u4e0d\u8db3',
 'SPARSE_CANDIDATE_NOT_ADOPTED':'\u5df2\u9078\u51fa\u7a00\u758f\u5019\u9078\uff0c\u4f46\u672a\u901a\u904e\u63a1\u7528\u9580\u6abb',
 'SPARSE_INNER_VALIDATION_ACCEPTED':'\u901a\u904e\u904e\u53bb\u5340\u584a\u63a1\u7528\u6aa2\u67e5\uff08\u4ecd\u975e\u4e0b\u55ae\u6307\u4ee4\uff09',
}


def report_compatible(report,h,cfg,asof):
    settings=report.get('settings',{})
    info=report.get('weights',{}).get(str(h),{}).get('fit',{})
    critical=('years','stocks','top_k','min_universe','min_turnover','max_entry_gap',
              'buy_fee','sell_fee','sell_tax','benchmark_sell_tax','slippage','benchmark',
              'prefer_adjusted','universe_seed','sparse_max_inputs','sparse_budget',
              'sparse_beam','sparse_cv_folds','sparse_ridge','sparse_fit_coverage',
              'sparse_complexity','sparse_near_best','sparse_corr_limit',
              'weight_shrinkage','validation_blocks','train_dates','improvement_floor',
              'financial_history','financial_timing','statement_lag_days','max_mae_deterioration',
              'min_validation_fill','warmup','factor_design','tournament_blocks','tournament_uncertainty','universe_mode','min_download_coverage')
    same_settings=all(settings.get(k,getattr(cfg,k))==getattr(cfg,k) for k in critical)
    return (same_settings and report.get('version')==VERSION and info.get('factor_design')==cfg.factor_design and
                str(info.get('cutoff','9999'))<=str(pd.Timestamp(asof).date()) and
                bool(settings.get('optional_data'))==bool(cfg.optional_data) and
                settings.get('financial_timing',cfg.financial_timing)==cfg.financial_timing)


def choose_model(report,h,cfg,asof,labels,key='v194_rank_mode'):
    import streamlit as st
    choices=['\u7a00\u758f\u6a21\u578b\u7814\u7a76\u6392\u540d', '\u901a\u904e\u63a1\u7528\u6aa2\u67e5\u7684\u6392\u540d', '\u539f\u6a21\u578b\u5c0d\u7167']
    if cfg.factor_design in ('tournament_v196','ten_angle_v197','balanced_v198'):
        choices=['\u4e8b\u524d\u6a21\u578b\u6bd4\u8f03\u7814\u7a76\u6392\u540d','\u901a\u904e\u63a1\u7528\u6aa2\u67e5\u7684\u6392\u540d','\u539f\u6a21\u578b\u5c0d\u7167']
    mode=st.selectbox('\u4f7f\u7528\u54ea\u500b\u6a21\u578b',choices,key=key)
    block=report.get('weights',{}).get(str(h),{});info=block.get('fit',{})
    settings=report.get('settings',{})
    compatible=report_compatible(report,h,cfg,asof)
    if not compatible:info={};block={}
    weights=prior(h,TECH);control=True
    if mode==choices[0]:
        if info.get('challenger_weights'):
            weights=info['challenger_weights'];control=False
            st.warning('\u9019\u662f\u57fa\u6e96\u65e5\u524d\u8a13\u7df4\u9078\u51fa\u7684\u5c11\u6578\u56e0\u5b50\u6a21\u578b\uff0c\u4e0d\u4ee3\u8868\u5df2\u901a\u904e\u63a1\u7528\uff0c\u4e5f\u4e0d\u662f\u8cb7\u9032\u6307\u4ee4\u3002')
        else:
            st.info('\u76ee\u524d\u9084\u6c92\u6709\u76f8\u540c\u8a2d\u5b9a\u7684\u7a00\u758f\u8a13\u7df4\u7d50\u679c\uff1b\u5148\u986f\u793a\u539f\u6a21\u578b\u5c0d\u7167\uff0c\u8acb\u57f7\u884c\u5feb\u53d6\u9a57\u8b49\u3002')
    elif mode==choices[1] and block.get('weights'):
        weights=block['weights'];control=info.get('adopted_is_legacy_control',True)
    status=STATES.get(info.get('status'),info.get('status','\u5c1a\u672a\u8a13\u7df4'))
    st.caption(status)
    if info.get('cutoff'):
        st.caption('權重訓練截止：'+str(info['cutoff'])+'；目前行情日期：'+str(pd.Timestamp(asof).date())+'。更新行情不代表權重已重新訓練。')
    if (info.get('failed_adoption_checks') or info.get('failed_checks')):
        names={'mean_improvement':'平均改善不足','not_one_winner':'改善集中少數區塊',
               'majority_blocks_improve':'多數區塊未改善','beats_benchmark_on_average':'未勝0050',
               'positive_net_after_cost':'成本後淨報酬未正','tail_risk_not_materially_worse':'下檔風險增加',
               'sufficient_fill_fraction':'假設成交率不足','beats_same_pool_equal_weight':'未勝相同股票池等權'}
        st.caption('未採用原因：'+'、'.join(names.get(k,k) for k in (info.get('failed_adoption_checks') or info.get('failed_checks',[]))))
    active=[k for k,v in weights.items() if v>0]
    st.write(('\u539f\u6a21\u578b\u5c0d\u7167' if control else '\u672c\u6b21\u7a00\u758f\u6a21\u578b')+
             f' | {len(active)} '+ '\u500b\u6709\u6b0a\u91cd\u9805\u76ee | '+
             str(len(input_support(active)))+' \u500b\u5e95\u5c64\u8f38\u5165')
    if cfg.factor_design in ('tournament_v196','ten_angle_v197','balanced_v198'):
        st.caption('\u6a21\u578b\u6bd4\u8f03\u9078\u64c7\uff1a'+str(info.get('selected_profile','\u5c1a\u672a\u8a13\u7df4'))+' | '+str(info.get('selection_blocks',0))+' \u500b\u5df2\u5230\u671f\u5340\u584a\uff1b\u9078\u51fa\u65b9\u6848\u4e0d\u7b49\u65bc\u5df2\u6709\u7a69\u5b9a\u8d85\u984d\u5831\u916c\u3002')
    selection_table(info,weights,labels)
    return weights,replace(cfg,factor_design='legacy_v192') if control else cfg


def selection_table(info,weights,labels):
    import streamlit as st
    with st.expander('\u7559\u4e0b\u54ea\u4e9b\u56e0\u5b50\uff1f\u522a\u9664\u54ea\u4e9b\uff1f',expanded=False):
        table=[]
        for k in FACTORS:
            w=float(weights.get(k,0.));coverage=info.get('feature_coverage',{}).get(k)
            reason=info.get('excluded',{}).get(k)
            state='\u672c\u6b21\u6392\u540d\u4f7f\u7528' if w>0 else '\u8cc7\u6599\u4e0d\u8db3\u6216\u672a\u555f\u7528' if reason else '\u672a\u5165\u9078\uff0c\u6b0a\u91cd\u70ba0'
            table.append({'\u56e0\u5b50':labels.get(k,k),'\u6b0a\u91cd%':round(100*w,3),
                          '\u8a13\u7df4\u8986\u84cb%':None if coverage is None else round(100*coverage,1),
                          '\u72c0\u614b':state})
        st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
        st.caption('\u6b0a\u91cd\u662f\u6392\u5e8f\u4fc2\u6578\uff0c\u4e0d\u662f\u6295\u8cc7\u90e8\u4f4d\u6216\u4e0a\u6f32\u6a5f\u7387\u3002\u522a\u9664\u56e0\u5b50\u4e0d\u6703\u56e0\u7e2e\u6e1b\u6b0a\u91cd\u53c8\u88ab\u52a0\u56de\u3002')
        if info.get('sparse_search'):
            search=info['sparse_search']
            st.caption('係數估計方式：'+{'equal':'入選因子等權','ridge':'以訓練資料估計權重'}.get(search.get('weight_fit'),str(search.get('weight_fit','未提供')))+'；重估訓練資料 '+str(search.get('refit_dates',0))+' 日 / '+str(search.get('refit_rows',0))+' 列。')
            st.json(search)


def report_supports(report,labels):
    import streamlit as st
    if report.get('settings',{}).get('factor_design') not in ('sparse_v194','sparse_v195','tournament_v196','ten_angle_v197','balanced_v198'):return
    st.markdown('### \u5404\u9031\u671f\u771f\u6b63\u9078\u4e2d\u7684\u56e0\u5b50')
    rows=[]
    for h,b in report.get('weights',{}).items():
        f=b.get('fit',{});w=f.get('challenger_weights',{})
        rows.append({'\u6301\u6709\u65e5':h,'\u7814\u7a76\u5019\u9078':', '.join(f'{labels.get(k,k)} {v:.1%}' for k,v in w.items() if v>0) or '\u8cc7\u6599\u4e0d\u8db3',
                     '\u63a1\u7528\u72c0\u614b':STATES.get(f.get('status'),f.get('status'))})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.caption('\u539f\u6a21\u578b\u3001\u4e8b\u524d\u8a13\u7df4\u5019\u9078\u3001\u901a\u904e\u9580\u6abb\u5f8c\u63a1\u7528\u3001\u73fe\u91d1\u653f\u7b56\u662f\u4e0d\u540c\u7d50\u679c\uff0c\u4e0d\u80fd\u6df7\u5728\u4e00\u8d77\u6bd4\u8f03\u3002')


def import_evidence_control(root):
    import streamlit as st
    with st.expander('\u532f\u5165\u6709\u767c\u5e03\u6642\u9593\u7684\u57fa\u672c\u9762\uff0f\u7c4c\u78bcCSV',expanded=False):
        file=st.file_uploader('CSV: ticker, available_at, source',type=['csv'],key='v194_evidence_file')
        st.caption('\u50c5\u6703\u4f7f\u7528\u57fa\u6e96\u65e5\u7576\u6642\u5df2\u53ef\u5f97\u7684\u8cc7\u6599\u3002\u8acb\u4f7f\u7528\u9644\u5e36\u6b04\u4f4d\u7bc4\u672c\uff1b\u532f\u5165\u5f8c\u9700\u91cd\u65b0\u9a57\u8b49\u3002')
        if file is not None and st.button('\u532f\u5165\u8cc7\u6599',key='v194_import_evidence'):
            try:
                content=file.getvalue()
                if len(content)>10_000_000:raise ValueError('CSV exceeds 10MB')
                frame=pd.read_csv(io.BytesIO(content))
                if not {'ticker','available_at','source'}.issubset(frame):raise ValueError('Missing required columns')
                if frame.empty or frame[['ticker','available_at','source']].isna().any().any():raise ValueError('Empty required values')
                if pd.to_datetime(frame.available_at,errors='coerce',utc=True).isna().any():raise ValueError('Invalid availability timestamps')
                from v19_data import atomic_csv
                atomic_csv(Path(root)/'optional_evidence.csv',frame)
                st.success('\u5df2\u532f\u5165\uff1b\u8acb\u52fe\u9078\u9078\u914d\u8cc7\u6599\u4e26\u7528\u5feb\u53d6\u91cd\u65b0\u9a57\u8b49\u3002')
            except Exception as exc:st.error(type(exc).__name__+': '+str(exc))
