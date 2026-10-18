"""V19.0 mobile-first research dashboard. Every empty field has an explicit state."""
from __future__ import annotations
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
import html
import json
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from v19_config import Settings, FACTORS, TECH, TZ, VERSION
from v19_data import read_json
from v19_features import current_candidates
from v191_ui import validation_overview, allocation_banner
from v192_ui import factor_section, database_download
from v19_service import (start_job,job_status,material_for_display,read_last_report,
                         weights_for,analog_summary,field_health,position_size,request_cancel)

ROOT=Path(__file__).resolve().parent
BASE=Path(os.getenv('ALPHA_RADAR_DATA_DIR',str(ROOT/'data')))
DATA=BASE/'v19'
LABELS={
 'trend':'趨勢／均線','relative_strength':'相對大盤強度','breakout':'平台突破',
 'reversal':'KD／MACD／RSI修復','pattern':'W底／頸線','volume':'價量搭配',
 'revenue':'營收成長','quality':'獲利品質','foreign':'外資持續買超','trust':'投信持續買超',
 'joint':'外投同步買超','dealer':'自營商自行買賣','margin':'融資整理','holders':'大戶持股變化',
 'ticker':'股票代碼','name':'名稱','date':'資料日期','close':'日線收盤','source':'行情來源',
 'rsi':'RSI(14)','k':'KD：K值','d':'KD：D值','macd_hist':'MACD柱體','bias20':'20日乖離％',
 'neckline':'W底頸線','real_market_validation':'隨機日期歷史研究'}
from v193_ui import LABELS as BROADER_LABELS, broader_results
LABELS.update(BROADER_LABELS)
from v194_factors import LABELS as SPARSE_LABELS
from v194_ui import STATES as SPARSE_STATES, choose_model, report_supports, import_evidence_control
LABELS.update(SPARSE_LABELS)
from v196_factors import LABELS as SCORECARD_LABELS
LABELS.update(SCORECARD_LABELS)
from v197_ten_pillars import NAMES as TEN_LABELS
LABELS.update(TEN_LABELS)
SETUP={'REVERSAL':'低檔修復觀察','BREAKOUT':'突破加速觀察','GROWTH_PULLBACK':'業績＋回檔觀察','WATCH':'趨勢追蹤'}
FIT={'OLD_REPORT_NOT_APPLIED':'舊版報告未套用，請重跑V19.8',
     'BASELINE_INSUFFICIENT_INDEPENDENT_BLOCKS':'不重疊區塊不足，保留基準',
     'BASELINE_TRAIN_COHORT_INCOMPLETE':'同口徑訓練資料不足',
     'BASELINE_VALIDATION_COHORT_INCOMPLETE':'同日內部驗證資料不足',
     'BASELINE_STABILITY_NOT_PROVEN':'調權穩定性未通過，保留基準',
     'ROBUST_INNER_VALIDATION_ACCEPTED':'通過過去區塊的調權檢查（研究）',
     'STARTING_PRIOR_NO_FIT':'起始權重，尚未訓練','BASELINE_INSUFFICIENT_TRAINING':'訓練樣本不足，沿用基準',
     'BASELINE_TECH_GAP':'技術資料不足，沿用基準','BASELINE_VALIDATION_NOT_BETTER':'內部驗證未明顯勝出，保留基準',
     'RESEARCH_FIT':'已依當時以前資料調整（研究用途）'}
FIT.update(SPARSE_STATES)
CSS="""
<style>
.block-container{max-width:1120px;padding-top:1.1rem;padding-bottom:3rem}
h1{font-size:1.85rem!important;letter-spacing:-.025em}h2{font-size:1.3rem!important}
.v19-note{border-left:4px solid #537cb6;background:#f1f5fb;padding:12px 16px;border-radius:6px;
color:#304663;margin:12px 0;line-height:1.7}
.v19-card{border:1px solid #dbe3ef;border-radius:16px;padding:18px 20px;margin:13px 0;background:#fff;
box-shadow:0 3px 12px rgba(38,60,90,.04);color:#20334c}
.v19-title{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;font-size:1.13rem;font-weight:750}
.v19-tag{display:inline-block;background:#ecf3fc;color:#395f90;padding:4px 9px;border-radius:7px;
font-size:.83rem;font-weight:600;margin:9px 6px 5px 0}
.v19-small{font-size:.88rem;color:#647287;line-height:1.65;overflow-wrap:anywhere}
.v19-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:12px 0}
.v19-metrics span{display:block;font-size:.78rem;color:#647287}.v19-metrics b{font-size:1.22rem}
.v19-up{color:#bc4740}.v19-down{color:#17795c}
.v19-muted{color:#68788c}
[data-testid="stRadio"] [role="radiogroup"]{flex-wrap:wrap;gap:5px}
[data-testid="stDataFrame"]{max-width:100%}
@media(max-width:640px){.block-container{padding-left:1rem;padding-right:1rem}
.v19-card{padding:15px 13px}.v19-title{font-size:1.05rem}.v19-metrics{gap:8px}
.v19-metrics b{font-size:1.07rem}.v19-metrics span{font-size:.75rem}}
</style>
"""

def esc(x):return html.escape(str(x if x is not None else ''),quote=True)

def number(value,digits=2):
    try:
        v=float(value)
        return f'{v:,.{digits}f}' if np.isfinite(v) else '—'
    except (TypeError,ValueError):return '—'

def percent(value,scale=100):
    try:
        v=float(value)*scale
        return f'{v:+.2f}%' if np.isfinite(v) else '—'
    except (TypeError,ValueError):return '—'

def _bootstrap():
    if not os.getenv('FINMIND_TOKEN'):
        try:
            token=st.secrets.get('FINMIND_TOKEN')
            if token:os.environ['FINMIND_TOKEN']=str(token)
        except Exception:pass

@st.cache_data(ttl=600,show_spinner=False)
def _display_data(root,price_mtime,evidence_mtime,legacy_mtime,cfg):
    ui_bars = 340 if cfg.stocks <= 600 else 270
    return material_for_display(root,cfg,recent_bars=ui_bars)

def _mtime(path):
    try:return Path(path).stat().st_mtime
    except OSError:return 0.

@st.fragment(run_every=3)
def progress_box():
    status=job_status(DATA)
    if not status:return
    state=status.get('state')
    stages={'prepare':'準備','download':'下載歷史日線','optional':'補充營收／籌碼','features':'計算因子','validate':'歷史驗證','database':'儲存資料庫','latest_fit':'計算最近權重','factor_report':'匯整因子報告'}
    stage=stages.get(status.get('stage'),status.get('stage',''))
    if state=='RUNNING':
        total=max(1,int(status.get('total',1)));done=int(status.get('done',0))
        st.progress(min(1.,done/total),text=f"{stage}：{done}/{total}｜{status.get('message','')}")
        elapsed=max(0,time.time()-float(status.get('started_epoch',time.time())))
        st.caption(f'已執行 {elapsed/60:.1f} 分鐘。手機重連不重開任務；同資料與設定可從已完成的日期／週期繼續。主機休眠或資料夾被清除則無法保證保留。')
        if st.button('暫停（保留已完成結果）',key='v192_pause'):
            request_cancel(DATA)
            st.info('已請求暫停；當前連線或計算結束後儲存進度。')
    elif state=='DONE':
        st.success('本次流程完成：'+str(status.get('message','')))
        key='v19_seen_'+str(status.get('run_id',''))
        if not st.session_state.get(key):
            st.session_state[key]=True
            _display_data.clear()
            st.rerun()
    elif state in ('FAILED','INTERRUPTED','PAUSED','NEEDS_MORE_DATA'):
        st.warning('本次更新未完成；先前成功結果仍保留。'+str(status.get('message','')))
    with st.expander('執行紀錄',expanded=False):
        st.json(status)

def candidate_card(row,analog,horizon,weights,index):
    valid=analog.get('state')=='HISTORICAL_ANALOG_NOT_CALIBRATED_FORECAST'
    median=analog.get('median_net') if valid else None
    mean=analog.get('mean_net') if valid else None
    count=analog.get('count',0)
    tone='v19-up' if median is not None and median>0 else 'v19-down' if median is not None else 'v19-muted'
    factors=sorted(((k,float(row.get(k,np.nan))*v) for k,v in weights.items()
                   if v>0 and pd.notna(row.get(k))),key=lambda x:x[1],reverse=True)[:3]
    reasons='／'.join(LABELS.get(k,k) for k,_ in factors) or '因子資料不足'
    risk=number(row.get('stop_distance_pct'))
    missing=[LABELS[k] for k in FACTORS if pd.isna(row.get(k))]
    evidence='可比較的歷史樣本不足' if not valid else f"樣本 {count} 組｜歷史正報酬比例 {percent(analog.get('positive_fraction'))}"
    st.markdown(f"""<div class="v19-card">
      <div class="v19-title"><div>#{index:02d}　{esc(row.get('name'))} <span class="v19-small">{esc(row['ticker'])}</span></div>
        <div>日線 {number(row.get('close'))}</div></div>
      <span class="v19-tag">{esc(SETUP.get(row.get('setup'),'研究觀察'))}</span>
      <span class="v19-tag">研究分 {number(row.get('score'),1)}</span>
      <span class="v19-tag">因子覆蓋 {number(float(row.get('factor_coverage',0))*100,0)}%</span>
      <div class="v19-metrics">
        <div><span>{horizon}日歷史平均淨報酬</span><b>{percent(mean)}</b></div>
        <div><span>歷史中位數（非保證）</span><b class="{tone}">{percent(median)}</b></div>
        <div><span>參考停損距離</span><b>{risk}%</b></div>
      </div>
      <div class="v19-small">{esc(reasons)}<br>{esc(evidence)}<br>
      觀察區 {number(row.get('reference_low'))}–{number(row.get('reference_high'))}｜
      防守參考 {number(row.get('stop_reference'))}｜不追價參考 {number(row.get('chase_ceiling'))}<br>
      資料日 {esc(str(row.get('date'))[:10])}｜來源 {esc(row.get('source'))}<br>
      這是日線研究候選，不是即時買進指令；價區以本次日線與ATR估算。</div></div>""",unsafe_allow_html=True)
    with st.expander(f"{row['ticker']}：因子數值與缺資料說明",expanded=False):
        table=[{'因子':LABELS.get(k,k),'分數':number(row.get(k),1),
                '本次權重':f"{weights.get(k,0)*100:.1f}%",
                '狀態':'缺資料，不填0' if pd.isna(row.get(k)) else '可用'} for k in FACTORS]
        st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)

def page_decisions(panel,report,cfg):
    st.subheader('先看機會，再看進場，最後控制投入金額')
    st.markdown('<div class="v19-note">排序、買點與資料可信度分開。研究分高不代表一定漲停；歷史平均與中位數不是未來報酬保證。沒有營收／獲利證據的股票，不會冒稱「業績優良」。</div>',unsafe_allow_html=True)
    if panel.empty:
        st.info('尚未載入行情。按上方「下載近年資料＋執行驗證」即可；已有V18資料庫時會先讀取，無須清除原資料。')
        return
    h=st.radio('持有期（交易日）',list(cfg.horizons),horizontal=True,format_func=lambda x:f'{x}日',key='v19_h')
    asof=panel.date.max();weights,state=weights_for(report,h,asof)
    st.caption(f"日線截至 {str(asof.date())}｜{FIT.get(state,state)}。來源標示與日期分開，不把來源待核一律叫舊日線。")
    allocation_banner(report,h,cfg,asof)
    scoring_cfg=cfg
    if cfg.factor_design in ('sparse_v194','sparse_v195','tournament_v196','ten_angle_v197','balanced_v198'):
        weights,scoring_cfg=choose_model(report,h,cfg,asof,LABELS)
    if cfg.factor_design not in ('sparse_v194','sparse_v195','tournament_v196','ten_angle_v197','balanced_v198'):
        design=st.selectbox('\u6392\u540d\u6a21\u5f0f',['\u6b77\u53f2\u63a1\u7528\u6aa2\u67e5','\u65b0\u56e0\u5b50\u8a66\u9a57\u6392\u540d'],key='v193_design')
        if design=='\u65b0\u56e0\u5b50\u8a66\u9a57\u6392\u540d':
            from v193_factors import expanded_prior, EXTENDED
            today=panel[(panel.date==asof)&panel.eligible]
            keys=[k for k in FACTORS if k in today and today[k].notna().mean()>=.85 and (cfg.optional_data or k in TECH+EXTENDED)]
            if keys:weights=expanded_prior(h,keys)
            st.warning('\u6b64\u7d44\u662f\u65b0\u56e0\u5b50\u8a66\u9a57\u6b0a\u91cd\uff0c\u672a\u5fc5\u901a\u904e\u63a1\u7528\u9580\u6abb\uff1b\u4e0d\u662f\u66f4\u9ad8\u52dd\u7387\u7684\u4fdd\u8b49\u3002')
    pool_counts=panel[panel.date==asof]
    st.caption(f"本日有資料 {pool_counts.ticker.nunique()-int((pool_counts.ticker==cfg.benchmark).any())} 檔個股；通過歷史長度／流動性／價格異常檢查 {int(pool_counts.eligible.sum())} 檔。這是已載入股票池，不冒稱全市場。")
    if cfg.factor_design in ('ten_angle_v197','balanced_v198'):
        valid=pool_counts.loc[pool_counts.eligible,'p197_market_breadth'].dropna()
        regimes=pool_counts.loc[pool_counts.eligible,'p197_market_gate']
        gate=regimes.mode().iat[0] if not regimes.empty else 'UNKNOWN'
        st.caption('市場背景（與選股分開）：'+str(gate)+'｜已載入股票池站上60日線占比 '+(f'{valid.iloc[0]:.0%}' if len(valid) else '資料不足')+'。市場廣度不是全市場資料；不等於盤中已確認買點。')
    mode=st.selectbox('機會類型',['全部研究候選','業績＋回檔','低檔修復','突破加速'],key='v19_setup')
    codes={'業績＋回檔':'GROWTH_PULLBACK','低檔修復':'REVERSAL','突破加速':'BREAKOUT'}
    rows=current_candidates(panel,scoring_cfg,weights,h,setup=codes.get(mode))
    if rows.empty:
        st.info('這個類型目前沒有通過條件的候選。可能是條件未成立或基本面證據不足，不等於市場沒有機會。')
    for i,(_,row) in enumerate(rows.head(5).iterrows(),1):
        candidate_card(row,analog_summary(panel,row,h,weights),h,weights,i)
    with st.expander('本次因子權重與使用方式',expanded=False):
        st.dataframe(pd.DataFrame([{'因子':LABELS.get(k,k),'權重％':round(v*100,2)} for k,v in weights.items()]),hide_index=True,use_container_width=True)
        st.write('一、研究分是相對排序，不是上漲機率。二、確認型態、量價與當日資料後才評估進場。三、用預定損失設定部位；停損不保證跳空時能成交。四、每次研究使用相同成本與股票池，與固定權重及0050比較。')
    with st.expander('資金配置試算（不是自動下單）',expanded=False):
        capital=st.number_input('可投入資金（元）',min_value=1000.,value=100000.,step=10000.,key='v19_capital')
        risk=st.slider('每筆規劃損失／資金（％）',.1,2.,.5,.1,key='v19_risk')
        entry=st.number_input('預計買進價',min_value=.01,value=100.,key='v19_entry')
        stop=st.number_input('預定防守價',min_value=.01,value=95.,key='v19_stop')
        sized=position_size(capital,entry,stop,risk/100)
        if sized['shares']:
            st.write(f"參考上限：{sized['shares']:,} 股；部位約 {number(sized['position_value'],0)} 元；含成本緩衝的規劃損失約 {number(sized['planned_max_loss'],0)} 元。")
        else:st.info('請確認防守價低於買進價，且輸入值合理。')
        st.caption('單股上限20％，可含零股；未處理實際最低手續費。跳空、流動性及滑價可能使損失超過規劃。')

def page_validation(panel,report,cfg):
    st.subheader(f'{cfg.dates}個基準日：先封存選股，再看後續表現')
    st.write(f'設定：{cfg.sampling_method}不放回抽樣，日期種子{cfg.seed}。權重僅使用基準日以前已完成的績效；訓練與內部驗證依時間分離。')
    st.caption('持有期為交易日：基準日T計算 → T+1開盤假設進場 → T+5／20／60收盤評價。同步匯出相對T日收盤的價格變動。未成交名額留現金，不以其他成功股票替補。')
    if not report:
        st.info('尚無已完成研究。可按「下載近年資料＋執行驗證」，或用已有快取執行。缺籌碼不會阻止價量因子的驗證。')
        return
    st.write(f"研究狀態：**{report.get('status')}**｜資料 {report.get('data_start')}～{report.get('data_end')}｜股票／基準合計 {report.get('symbols')} 檔")
    st.caption('目前資料池未完整驗證歷史下市成分；仍有存活者偏差。若來源為未還原價格，結果是扣成本的價格報酬，不等同含息總報酬。')
    universe_stats=report.get('universe_diagnostics') or {}
    if universe_stats:
        quality=universe_stats.get('universe_quality') or {}
        st.write('母池覆蓋：實際驗證股票 '+str(universe_stats.get('tested_tickers'))+' 檔｜當日可選 '+str(universe_stats.get('today_eligible'))+' 檔；上市／上櫃 '+str(universe_stats.get('tested_market')))
        if quality:
            st.caption('目前目錄 '+str(quality.get('catalog_size'))+' 檔、選取 '+str(quality.get('selected_size'))+' 檔、完成下載 '+str(quality.get('downloaded_size'))+' 檔（'+format(quality.get('download_coverage',0),'.1%')+'）。這是當前成分股的覆蓋，不是已核實的歷史存活者完整母體。')
            if 'warmup_eligible_symbols' in quality:
                st.caption('其中歷史K線至少'+str(cfg.warmup)+'筆：'+str(quality.get('warmup_eligible_symbols'))+'檔（'+format(quality.get('warmup_eligible_fraction',0),'.1%')+'）；至少252筆：'+str(quality.get('has_252_bar_history_symbols'))+'檔。下載成功不等於有足夠完整的歷史可測。')
            with st.expander('產業／市場覆蓋明細（下載母體稽核）'):
                st.dataframe(pd.DataFrame(quality.get('sector_comparison') or []),use_container_width=True,hide_index=True)
    kind=report.get('data_kind','UNCLASSIFIED')
    st.caption('資料分類：'+kind+'；RESEARCH_COMPLETE只表示本次資料的計算流程完成，不代表最佳策略已獲證實。')
    if kind=='SYNTHETIC_TEST_ONLY':
        st.warning('此為合成測試資料，不能當作真實市場績效。')
    validation_overview(report)
    broader_results(report)
    report_supports(report,LABELS)
    factor_section(report,LABELS)
    counts=report.get('fit_status_counts') or {}
    if counts:
        with st.expander('各次權重調整結果',expanded=False):
            st.json(counts)
    from v196_factors import simple_summary,profile_label
    simple=simple_summary(report)
    if simple:
        with st.expander('\u7c21\u55ae\u5c0d\u7167\u7d44\u6709\u6c92\u6709\u52dd\u904e\u8907\u96dc\u6a21\u578b\uff1f',expanded=False):
            st.caption('\u9019\u4e9b\u662f\u56fa\u5b9a\u6b0a\u91cd\u5c0d\u7167\uff0c\u4e0d\u662f\u4e8b\u5f8c\u6311\u9078\u7684\u6b63\u5f0f\u63a8\u85a6\u3002\u6bcf\u5217\u53ea\u6bd4\u8f03\u540c\u65e5\u3001\u540c\u50f9\u683c\u57fa\u790e\u7684\u6709\u6548\u7d50\u679c\uff0c\u975e\u7d2f\u7a4d\u5831\u916c\u3002')
            st.dataframe(pd.DataFrame([{'\u6301\u6709\u65e5':x['horizon'],'\u56fa\u5b9a\u5c0d\u7167':profile_label(x['profile']),'\u914d\u5c0d\u6578':x['paired_cases'],'\u5e73\u5747\u6de8\u5831\u916c':percent(x['net']),'0050\u540c\u65e5':percent(x['benchmark_net']),'\u5dee\u7570':percent(x['excess'])} for x in simple]),hide_index=True,use_container_width=True)
    if cfg.factor_design in ('ten_angle_v197','balanced_v198'):
        from v197_selection import ten_summary
        from v197_ten_pillars import LENSES
        with st.expander('十面向研究：哪些訊號有效？',expanded=True):
            st.caption('市場狀態與可成交性是風控，不當成各股票加分；產業成分未歷史核實者僅研究。缺財報或公告時點時不補假數據。')
            coverage=report.get('ten_pillar_coverage',[])
            if coverage:st.dataframe(pd.DataFrame([{'因子':x['name'],'資料覆蓋率':f"{x['coverage']:.0%}",'證據註記':x['reason']} for x in coverage]),hide_index=True,use_container_width=True)
            d=report.get('ten_pillar_shadow_summary') or ten_summary(report)
            if d:
                compare=pd.DataFrame([{'週期':x['horizon'],'固定假說':x['profile'],'同期有效組數':x['paired_cases'],'淨報酬':percent(x['mean_net']),'舊模型':percent(x['legacy_mean']),'0050':percent(x['benchmark_mean']),'相對0050':percent(x['vs_0050'])} for x in d])
                st.dataframe(compare,hide_index=True,use_container_width=True)
            st.caption('固定假說只是研究對照。這批100個日期已被反覆分析，不是新的盲測；不同週期的可比較組數不同。')
    records=pd.DataFrame(report.get('results',[]))
    if not records.empty:
        table=records.pivot_table(index=['date','horizon'],columns='strategy',values='mean_net',aggfunc='first',dropna=False).reset_index()
        table=table.rename(columns={'date':'基準日','horizon':'持有交易日','baseline':'固定權重','adaptive':'逐步調整權重',
                                    'benchmark_0050':'0050同口徑','equal_weight_universe':'股票池等權','guarded':'證據門檻／現金',
                                    'training_challenger_research':'事前訓練候選（未必採用）',
                                    'expanded_fixed_research':'舊擴充固定試驗組' })
        for c in ('固定權重','逐步調整權重','0050同口徑','股票池等權','證據門檻／現金','事前訓練候選（未必採用）','舊擴充固定試驗組'):
            if c in table:table[c]=table[c].map(percent)
        for c in list(table.columns):
            if str(c).startswith('simple_control_'):
                table[c]=table[c].map(percent)
                table=table.rename(columns={c:profile_label(c)})
        for c in list(table.columns):
            if str(c).startswith('ten_shadow_'):
                table[c]=table[c].map(percent)
                table=table.rename(columns={c:'十面向研究：'+c.removeprefix('ten_shadow_')})
        st.dataframe(table,hide_index=True,use_container_width=True)
        st.caption('「—」代表該組不完整，不是報酬0。60日結果可能跨除權息、分割或停牌；異常未還原價格區間不拿來證明策略勝出。')
    with st.expander('每次訓練、排除因子與完整來源紀錄',expanded=False):
        st.json({'overlap':report.get('overlapping_horizons'), 'sampling_plan':report.get('sampling_plan'),
                 'resumed_cases':report.get('resumed_cases'), 'adjusted_bar_fraction':report.get('adjusted_bar_fraction')})
        st.caption('完整逐輪權重與來源請下載JSON或診斷包，避免手機一次載入數千筆訓練紀錄。')
    pointer=read_json(DATA/'last_success.json')
    directory=Path(pointer['directory']) if pointer.get('directory') else None
    if directory and directory.resolve().is_relative_to((DATA/'runs').resolve()):
        for fn,label,mime in [('v198_diagnostic_bundle.zip','下載V19.8分層母池完整驗證包（股價／因子／權重）','application/zip'),
                              ('v197_diagnostic_bundle.zip','下載V19.7相容格式資料包','application/zip'),
                              ('sparse_factor_selection.csv','下載每次因子去留與權重','text/csv'),
                              ('sparse_latest_weights.csv','下載各週期最新候選權重','text/csv'),
                              ('factor_validation_summary.csv','下載因子效果總表','text/csv'),
                              ('factor_validation_cases.csv','下載每日每因子驗證','text/csv'),
                              ('sampling_plan.json','下載抽樣日期與種子','application/json'),
                              ('v191_diagnostic_bundle.zip','下載相容格式診斷包','application/zip'),
                              ('comparison.csv','下載全部基準日比較表','text/csv'),
                              ('selected_trades.csv','下載每檔選股／買賣基準／結果','text/csv'),
                              ('v19_results.json','下載完整結果與每輪權重','application/json'),
                              ('v19_validation.md','下載研究報告','text/markdown')]:
            path=directory/fn
            if path.exists():st.download_button(label,path.read_bytes(),file_name=fn,mime=mime,key='dl_'+fn)
    with st.expander('逐欄位檢查（全部已載入股票）',expanded=False):
        health=pd.DataFrame(field_health(panel,report))
        if 'field' in health:health['field']=health.field.map(lambda x:LABELS.get(x,x))
        st.dataframe(health.rename(columns={'field':'欄位','present':'有值筆數','total':'母數','coverage':'有值比例','state':'狀態'}),hide_index=True,use_container_width=True)
        st.caption('有值比例不等於資料正確率；頸線空值也可能表示沒有W底。營收、法人、大戶等缺值不會補成0或捏造連買日數。')

@st.fragment(run_every=20)
def live_tile(ticker):
    from v19_live import observation
    quote=observation(ticker)
    if quote.get('pending') and quote.get('price') is None:
        st.info('正在查詢當日行情；日線分析仍可使用，不需要等待整頁重新計算。')
    if quote.get('price') is not None:
        st.metric('當日觀察價',number(quote['price']),delta=percent(quote.get('change_pct'),1))
        st.caption(f"來源 {quote.get('source','—')}｜成交／分鐘時間 {quote.get('timestamp','—')}｜{quote.get('age_label','—')}")
        if quote.get('observed_black_to_red'):st.info('本次追蹤已觀察到由昨收下方轉至上方；這是已發生的現象，不是事前預測命中。')
    else:
        st.warning('當日成交價尚未取得：'+str(quote.get('reason') or '來源仍查詢中'))
    st.caption('盤中與日線日期分開。第三方分鐘價、過期官方價不冒充可成交買點；V19新權重尚不自動發布下單指令。')
    with st.expander('盤中來源診斷',expanded=False):
        st.json(quote)

def page_doctor(panel,report,cfg):
    st.subheader('個股診斷：日線結構與當日行情分開')
    if panel.empty:
        st.info('請先下載歷史日線，或使用原有V18資料庫。')
        return
    names=panel.sort_values('date').drop_duplicates('ticker',keep='last').set_index('ticker')['name'].to_dict()
    tickers=sorted(names)
    default=tickers.index('2330.TW') if '2330.TW' in tickers else 0
    ticker=st.selectbox('已載入股票',tickers,index=default,format_func=lambda t:f"{names[t]} {t}",key='v19_doctor_ticker')
    h=st.radio('評估週期',list(cfg.horizons),horizontal=True,format_func=lambda x:f'{x}交易日',key='v19_doctor_h')
    stock=panel[panel.ticker==ticker].sort_values('date');row=stock.iloc[-1]
    weights,state=weights_for(report,h,row.date)
    scoring_cfg=cfg
    if cfg.factor_design in ('sparse_v194','sparse_v195','tournament_v196','ten_angle_v197','balanced_v198'):weights,scoring_cfg=choose_model(report,h,cfg,row.date,LABELS,key='v194_doctor_mode')
    # Display does not require a stock to pass the candidate filter.
    from v19_features import score_rows
    scored=stock.tail(1).copy();scored['score'],scored['factor_coverage']=score_rows(scored,weights,min_coverage=1.-1e-9 if scoring_cfg.factor_design in ('sparse_v194','sparse_v195','tournament_v196','ten_angle_v197','balanced_v198') else .75)
    observed=scored.iloc[-1].to_dict()
    observed['setup']='WATCH';observed['stop_reference']=max(float(row.support10),float(row.close-1.8*row.atr)) if pd.notna(row.atr) and pd.notna(row.support10) else np.nan
    observed['stop_reference']=min(observed['stop_reference'],float(row.close)*.99)
    observed['stop_distance_pct']=(row.close-observed['stop_reference'])/row.close*100
    observed['reference_low']=row.close-.5*row.atr;observed['reference_high']=row.close+.25*row.atr
    observed['chase_ceiling']=row.close+min(row.atr,row.close*.03)
    candidate_card(observed,analog_summary(panel,row,h,weights),h,weights,1)
    st.caption(f"股票最新日線 {str(row.date.date())}；整體模型最新資料 {str(panel.date.max().date())}。")
    if st.checkbox('顯示當日盤中觀察（每20秒檢查）',value=False,key='v19_live_enable'):
        live_tile(ticker)
    data=stock.tail(120)
    fig=go.Figure(go.Candlestick(x=data.date,open=data.open,high=data.high,low=data.low,close=data.close,name='日K'))
    fig.add_trace(go.Scatter(x=data.date,y=data.ma20,name='20日均線'))
    fig.add_trace(go.Scatter(x=data.date,y=data.ma60,name='60日均線'))
    fig.update_layout(height=390,margin=dict(l=8,r=8,t=20,b=10),xaxis_rangeslider_visible=False,legend=dict(orientation='h'))
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(pd.DataFrame([{'指標':LABELS.get(k,k),'數值':number(row.get(k))} for k in ('rsi','k','d','macd_hist','bias20','neckline')]),hide_index=True,use_container_width=True)
    if not bool(row.eligible):st.warning('本檔未通過目前候選池條件；可查看資料，不會自動當成推薦股。')

def news_block():
    with st.expander('可能影響台股：最近72小時新聞與國際摘要',expanded=False):
        from briefing_jobs import load_brief,ensure_brief
        from market_briefing import _only_recent_news
        if st.button('更新新聞／國際摘要',key='v19_news_refresh'):
            ensure_brief(BASE,force=True)
            st.info('已啟動新聞更新；稍後重新開啟此區即可查看。')
        brief=load_brief(BASE);news=_only_recent_news(brief.get('news') or {},now=datetime.now(TZ),max_age_hours=72)
        if not news.get('items'):st.caption('目前沒有可核對發布時間的72小時內新聞；不以舊新聞補滿。')
        for row in (news.get('items') or [])[:10]:
            st.markdown('**'+str(row.get('title',''))+'**')
            st.caption(str(row.get('published_iso',''))+'｜'+str(row.get('source','')))
            if row.get('summary'):st.write(row['summary'])
            if row.get('impact'):st.write(row['impact'])
            if str(row.get('link','')).startswith(('https://','http://')):st.link_button('來源文章',row['link'])

def main(configure_page=True):
    # app.py sets page config before calling this in Cloud; standalone callers
    # can still ask this function to configure its own page exactly once.
    if configure_page:
        st.set_page_config(page_title='Alpha Radar V19.8.2',page_icon='📈',layout='wide',initial_sidebar_state='collapsed')
    _bootstrap();st.markdown(CSS,unsafe_allow_html=True)
    st.title('Alpha Radar V19.8')
    st.caption('近年日線驗證 × 可追溯權重 × 投資規劃｜研究版，非保證獲利或自動交易')
    with st.sidebar:
        st.subheader('資料與測試設定')
        years=st.selectbox('歷史區間（年）',[1,2,3],index=2,key='v19_years')
        n=st.selectbox('股票池上限（上市＋上櫃，歷史行情一次下載）',[100,300,600,1000,1800,2500],index=2,key='v19_stocks')
        universe_mode=st.selectbox('股票母池選取方式',['balanced','proportional','full_current','legacy_random'],index=0,
            format_func=lambda x:{'balanced':'上市／上櫃×產業分層，兼顧多樣性（建議）','proportional':'按當前產業家數比例抽樣','full_current':'盡量下載目前完整上市櫃清單','legacy_random':'舊版純隨機比較（不建議）'}[x],key='v198_pool_mode')
        st.caption('600檔是選取目標，不是已下載數。完整市場模式可能超過千檔，需分批續跑；目前存續公司名錄不是歷史完整母體。可將具來源的 historical_universe.csv 放在 data/v19 匯入歷史公司，但須自行核實完整性。')
        end=st.date_input('資料截止日',value=datetime.now(TZ).date()-timedelta(days=1),
                          max_value=datetime.now(TZ).date()-timedelta(days=1),key='v19_end')
        extra=st.checkbox('第二階段：補抓營收／法人／融資（較耗額度）',value=False,key='v19_optional')
        st.caption('建議第一輪先下載600檔價量行情，待下載率足夠後，再開啟選配資料分批補抓。未勾選時不能評斷營收與法人因子效果；無公告時點的季報在嚴格模式不計分。')
        st.markdown('#### \u56e0\u5b50\u9078\u64c7\u8a2d\u5b9a')
        design=st.selectbox('\u7814\u7a76\u5f15\u64ce',['balanced_v198','ten_angle_v197','tournament_v196','sparse_v195'],format_func=lambda x:'V19.8 分週期少數因子＋母池代表性' if x=='balanced_v198' else 'V19.7 十面向研究＋過去區塊決策驗證' if x=='ten_angle_v197' else 'V19.6.1 \u7c21\u55ae\u5c0d\u7167\u8207\u4e8b\u524d\u6c7a\u7b56\u6aa2\u67e5' if x=='tournament_v196' else 'V19.5 \u7a00\u758f\u56e0\u5b50\u641c\u5c0b\uff08\u4fdd\u7559\u6bd4\u8f03\uff09',key='v196_engine')
        st.caption('V19.8保留十面向的可用資料限制，新增上市／上櫃×產業分層母池，以及5／20／60日不同的少因子研究組合；只有通過既定歷史門檻才可採用。下方稀疏模型因子數與搜尋預算僅供V19.5引擎使用。')
        max_factors=st.selectbox('\u6700\u591a\u5e7e\u500b\u5e95\u5c64\u56e0\u5b50',[2,3,4,5,6],index=4,key='v194_max_factors')
        search_budget=st.selectbox('\u6bcf\u6b21\u641c\u5c0b\u7d44\u5408\u4e0a\u9650',[64,96,160,256],index=2,key='v194_search_budget')
        finance=st.checkbox('\u4e26\u4e0b\u8f09\u5b63\u5831\u8207\u672c\u76ca\u6bd4\uff08\u9700\u66f4\u591a\u984d\u5ea6\uff09',value=False,key='v194_finance')
        financial_timing=st.selectbox('\u5b63\u5831\u6216\u71df\u6536\u7f3a\u516c\u544a\u6642\u9ede', ['disclosure_required','lagged_research'],
            format_func=lambda x:'\u56b4\u683c\uff1a\u53ea\u5feb\u53d6\uff0c\u4e0d\u7d0d\u5165' if x=='disclosure_required' else '\u7814\u7a76\uff1a\u5b63\u5831120\u65e5\uff0f\u71df\u6536\u6708\u6a19\u7c6414\u65e5\uff08\u975e\u771f\u5be6\u516c\u544a\uff09',key='v194_timing')
        st.caption('\u5b63\u5e95\u4e0d\u662f\u516c\u544a\u65e5\u3002120\u65e5\u843d\u5f8c\u662f\u7814\u7a76\u5047\u8a2d\uff0c\u4e0d\u662f\u539f\u59cb\u516c\u544a\u6216\u4fee\u8a02\u7d00\u9304\u3002')
        st.caption('月營收的月初日期不是公告日。嚴格模式會排除無時點證據的舊快取；研究延後情境不能作為實盤採用依據。')
        count=st.number_input('隨機基準日數（不是股票數）',min_value=2,max_value=100,value=100,step=1,key='v192_dates')
        seed=st.number_input('日期隨機種子',min_value=0,max_value=2**31-1,value=1920,step=1,key='v192_seed')
        sampling=st.selectbox('抽樣方式',['uniform','stratified'],index=0,
                              format_func=lambda x:'全期隨機' if x=='uniform' else '分段隨機',key='v192_sampling')
        replay=st.checkbox('改用舊報告10個日期（不執行100點）',value=False,key='v192_replay')
        refresh=st.checkbox('強制重新下載（一般續跑不勾）',value=False,key='v192_refresh')
        budget=st.selectbox('本輪API請求上限（並非來源帳號可用額度）',[600,1200,2500,4000],index=1,key='v192_budget')
        st.caption('股票池使用獨立種子1900；更換日期不會順便換股票。API時數配額仍依來源限制，不會因提高本輪上限而繞過。')
        if extra and int(budget)<(8 if finance else 4)*int(n)+10:
            st.warning('\u672c\u8f2aAPI\u4e0a\u9650\u53ef\u80fd\u4e0d\u8db3\u4ee5\u53d6\u5f97\u5168\u80a1\u7968\u6c60\u7684\u71df\u6536\uff0f\u6cd5\u4eba\uff0f\u878d\u8cc7\u3002\u53ef\u5206\u6279\u7e8c\u4e0b\u8f09\uff1b\u4f86\u6e90\u9650\u6d41\u4ecd\u6703\u505c\u6b62\u3002')
        adjusted=st.checkbox('優先使用還原價（失敗時保留原價並標示）',value=True,key='v191_adjusted')
        legacy=st.checkbox('開啟V18.21原工作台（保留舊模型）',value=False,key='v19_legacy')
    if legacy:
        st.info('以下是保留的V18.21原工作台，不是V19新研究排名。')
        import legacy_app
        legacy_app.main(configure_page=False)
        return
    reference=ROOT/'reference_validation_v191.md'
    if not reference.exists():reference=ROOT/'reference_validation_v190.md'
    replay_dates=()
    if replay and reference.exists():
        from v191_audit import parse_markdown
        replay_dates=tuple(parse_markdown(reference.read_text(encoding='utf-8'))['cutoffs'])
    cfg=Settings(years=years,stocks=n,dates=len(replay_dates) if replay_dates else int(count),
                 seed=int(seed),sampling_method=sampling,as_of=end.isoformat(),optional_data=extra,
                 prefer_adjusted=adjusted,replay_cutoffs=replay_dates,refresh_history=refresh,request_budget=int(budget),
                 universe_mode=universe_mode,
                 sparse_max_inputs=int(max_factors),sparse_budget=int(search_budget),
                 financial_history=finance,financial_timing=financial_timing,factor_design=design)
    st.caption(f'本次設定：目標母池 {n} 檔（{universe_mode}），{cfg.dates} 個基準日，5／20／60交易日。下載一次歷史資料後，以本地快取重複計算；未達80%下載率或600檔目標的80%時會保留快取、提示續跑，不把少數股票冒稱完整母池。')
    if replay_dates and years<3:
        st.warning('附件日期含2024年。重播請選3年資料，或關閉重播改用新抽樣。')
    c1,c2,c3=st.columns(3)
    with c1:
        if st.button(f'線上下載＋{cfg.dates}點驗證',type='primary',key='v19_start'):
            st.info(str(start_job(DATA,cfg,'download_validate').get('state')))
    with c2:
        if st.button('只用既有快取重新驗證',key='v19_cached'):
            st.info(str(start_job(DATA,cfg,'cache_validate').get('state')))
    with c3:
        if st.button('只下載歷史資料庫',key='v192_download_only'):
            st.info(str(start_job(DATA,cfg,'download_only').get('state')))
    progress_box()
    database_download(DATA)
    import_evidence_control(DATA)
    # The original app eagerly built the ENTIRE 3-year 600-stock factor panel
    # before the user selected a page.  This can exceed free Cloud RAM.  Only
    # load the recent UI panel after an explicit click; batch backtests continue
    # to use the full historical database through the job runner.
    if st.button('載入最新選股與個股分析（節省記憶體）', key='ar_cloud_load_panel'):
        st.session_state['ar_cloud_allow_panel'] = True
    if cfg.stocks > 1000 and st.session_state.get('ar_cloud_allow_panel', False):
        st.error('目前設定超過1000檔。Cloud前端不直接建構超大型面板；請改為600檔展示，完整歷史研究可用資料庫／離線工具執行。')
        return
    if not st.session_state.get('ar_cloud_allow_panel', False):
        last = read_last_report(DATA)
        st.info('資料與設定控制已就緒。需要查看即時候選或個股圖表時，請按「載入最新選股與個股分析」。完整歷史驗證由上方任務執行，無須首頁先載入全部K線。')
        if last:
            st.caption('已有歷史研究報告；可選擇上方任務重新計算，或再載入分析畫面。')
        return
    try:
        panel,_=_display_data(str(DATA),_mtime(DATA/'history_prices.csv.gz'),
                             max(_mtime(DATA/'historical_evidence.csv.gz'),_mtime(DATA/'optional_evidence.csv')),_mtime(BASE/'daily_prices.sqlite'),cfg)
        report=read_last_report(DATA)
    except Exception as exc:
        panel=pd.DataFrame();report=read_last_report(DATA)
        st.error('資料檢查未通過，未填入假數字：'+type(exc).__name__+': '+str(exc))
    view=st.radio('功能',['選股與規劃','歷史驗證','個股診斷'],horizontal=True,label_visibility='collapsed',key='v19_view')
    if view=='選股與規劃':page_decisions(panel,report,cfg);news_block()
    elif view=='歷史驗證':page_validation(panel,report,cfg)
    else:page_doctor(panel,report,cfg)
    st.caption('公開日線可以先驗證技術策略；歷史營收、籌碼與大戶資料只有在發布時間可用時才加入。100個隨機基準日仍是開發診斷，不足以證明全域最佳權重。')
