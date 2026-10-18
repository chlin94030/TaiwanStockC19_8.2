"""Mobile-first explanation of the V18 evidence-aware factor engine.

Streamlit remains the interaction layer; HTML is only for compact, escaped
static scorecards.  No provider calls or expensive 2k-stock calculations.
"""
from __future__ import annotations

import html
import math

import pandas as pd

from v18_multifactor import FACTOR_GROUPS
from v187_experience import signal_light

GROUP_LABEL = {
    "quality": ("01", "財報品質", "#4462cf", "blue"),
    "chip": ("02", "籌碼動向", "#8b57bf", "violet"),
    "momentum": ("03", "技術動能", "#0d8877", "teal"),
    "sector": ("04", "族群共振", "#d18b38", "amber"),
    "risk": ("05", "風險防守", "#ca5266", "rose"),
}

CSS = """
<style>
/* V18 visual vocabulary: factor colors encode dimensions, not arbitrary rank. */
.factor-hero{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:18px;
background:linear-gradient(112deg,#142547,#243b65 61%,#4b3972);color:#fff;
border-radius:17px;padding:20px 23px;margin:9px 0 15px;box-shadow:0 16px 30px rgba(31,45,85,.14)}
.factor-hero h2{font-size:1.6rem;letter-spacing:-.035em;margin:3px 0 8px;color:#fff;font-weight:950}
.factor-hero p{font-size:.87rem;color:#e0e9fa;line-height:1.55;margin:0}
.factor-version{font-size:.75rem;color:#a8c7fa;font-weight:850;letter-spacing:.08em}
.factor-hero-score{text-align:right;font-variant-numeric:tabular-nums;min-width:100px}
.factor-hero-score strong{display:block;font-size:2.8rem;line-height:1;font-weight:950;color:#fff}
.factor-hero-score span{font-size:.74rem;color:#c5d6fb}
.factor-stats{display:grid;grid-template-columns:repeat(3, minmax(0,1fr));gap:9px;margin:10px 0}
.factor-stat{background:#fff;border:1px solid #dfe5f1;border-radius:12px;padding:11px 13px;min-width:0}
.factor-stat small{display:block;color:#61708e;font-size:.73rem;font-weight:800}
.factor-stat b{display:block;color:#20355d;font-size:1.26rem;font-weight:950;margin-top:4px;font-variant-numeric:tabular-nums}
.factor-stat em{display:block;color:#61708e;font-size:.70rem;font-style:normal}.factor-stat strong{display:block;font-size:.79rem;margin:4px 0;font-weight:900}.factor-light-green{background:#e8f6ed;border-color:#aed9c0}.factor-light-amber{background:#fff8e9;border-color:#e5d3ad}.factor-light-red{background:#fff1ef;border-color:#eac0b8}.factor-light-gray{background:#f4f6f9;border-color:#dae0ec}
.factor-stat.alpha{border-top:4px solid #256dca}.factor-stat.quality{border-top:4px solid #16a085}.factor-stat.confidence{border-top:4px solid #be7b29}
.factor-state{font-size:.77rem;color:#5d6980;margin-top:5px;line-height:1.35}.factor-status-note{font-size:.78rem;line-height:1.65;padding:11px 13px;border:1px solid #e2e7f2;background:#fff;border-radius:11px;margin:11px 0}
.factor-stat-note{font-size:.77rem;color:#596c8e;line-height:1.6;margin-top:6px}
.factor-rows{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:12px 0}
.factor-group{background:#fff;border:1px solid #dfe5f0;border-radius:13px;padding:14px 16px;min-width:0;box-shadow:0 4px 15px rgba(40,59,109,.035)}
.factor-head{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:12px}
.factor-head b{font-weight:950;font-size:1rem;color:#203452}
.factor-head strong{font-size:1.24rem;font-variant-numeric:tabular-nums;color:#233f68}
.factor-meter{height:8px;background:#edf1f7;border-radius:999px;overflow:hidden;margin:8px 0}
.factor-meter div{height:100%;border-radius:999px}
.factor-summary{font-size:.77rem;color:#61708a;line-height:1.5;margin-top:7px}
.factor-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.factor-cell{display:flex;justify-content:space-between;align-items:center;gap:8px;background:#f7f9fd;border:1px solid #e8edf6;border-radius:8px;padding:9px 10px;min-height:48px}
.factor-cell b{font-size:.78rem;color:#314461;font-weight:840}
.factor-cell strong{font-size:.90rem;color:#233b69;font-variant-numeric:tabular-nums}
.factor-cell.missing{background:#fbfaf8;border-style:dashed}.factor-cell.missing strong{color:#9299aa}
.factor-risk{background:#fff8ef;border:1px solid #f2dfc5;border-left:5px solid #d28f37;border-radius:11px;padding:11px 13px;color:#765426;line-height:1.55;font-size:.85rem;margin:12px 0}
.factor-explain{background:#f0f5ff;border:1px solid #dfe8fc;border-radius:10px;padding:10px 13px;color:#354b74;font-size:.82rem;line-height:1.55;margin:10px 0}
.pick-mini-factor{grid-column:1/-1;display:flex;gap:5px;flex-wrap:wrap;align-items:center;font-size:.73rem;padding-top:1px}
.pick-mini-factor .f-pill{display:inline-flex;border-radius:6px;border:1px solid #dde5f2;background:#f5f7fd;padding:3px 7px;color:#4a5c80;font-weight:800}
.pick-mini-factor .f-pill.alpha{border-color:#d9cefa;background:#f3efff;color:#6645a6}
.pick-mini-factor .f-pill.stage{border-color:#cce9e3;background:#effaf7;color:#167367}
.pick-mini-factor .f-pill.low{border-color:#f1d9c5;background:#fff5ec;color:#a66322}
@media(max-width:720px){
.factor-hero{padding:17px;gap:8px}.factor-hero h2{font-size:1.32rem}.factor-hero p{font-size:.79rem}
.factor-hero-score strong{font-size:2.20rem}.factor-hero-score{min-width:60px}
.factor-stats{grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}.factor-stat{padding:10px}
.factor-rows{grid-template-columns:1fr}.factor-group{padding:12px}.factor-grid{gap:6px}
.factor-cell{padding:8px 9px}.factor-cell b{font-size:.74rem}.factor-cell strong{font-size:.80rem}
.pick-mini-factor{margin:2px 0 0}
}
@media(max-width:390px){.factor-hero-score strong{font-size:1.93rem}.factor-hero h2{font-size:1.17rem}}
</style>
"""


def _esc(s) -> str:
    return html.escape(str(s if s is not None else "—"), quote=True)


def _metric(v, digits=0) -> str:
    try:
        x = float(v)
        return f"{x:.{digits}f}" if math.isfinite(x) else "—"
    except (TypeError, ValueError):
        return "—"


def _factor_grid(profile: dict) -> str:
    factors = profile.get("factors") or {}
    blocks = []
    for group, fields in FACTOR_GROUPS.items():
        _, label, color, _ = GROUP_LABEL[group]
        cells = []
        observed_count = sum((factors.get(key) or {}).get('score') is not None for key,_ in fields)
        positive_count = len(fields)
        for key, name in fields:
            factor = factors.get(key) or {}
            score = factor.get("score")
            tip = factor.get("note") or ""
            tip_text = f"{tip} · 來源：{factor['source']}" if factor.get("source") else tip
            tip_text += f" · 狀態：{factor.get('status') or '待核對'}"
            if factor.get('first_seen_at'): tip_text += f" · 系統首次取得：{factor['first_seen_at']}"
            cells.append(f"<div class='factor-cell {'missing' if score is None else ''}' title='{_esc(tip_text)}'>"
                         f"<b>{_esc(name)}</b><strong>{_metric(score) if score is not None else '缺資料'}</strong></div>")
        if group == "risk":
            heading = f"<b style='color:{color}'>{_esc(label)} · 不扣選股分</b><strong>風險參考</strong>"
            scale = ""
        else:
            value = float((profile.get("dimensions") or {}).get(group, 50))
            heading = f"<b style='color:{color}'>{_esc(label)} <small style='font-size:.68rem;color:#697994'>{observed_count}/{positive_count}項有數值</small></b><strong>{value:.0f}<small>/100</small></strong>"
            scale = f"<div class='factor-meter'><div style='width:{max(0,min(100,value)):.0f}%;background:{color}'></div></div>"
        blocks.append(f"<section class='factor-group'><div class='factor-head'>{heading}</div>{scale}"
                      f"<div class='factor-grid'>{''.join(cells)}</div></section>")
    return "<div class='factor-rows'>" + "".join(blocks) + "</div>"


def render_page(st, snap: dict | None, global_context: dict | None = None, *,
                selected_ticker: str | None = None, selected_horizon: str | None = None,
                compact: bool = False):
    """Explain one selected stock inline or expose the legacy full selector."""
    if not compact:
        st.markdown("<div class='section-kicker'>EVIDENCE LAB</div><div class='section-title'>因子透視｜資料與選股理由</div>", unsafe_allow_html=True)
    if not snap:
        st.info("尚無 V18 完整日線模型；可先查看國際市場，背景更新完成後即能比較因子。")
        return
    if not str(snap.get("model_version") or "").startswith("v18"):
        st.warning("這仍是 V17 舊快照，必須執行一次 V18 日線更新才有可比因子資料。")
        return
    stats = snap.get("v18_factor_summary") or {}
    st.markdown(f"<div class='factor-explain'><b>資料可信度優先</b>：已完成模型日 {_esc(snap.get('price_date'))}；"
                f"全市場平均有數值因子 {_metric(stats.get('avg_observed_factors'),1)} / 13 項（不代表已通過時間核對）。"
                "另外 2 項為進場風險；新因子僅依已驗證證據參與分數融合。</div>", unsafe_allow_html=True)
    h = selected_horizon if selected_horizon in ("short","mid","long") else st.radio(
        "分析週期", ["short", "mid", "long"],
        format_func=lambda x: {"short":"短線｜1週（5交易日）", "mid":"中線｜約2個月", "long":"長線｜約6個月"}[x],
        horizontal=True, key="v18_factor_period")
    # Fast view: use completed snapshot only; no per-stock history fetch.
    candidates = [s for s in snap.get("stocks", []) if ((s.get("horizons") or {}).get(h) or {}).get("rank_v18") and s.get("v18_profile")]
    candidates.sort(key=lambda s: float((s["horizons"][h]).get("ranking_score") or 0), reverse=True)
    if not candidates:
        st.info("本次快照沒有可顯示的 V18 因子。")
        return
    if selected_ticker:
        ticker = str(selected_ticker)
        stock = next((s for s in candidates if s.get("ticker") == ticker), None)
        if stock is None:
            st.info("此股票暫無完整因子證據；可在背景模型更新後再次查看。")
            return
    else:
        limit = st.select_slider("股票列表範圍", options=[10,25,50,100,200], value=50, key="v18_factor_limit")
        ticker = st.selectbox("選一檔看完整評分依據", [s["ticker"] for s in candidates[:int(limit)]],
                              format_func=lambda tick: f"{next((x.get('name') for x in candidates if x.get('ticker')==tick),tick)} · {tick}",
                              key="v18_selected_ticker")
        stock = next(s for s in candidates if s["ticker"] == ticker)
    profile = stock["v18_profile"]
    block = stock["horizons"][h]
    ranked = block.get("rank_v18") or {}
    overall = float(block.get("ranking_score") or 0)
    observed = int(profile.get("evidence_n") or 0)
    total = int(profile.get("evidence_total") or 13)
    st.markdown(f"<div class='factor-hero'><div><div class='factor-version'>ALPHA RADAR · EVIDENCE AWARE · {h.upper()}</div>"
                f"<h2>{_esc(stock.get('name'))} <small style='font-size:.72rem;color:#b9c9e4'>{_esc(ticker)}</small></h2>"
                f"<p>{_esc(stock.get('fine_industry') or stock.get('industry'))} · {_esc(profile.get('stage'))} · "
                f"有數值因子 {observed}/{total} · 模型日 {_esc(profile.get('date'))}</p></div>"
                f"<div class='factor-hero-score'><strong>{overall:.1f}</strong><span>選股評分（非買點）</span></div></div>", unsafe_allow_html=True)
    confidence = ranked.get('confidence') or {}
    actually_usable = sum(float(v or 0)>0 for v in (confidence.get('factor_reliabilities') or {}).values())
    eligibility = ranked.get('eligibility') or {}
    dq, mc = ranked.get('dq_score'), ranked.get('mc_score')
    score_values = [
        ('Alpha｜選股吸引力', _metric(overall,1), '投資研究分數，非預測勝率', 'alpha'),
        ('DQ｜資料品質', _metric(dq,1), '依本週期因子加權與資料時效', 'quality'),
        ('MC｜模型信心', _metric(mc,1), '歷史案例＋全市場排名穩健度', 'confidence'),
    ]
    states = [signal_light(overall,'alpha'),signal_light(dq,'dq'),signal_light(mc,'mc')]
    st.markdown("<div class='factor-stats'>" + "".join(
        f"<div class='factor-stat {cls} factor-light-{state['color']}'>"
        f"<small>{state['icon']} {_esc(label)}</small><b>{_esc(val)}</b>"
        f"<strong>{_esc(state['label'])}</strong><em>{_esc(note)}</em></div>"
        for (label,val,note,cls),state in zip(score_values,states)) + "</div>", unsafe_allow_html=True)
    st.caption('燈號只代表各自分數：Alpha＝研究優先性；DQ＝資料完整程度；MC＝歷史證據與穩健性。綠燈不是立即買進，須以官方即時價核對買點。')
    status = str(eligibility.get('status_label') or '模型資料待更新')
    reasons = eligibility.get('failed_reasons') or []
    detail = '；'.join(str(x) for x in reasons[:3]) if reasons else '此檔目前符合初始研究證據門檻；仍須獨立確認買點、價格與下跌風險'
    st.markdown(f"<div class='factor-status-note'><b>{_esc(status)}</b>｜{_esc(detail)}"
                f"<div class='factor-stat-note'>V17 原分 {_esc(_metric(ranked.get('score_v17'),1))} → 新因子 {_esc(_metric(ranked.get('score_v18'),1))}"
                f" → 有效融合 {_esc(_metric(100*float(ranked.get('overlay_weight') or 0),1))}%  · "
                f"覆蓋率 {_esc(_metric(ranked.get('coverage_pct'),1))}%  · 可採信因子 {actually_usable}/{total}  · "
                f"獨立證據來源 {int(confidence.get('independent_categories_count') or 0)} 類</div></div>", unsafe_allow_html=True)
    missing_factors = [label for group,fields in FACTOR_GROUPS.items() if group != 'risk'
                       for key,label in fields if (profile.get('factors',{}).get(key) or {}).get('score') is None]
    if missing_factors:
        st.caption(f"尚缺 {len(missing_factors)} 項：" + '、'.join(missing_factors[:6])+
                   ('等。' if len(missing_factors)>6 else '。')+
                   ' 缺資料表示目前未觀測或未取得，不代表該公司基本面差。')
    st.markdown(_factor_grid(profile), unsafe_allow_html=True)
    with st.expander('資料時間、因子可靠度與信心計算依據', expanded=False):
        st.caption('DQ 只用正向13因子；同業/技術同屬價格訊號，不能視為完全獨立的兩類證據。')
        st.caption(f"歷史實證 H：{_metric(confidence.get('historical_score'),1)}；全市場穩健度 S：{_metric(confidence.get('stability_score'),1)}；模型信心 MC：{_metric(mc,1)}。{_esc(eligibility.get('oos_status') or '樣本外績效待驗證')}")
        section = (stock.get('ranking_research') or {})
        rows = []
        for key,title in [('monthly_revenue','營收'),('financials','財報'),('institutional_flow','法人'),('valuation','估值')]:
            pit = (section.get(key) or {}).get('pit') or {}
            rows.append({'來源':title,'PIT狀態':pit.get('status') or '缺',
                         '所屬資料期末':pit.get('period_end') or '—',
                         '首次取得':pit.get('first_seen_at') or '—',
                         '公告時間':pit.get('published_at') or '未證實'})
        for record in rows:
            st.markdown('<div class="factor-status-note"><b>'+_esc(record['來源'])+'</b> · '+_esc(record['PIT狀態'])+
                        '<div class="factor-state">資料期末 '+_esc(record['所屬資料期末'])+
                        ' · 首次取得 '+_esc(record['首次取得'])+
                        ' · 公告 '+_esc(record['公告時間'])+'</div></div>', unsafe_allow_html=True)
        st.caption('無時間戳記不能宣稱以前已知道這筆資料。首次取得時間並不等同於官方公告時間。')
    danger = float(profile.get("risk_penalty_for_entry") or 0)
    dynamic_macro = None
    ctx = global_context or {}
    if ctx.get("available") and not ctx.get("stale"):
        try:
            dynamic_macro = float(ctx.get("shock_score"))
        except (TypeError, ValueError):
            dynamic_macro = None
    dynamic_macro_penalty = min(4, max(0, dynamic_macro - 60) * 0.10) if dynamic_macro is not None else 0.0
    risk_line = (f"完整日線過熱風險扣分 {danger:.1f} 點；" +
                 (f"最新已核對國際壓力 {dynamic_macro:.0f}/100、額外觀察風險 {dynamic_macro_penalty:.1f} 點。" if dynamic_macro is not None
                  else "國際壓力未取得新鮮可驗資料，不擅自設為0或判為安全。"))
    st.markdown(f"<div class='factor-risk'><b>風控只看進場：</b>{_esc(risk_line)} "
                "風控分數是輔助觀察，正式買點仍以『選股與買點』的價格、剩餘報酬及官方行情核對為準。</div>", unsafe_allow_html=True)
    if not compact:
        with st.expander("看全市場比較：哪些股票因 V18 而升降？", expanded=False):
            rows = []
            for item in candidates[: min(150, len(candidates))]:
                b = item["horizons"][h]
                rank = b.get("rank_v18") or {}
                if not rank:
                    continue
                rows.append({"股票":item.get("name"), "代號":item.get("ticker"), "最新綜合分":rank.get("score"),
                             "V17原分":rank.get("score_v17"),"V18因子分":rank.get("score_v18"),
                             "新舊差異":round(float(rank.get("score"))-float(rank.get("score_v17")),1),
                             "證據項數":f"{int((item.get('v18_profile') or {}).get('evidence_n') or 0)}/13",
                             "資料品質 DQ":rank.get('dq_score'),"模型信心 MC":rank.get('mc_score'),
                             "研究資格":(rank.get('eligibility') or {}).get('status_label','待核'),
                             "關鍵階段":(item.get("v18_profile") or {}).get("stage")})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("V18.4 分數為尚未完成完整實盤樣本外驗證的研究指標，不代表勝率或報酬保證。回測必須使用當時實際已公布的財報、籌碼資料，不可用今日資料回填過去。")
