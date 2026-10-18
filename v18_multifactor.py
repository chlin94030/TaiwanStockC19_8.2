"""Alpha Radar V18 multi-factor evidence layer.

15 transparent signals: 3 quality + 4 chips + 4 momentum + 2 sector +
2 execution-only risks.  No missing market-data value is invented. Prices
must be completed-session data, already cut off by radar_service.run_scan.
This module makes no network calls and does not replace V17 analog forecasts.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from industry_profile import fine_industry
from strategy_config import V18_DIMENSION_WEIGHTS, V18_OVERLAY_BLEND
from v18_confidence_engine import (
    FactorStatus, profile_evidences, calculate_dq_and_mc, historical_reliability,
    perturbation_stability, fuse_alpha_scores, evaluate_recommendation_eligibility,
)

FACTOR_GROUPS = {
    "quality": [("revenue", "營收成長／加速"), ("margin", "毛利率擴張"), ("cash", "現金流純度")],
    "chip": [("trust", "投信資金強度"), ("holders", "集保大戶變化"), ("margin_clean", "融資沉澱"), ("institutions", "法人資金強度")],
    "momentum": [("rs", "相對大盤強度"), ("squeeze", "布林壓縮確認"), ("volume", "量能放大"), ("breakout", "突破結構")],
    "sector": [("peers", "同業相對強勢"), ("breadth", "產業廣度共振")],
    "risk": [("heat", "均線乖離過熱"), ("macro", "美股隔夜壓力")],
}


# Stable, pre-declared sector clusters (no return-based reassignment).  A stock
# has exactly one primary cluster; broad official industries are fallback-only.
# These are analyst taxonomies, NOT formal exchange sector classifications.
CLUSTER_MEMBERS = {
    "AI伺服器與組裝": ("2382", "3231", "6669", "2317", "2376", "4938", "2324", "2356"),
    "水冷散熱與電源": ("2308", "3017", "3324", "3653", "2421"),
    "先進製程與封測": ("2330", "3711", "3131", "3583", "6239", "6187", "1560", "6643"),
    "高速CCL與PCB": ("2383", "6274", "2368", "3037", "8046", "3189"),
    "半導體IC設計": ("2454", "3034", "3661", "2379", "3443", "6531"),
    "重電及輸配電": ("1519", "1503", "1513", "1609", "1504"),
    "金控及金融": ("2881", "2882", "2891", "2886", "2884", "5871"),
    "航運運輸": ("2603", "2609", "2615", "2618", "2610"),
}
_CODE_CLUSTER = {code: group for group, members in CLUSTER_MEMBERS.items() for code in members}


def sector_group(ticker: str, fine: str, broad: str) -> tuple[str, str]:
    code = str(ticker or "").split(".")[0]
    if code in _CODE_CLUSTER:
        return _CODE_CLUSTER[code], "curated_supply_chain"
    if fine and "待補" not in fine and "－" in fine and not fine.endswith(("－產業鏈", "－供應鏈")):
        return fine, "fine_industry"
    broad = str(broad or "").strip()
    if broad and broad not in {"其他", "其他電子", "其他電子業", "未分類", "不分類"}:
        return "官方產業－" + broad, "official_broad"
    return "", "unknown"


def _number(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        n = float(str(value).replace(",", "").replace("%", "").strip())
        return float(n) if np.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def _bound(x: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(np.clip(x, low, high))


def _entry(value: float | None, note: str, source: str = "") -> dict:
    available = value is not None
    return {"score": round(_bound(value), 1) if available else None,
            "available": available, "note": note, "source": source,
            "status": "PARTIAL" if available else "MISSING",
            "reliability": 0.0, "period_end": None, "first_seen_at": None,
            "published_at": None}


def price_signals(df: pd.DataFrame, market_ret20: float | None = None) -> dict:
    """Evaluate **completed** candles only. Inputs are not fetched again.

    Rolling BB percentiles use only the prior 120 windows + current window, so
    the signal cannot see any future volatility.  A tight bearish band alone
    is not treated as a positive buy breakout.
    """
    if df is None or df.empty or len(df) < 65 or not {"Close", "Volume"}.issubset(df.columns):
        return {}
    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    volume = pd.to_numeric(df["Volume"], errors="coerce").astype(float)
    if not np.isfinite(close.iloc[-65:]).all() or (close.iloc[-65:] <= 0).any():
        return {}
    last = float(close.iloc[-1]); ma20 = float(close.iloc[-20:].mean())
    ma60 = float(close.iloc[-60:].mean())
    r20 = last / float(close.iloc[-21]) - 1 if len(close) >= 21 else None
    r5 = last / float(close.iloc[-6]) - 1 if len(close) >= 6 else None
    bmk = _number(market_ret20)
    rs20 = (r20 - bmk) if r20 is not None and bmk is not None else None
    vrden = _number(volume.iloc[-20:].mean())
    vol_ratio = float(volume.iloc[-5:].mean() / vrden) if vrden and vrden > 0 else None
    trailing_high = float(close.iloc[-21:-1].max()) if len(close) >= 21 else None
    trend_ok = last > ma20 and ma20 >= ma60
    breakout = bool(trailing_high is not None and last > trailing_high * 1.001)
    ma_bias = (last / ma20 - 1) * 100 if ma20 else None

    mean20 = close.rolling(20).mean()
    std20 = close.rolling(20).std(ddof=1)
    widths = ((std20 * 4.0) / mean20.replace(0, np.nan)).dropna().tail(121)
    width = float(widths.iloc[-1]) if not widths.empty else None
    width_percentile = float((widths <= width).mean()) if width is not None and len(widths) >= 45 else None

    values: dict[str, dict] = {}
    if rs20 is not None:
        values["rs"] = _entry(50 + rs20 * 700 + (5 if trend_ok else -5), "20日報酬相對加權指數", "官方完成日K／大盤")
    else:
        values["rs"] = _entry(None, "缺少同日大盤資料，不能假設RS為正")

    if width_percentile is None:
        values["squeeze"] = _entry(None, "布林歷史視窗不足")
    else:
        # Historical squeeze must coincide with constructive price action;
        # a thin band amid continuing decline is not a buy signal.
        compression = width_percentile <= 0.22
        if compression and trend_ok and (r5 or 0) > 0:
            squeeze_score = 81 + 12 * (0.22 - width_percentile) / 0.22
            description = "近半年低位壓縮，且價格結構轉強"
        elif compression and not trend_ok:
            squeeze_score = 30
            description = "雖壓縮，但趨勢未站穩，不視為起漲"
        elif compression:
            squeeze_score = 58
            description = "低波壓縮，尚待價量確認"
        else:
            squeeze_score = 48 + (0.50 - width_percentile) * 18
            description = "未達明顯壓縮位置"
        values["squeeze"] = _entry(squeeze_score, description, "完成日K／120窗分位")

    values["volume"] = (_entry(50 + (vol_ratio - 1.0) * 70, "近5日均量／20日均量", "完成日K")
                        if vol_ratio is not None else _entry(None, "成交量分母不足"))
    if trailing_high and r20 is not None:
        near_high = last >= trailing_high * 0.975
        confirms = breakout and (vol_ratio or 0) >= 1.15 and (rs20 is not None and rs20 > 0)
        break_score = (90 if confirms else 71 if breakout else 64 if near_high and trend_ok else 48 if trend_ok else 28)
        values["breakout"] = _entry(break_score,
            "突破＋量比≥1.15＋相對大盤轉強" if confirms else "未完成三項同時確認，避免把假突破當買點", "完成日K")
    else:
        values["breakout"] = _entry(None, "缺少20日突破基準")

    return {
        "price_date": str(df.index[-1].date()) if isinstance(df.index, pd.DatetimeIndex) else str(df.index[-1]),
        "last": round(last, 3), "ma20": round(ma20, 3), "ret20": r20,
        "ret5": r5, "above_ma20": last > ma20, "trend_ok": trend_ok,
        "vol_ratio": round(vol_ratio, 3) if vol_ratio is not None else None,
        "rs20": round(rs20, 4) if rs20 is not None else None,
        "bb_width": round(width, 5) if width is not None else None,
        "bb_percentile": round(width_percentile, 3) if width_percentile is not None else None,
        "bias_ma20_pct": round(ma_bias, 2) if ma_bias is not None else None,
        "breakout_confirmed": bool(breakout and (vol_ratio or 0) >= 1.15 and (rs20 or -1) > 0),
        "technical_factors": values,
    }


def sector_context(liquid: list[dict]) -> dict[str, dict]:
    """Low-cost, same-date peer co-movement computed from liquid stock prices.

    A peer needs data and the same completed trading date.  The target stock
    itself is excluded.  Generic sector labels cannot impersonate fine groups.
    """
    clusters: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for item in liquid:
        ticker = str(item.get("ticker") or "")
        fine = str(item.get("fine_industry") or fine_industry(ticker, item.get("name"), item.get("industry")))
        sig = item.get("v18_price") or {}
        sector, precision = sector_group(ticker, fine, str(item.get("industry") or ""))
        if not sector or sig.get("ret20") is None:
            continue
        clusters[(sector, str(sig.get("price_date") or ""), precision)].append(item)
    output: dict[str, dict] = {}
    for (sector, asof, precision), members in clusters.items():
        for member in members:
            ticker = str(member.get("ticker"))
            peers = [m for m in members if str(m.get("ticker")) != ticker]
            minimum = 8 if precision == "official_broad" else 3
            if len(peers) < minimum:
                output[ticker] = {"available": False, "sector": sector, "peer_count": len(peers),
                                  "precision": precision, "date": asof}
                continue
            peer_ret = [float(p["v18_price"]["ret20"]) for p in peers]
            leaders = sorted(peers, key=lambda p: float(p.get("avg_turnover_20d") or 0), reverse=True)[:3]
            leader_return = float(np.mean([float(p["v18_price"]["ret20"]) for p in leaders]))
            breadth = sum(1 for p in peers if bool(p["v18_price"].get("above_ma20"))) / len(peers)
            output[ticker] = {
                "available": True, "sector": sector, "date": asof, "precision": precision,
                "peer_count": len(peers),
                "peer_median_ret20": round(float(np.median(peer_ret)), 5),
                "leader_avg_ret20": round(leader_return, 5),
                "peers_above_ma20_pct": round(100 * breadth, 1),
                "leader_tickers": [str(p.get("ticker")) for p in leaders],
            }
    return output


def _from_research(research: dict, sig: dict, *, avg_vol_shares: float) -> dict[str, dict]:
    rev = research.get("monthly_revenue") or {}
    fin = research.get("financials") or {}
    inst = research.get("institutional_flow") or {}
    # Hard PIT invalidity also prevents numeric use, not just confidence scoring.
    if rev.get('pit') and not rev['pit'].get('valid'):
        rev = {}
    if fin.get('pit') and not fin['pit'].get('valid'):
        fin = {}
    if inst.get('pit') and not inst['pit'].get('valid'):
        inst = {}
    values: dict[str, dict] = {}

    yoy = _number((rev.get("latest") or {}).get("yoy_pct"))
    monthly_rows = [r for r in (rev.get("rows") or []) if isinstance(r, dict)]
    monthly_rows = sorted(monthly_rows, key=lambda r: str(r.get("month") or ""))
    if yoy is None and monthly_rows:
        yoy = _number(monthly_rows[-1].get("yoy_pct"))
    yoy_previous = _number(monthly_rows[-2].get("yoy_pct")) if len(monthly_rows) >= 2 else None
    if yoy is not None:
        # Level works even when official breadth only has one monthly observation.
        revenue_score = 50 + yoy * 1.1
        accel = None if yoy_previous is None else yoy - yoy_previous
        if accel is not None:
            revenue_score += _bound(accel * 0.9, -20, 20)
        values["revenue"] = _entry(revenue_score,
            f"YoY {yoy:+.1f}%" + (f"、加速度 {accel:+.1f} 個百分點" if accel is not None else "（前月缺資料，未宣稱加速）"),
            str(rev.get("source") or "官方月營收"))
    else:
        values["revenue"] = _entry(None, "未取得可比單月營收YoY")

    gm_now = _number(fin.get("gross_margin_latest_pct"))
    gm_avg = _number(fin.get("gross_margin_4q_avg_pct"))
    quarters = [q for q in (fin.get("quarters") or []) if isinstance(q, dict)]
    if (gm_now is None or gm_avg is None) and len(quarters) >= 2:
        margins = [_number(q.get("gross_margin_pct")) for q in quarters]
        margins = [x for x in margins if x is not None]
        if len(margins) >= 2:
            gm_now = margins[-1]
            gm_avg = float(np.mean(margins[:-1]))
    if gm_now is not None and gm_avg is not None:
        values["margin"] = _entry(50 + (gm_now - gm_avg) * 7,
            f"毛利率相較歷史基準 {gm_now-gm_avg:+.1f} 個百分點", str(fin.get("source") or "財報"))
    else:
        values["margin"] = _entry(None, "官方全市場快照只有最新季度，不能捏造毛利率擴張")

    cfo_ratio = _number(fin.get("cfo_to_net_income"))
    if cfo_ratio is None:
        cfo_ratio = _number(research.get("cfo_to_net_income"))
    if cfo_ratio is not None and cfo_ratio >= 0:
        values["cash"] = _entry(50 + (cfo_ratio - 0.7) * 75,
            f"營運現金流／正淨利 {cfo_ratio:.2f}", str(fin.get("source") or "財報"))
    else:
        values["cash"] = _entry(None, "缺少已核對的營運現金流與正淨利分母")

    sessions = _number(inst.get("sessions")) or 0
    ave_lots = avg_vol_shares / 1000 if avg_vol_shares > 0 else None
    trust_lots = _number(inst.get("trust_net_lots"))
    total_lots = _number(inst.get("total_net_lots"))
    # Official breadth usually contains ONE trading day; never divide by total
    # issued capital with an invented denominator.  Attenuate one-day noise.
    shrink = 0.30 + 0.70 * min(1, sessions / 10)
    if ave_lots and sessions >= 1 and trust_lots is not None:
        ratio = trust_lots / ave_lots / max(1, sessions)
        values["trust"] = _entry(50 + (ratio * 210) * shrink,
            f"投信買超／同期間均量 {ratio*100:+.2f}%（{int(sessions)}日證據）", str(inst.get("source") or "官方法人"))
    else:
        values["trust"] = _entry(None, "缺少可比投信買超／成交量資料")
    if ave_lots and sessions >= 1 and total_lots is not None:
        ratio = total_lots / ave_lots / max(1, sessions)
        values["institutions"] = _entry(50 + (ratio * 180) * shrink,
            f"三大法人買超／同期間均量 {ratio*100:+.2f}%", str(inst.get("source") or "官方法人"))
    else:
        values["institutions"] = _entry(None, "缺少可比三大法人資料")
    dist = research.get("holding_distribution") or {}
    holder_delta = _number(dist.get("major_holder_1w_delta_pct"))
    if holder_delta is not None:
        values["holders"] = _entry(50 + holder_delta * 22, f"大戶持股變動 {holder_delta:+.2f} 個百分點", "集保週資料")
    else:
        values["holders"] = _entry(None, "集保大戶資料未納入全市場可比快照")
    margin = research.get("margin") or research.get("margin_trading") or {}
    margin_delta = _number(margin.get("margin_lots_delta_5d"))
    ret5 = _number(sig.get("ret5"))
    if margin_delta is not None and ret5 is not None:
        clean = ret5 > 0 and margin_delta < 0
        values["margin_clean"] = _entry(86 if clean else 46 if ret5 < 0 and margin_delta > 0 else 55,
            "價漲融資退場" if clean else "未出現價漲資減", "官方融資餘額／完成日K")
    else:
        values["margin_clean"] = _entry(None, "缺少同期間融資餘額變化")
    return values


def build_profile(sig: dict, research: dict | None, peers: dict | None, *, avg_vol_shares: float = 0.0,
                  global_shock: float | None = None) -> dict:
    """Generate a portable 15-factor profile with explicit evidence gaps."""
    research = research if isinstance(research, dict) else {}
    peers = peers if isinstance(peers, dict) else {}
    observed = _from_research(research, sig, avg_vol_shares=float(avg_vol_shares or 0))
    observed.update(sig.get("technical_factors") or {})
    for fid in ('revenue', 'margin', 'cash', 'trust', 'institutions', 'holders', 'margin_clean'):
        if fid not in observed or observed[fid].get('score') is None:
            continue
        section_name = {'revenue':'monthly_revenue', 'margin':'financials', 'cash':'financials',
                        'trust':'institutional_flow', 'institutions':'institutional_flow',
                        'holders':'holding_distribution', 'margin_clean':'margin'}.get(fid)
        section = research.get(section_name) or {}
        pit = section.get('pit') or {}
        if not pit.get('valid'):
            observed[fid].update(status=pit.get('status', 'UNVERIFIED_TIME'), reliability=0.0)
        else:
            status = pit.get('status', 'OBSERVED_ONLY')
            strength = {'revenue': .91, 'margin': .87, 'cash': .88,
                        'trust': .90, 'institutions': .90, 'holders': .86,
                        'margin_clean': .86}[fid]
            if fid in ('trust', 'institutions'):
                sessions = _number(section.get('sessions')) or 0
                if sessions < 5:
                    status = 'PARTIAL'
                    strength *= max(.40, min(1.0, sessions / 10.0))
            observed[fid].update(status=status, reliability=strength,
                period_end=pit.get('period_end'), pit_verified=True,
                published_at=pit.get('published_at'), first_seen_at=pit.get('first_seen_at'))
    for fid in ('rs', 'squeeze', 'volume', 'breakout'):
        entry = observed.get(fid)
        if entry and entry.get('score') is not None and sig.get('price_date'):
            entry.update(status='VERIFIED', reliability=.97 if fid != 'squeeze' else .92,
                         period_end=sig.get('price_date'), pit_verified=True)
    if peers.get("available"):
        avg = _number(peers.get("peer_median_ret20"))
        leader = _number(peers.get("leader_avg_ret20"))
        breadth = _number(peers.get("peers_above_ma20_pct"))
        precision = str(peers.get("precision") or "")
        confidence = 0.40 if precision == "official_broad" else 1.0
        observed["peers"] = _entry(50 + ((avg or 0) * 550 + (leader or 0) * 150) * confidence,
            f"同群 {int(peers.get('peer_count') or 0)} 檔20日中位報酬 {((avg or 0)*100):+.1f}%" +
            ("（官方大業別，降低權重）" if precision == "official_broad" else ""), "完成日K／同日同群")
        observed["breadth"] = _entry(50 + (breadth - 50)*confidence, f"同群站上20日均線 {breadth:.0f}%", "完成日K／同日同群") if breadth is not None else _entry(None, "缺少產業廣度")
        for fid in ('peers', 'breadth'):
            entry = observed.get(fid) or {}
            if entry.get('score') is not None and sig.get('price_date') == peers.get('date'):
                entry.update(status='VERIFIED', reliability=.86 if precision != 'official_broad' else .53,
                             period_end=peers.get('date'), pit_verified=True)
            else:
                entry.update(status='UNVERIFIED_TIME', reliability=0.0)
    else:
        observed["peers"] = _entry(None, "同細產業可比同業少於3檔")
        observed["breadth"] = _entry(None, "同業不足，避免假設族群共振")
    bias = _number(sig.get("bias_ma20_pct"))
    if bias is not None:
        heat = 0 if bias <= 12 else min(100, (bias - 12) * 10)
        observed["heat"] = _entry(heat, f"距20日線 {bias:+.1f}%（愈高風險愈大）", "完成日K")
    else:
        observed["heat"] = _entry(None, "缺少20日線偏離")
    shock = _number(global_shock)
    observed["macro"] = (_entry(shock, f"已核對國際市場壓力 {shock:.0f}/100", "國際快讀")
                         if shock is not None else _entry(None, "國際訊號尚未核對，不假設隔夜下跌"))

    factors = {}
    scores = {}
    for group, field_list in FACTOR_GROUPS.items():
        values = []
        for key, label in field_list:
            item = dict(observed.get(key) or _entry(None, "尚無同口徑證據"))
            item["label"] = label
            factors[key] = item
            if group != "risk":
                values.append(50.0 if item["score"] is None else float(item["score"]))
        if group != "risk":
            scores[group] = round(float(np.mean(values)), 1)
    positive_n = sum(len(v) for k, v in FACTOR_GROUPS.items() if k != "risk")
    available_n = sum(bool(factors[k]["available"]) for group, v in FACTOR_GROUPS.items() if group != "risk" for k, _ in v)
    heat_value = factors["heat"]["score"]
    macro_value = factors["macro"]["score"]
    # Penalty budgets total at most 10 POINTS and are reserved for *execution*.
    heat_penalty = min(6.0, float(heat_value or 0.0) * 0.06)
    macro_penalty = min(4.0, max(0.0, float(macro_value or 0.0) - 60) * 0.10)
    risk_penalty = round(heat_penalty + macro_penalty, 1)
    if bias is not None and bias >= 18:
        # An overheated stock can remain high-quality, but its entry stage
        # must never be described as an unqualified breakout opportunity.
        state = "過熱待回測"
    elif bool(sig.get("breakout_confirmed")):
        state = "突破確認"
    elif (sig.get("bb_percentile") is not None and sig["bb_percentile"] <= 0.22 and sig.get("trend_ok")):
        state = "壓縮蓄勢"
    elif sig.get("trend_ok"):
        state = "趨勢觀察"
    else:
        state = "結構待確認"
    return {
        "engine": "v18-evidence-aware", "date": sig.get("price_date"),
        "factors": factors, "dimensions": scores,
        "evidence_n": int(available_n), "evidence_total": positive_n,
        "evidence_coverage": round(available_n / positive_n, 3) if positive_n else 0,
        "risk_penalty_for_entry": risk_penalty,
        "risk_parts": {"heat": round(heat_penalty, 1), "macro": round(macro_penalty, 1)},
        "stage": state,
        "metrics": {k: sig.get(k) for k in ("rs20", "vol_ratio", "bb_width", "bb_percentile", "bias_ma20_pct", "breakout_confirmed")},
        "sector": {k: peers.get(k) for k in ("sector", "precision", "peer_count", "leader_tickers")},
    }


def score_horizon(profile: dict, horizon: str, v17_score: float,
                  forecast: dict | None = None, *, financial_industry: bool = False) -> dict:
    """Evidence-weighted positive ranking with separate DQ / MC / execution risk.

    Data-poor names retain the old baseline rather than losing a fixed 3 points.
    Risk (heat/macro) never changes selection. Historical confidence is NOT OOS.
    """
    evidences = profile_evidences(profile, financial_industry=financial_industry)
    h_score = historical_reliability(forecast)
    stability = perturbation_stability(evidences, horizon)
    quality = calculate_dq_and_mc(evidences, horizon,
                                  historical_sample_score=h_score,
                                  signal_stability_score=stability)
    blend = fuse_alpha_scores(v17_score, quality['v18_pure_alpha'],
                              quality['dq_score'], V18_OVERLAY_BLEND[horizon])
    gate = evaluate_recommendation_eligibility(
        blend['score'], quality['dq_score'], quality['mc_score'],
        quality['independent_categories_count'], horizon,
        historical_oos_verified=False,
        pit_violations=sum(e.status == FactorStatus.LOOKAHEAD for e in evidences),
    )
    return {
        **blend,
        'raw_overlay_budget': V18_OVERLAY_BLEND[horizon],
        'risk_adjusted_watch': round(_bound(blend['score'] - float(profile.get('risk_penalty_for_entry') or 0)), 1),
        'dq_score': quality['dq_score'], 'coverage_pct': quality['coverage_pct'],
        'mc_score': quality['mc_score'], 'confidence': quality,
        'eligibility': gate, 'signal_stability_type': 'local_perturbation_not_OOS',
    }
