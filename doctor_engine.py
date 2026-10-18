"""Single-stock decision layer for Alpha Radar V16.7.

This module is deliberately downstream from the frozen selection model.
It answers two user questions without feeding back into ranking:
1) If I do not own the stock, is the current/next-session entry attractive?
2) If I already own it, does the current structure support holding, reducing, or exiting?

No result is a guarantee of return.  Historical-analog estimates are used only
when the evidence gate is met; otherwise the decision leans on structure/trend.
"""
from __future__ import annotations

import math
import numpy as np

from recommendation_engine import (
    evaluate_buy_opportunity,
    evaluate_next_session,
    expected_return_profile,
    suggested_buy_zone,
)
from strategy_config import DOCTOR_HOLD_POLICY
from trading_calendar import INTRADAY


def _f(v, default=None):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _clip_score(v: float) -> float:
    return float(np.clip(v, 0.0, 100.0))


def _scale(v, lo, hi, neutral=50.0):
    x = _f(v)
    if x is None or hi <= lo:
        return neutral
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0) * 100.0)


def buy_decision(stock: dict, horizon: str, phase: str, global_context: dict | None = None) -> dict:
    """Return a buy/observe decision using realtime data only during INTRADAY."""
    live = phase == INTRADAY and _f((stock.get("intraday") or {}).get("price")) is not None
    if phase == INTRADAY and not live:
        d = evaluate_next_session(stock, horizon)
        return {
            **d,
            "mode": "INTRADAY_FALLBACK",
            "action_code": "DATA",
            "headline": "即時資料不足，暫不給盤中買入訊號",
            "detail": "已保留最近完整交易日的觀察結論；取得官方即時成交價後再判斷是否可買。",
            "price": _f(stock.get("price")),
            "price_source": "最近完整交易日收盤（非即時）",
            "eligible_now": False,
        }
    if live:
        d = evaluate_buy_opportunity(stock, horizon, live=True, global_context=global_context)
        state = str(d.get("state") or "WATCH")
        mapping = {
            "BUY": ("BUY", "可分批買入"),
            "SMALL": ("SMALL", "可小量試單"),
            "CONTINUATION": ("SMALL", "續攻確認，可小量試單"),
            "PANIC_REVERSAL": ("SMALL", "恐慌殺低後止穩，可小量分批試單"),
            "WAIT_PULLBACK": ("WAIT", "等待拉回買入區"),
            "WAIT_BREAKOUT": ("WAIT", "等待突破確認"),
            "WATCH": ("WAIT", "暫不買入"),
        }
        action_code, headline = mapping.get(state, ("WAIT", str(d.get("action") or "暫不買入")))
        quote_source = str((stock.get("intraday") or {}).get("quote_source") or "官方即時報價")
        return {
            **d,
            "mode": "INTRADAY",
            "action_code": action_code,
            "headline": headline,
            "detail": str(d.get("action") or headline),
            "price_source": quote_source + " 即時價",
        }

    d = evaluate_next_session(stock, horizon)
    stance = str(d.get("stance") or "")
    entry = _f(d.get("entry_score"), 0.0) or 0.0
    expected = _f(d.get("expected_return"))
    if "優先觀察" in stance and entry >= 70 and (expected is None or expected > 0):
        action_code = "READY"
        headline = "下一交易日可優先觀察買入"
    elif "突破" in stance:
        action_code = "WAIT"
        headline = "下一交易日等待突破確認"
    elif "不追" in stance:
        action_code = "WAIT"
        headline = "強勢但不追價"
    else:
        action_code = "WAIT"
        headline = "等待較佳買入區"
    return {
        **d,
        "mode": "COMPLETED_DAY",
        "action_code": action_code,
        "headline": headline,
        "detail": stance or headline,
        "price": _f(stock.get("price")),
        "price_source": "最近完整交易日收盤",
        "eligible_now": False,
    }


def _trend_score(stock: dict, horizon: str, price: float) -> tuple[float, str, float | None]:
    setup = stock.get("setup") or {}
    if horizon == "short":
        primary = _f(setup.get("ma20"))
        secondary = _f(setup.get("ma60"))
        label = "20日線"
    elif horizon == "mid":
        primary = _f(setup.get("ma60")) or _f(setup.get("ma20"))
        secondary = _f(setup.get("ma120"))
        label = "60日線"
    else:
        primary = _f(setup.get("ma120")) or _f(setup.get("ma60"))
        secondary = _f(setup.get("ma240"))
        label = "120日線"

    if primary is None:
        return 50.0, "中期趨勢基準不足", None
    if price >= primary and (secondary is None or price >= secondary):
        return 82.0, f"價格守在{label}與主要趨勢之上", primary
    if price >= primary:
        return 68.0, f"價格仍守住{label}", primary
    if secondary is not None and price >= secondary:
        return 43.0, f"已跌破{label}，但較長趨勢尚未完全失守", primary
    return 25.0, f"價格已跌破{label}及較長趨勢", primary


def _model_profit_protection(stock: dict, horizon: str, price: float, cost: float | None) -> float | None:
    if cost is None or cost <= 0 or price <= cost:
        return None
    block = (stock.get("horizons") or {}).get(horizon) or {}
    plan = block.get("plan") or {}
    setup = stock.get("setup") or {}
    atr = _f(plan.get("atr14"), _f(setup.get("atr14")))
    if atr is None or atr <= 0:
        return None
    gain = price / cost - 1.0
    activation = float(DOCTOR_HOLD_POLICY["profit_protect_activation"].get(horizon, 0.12))
    if gain < activation:
        return None
    if horizon == "short":
        anchor = _f(setup.get("ma5"), _f(setup.get("ma20")))
    elif horizon == "mid":
        anchor = _f(setup.get("ma20"), _f(setup.get("ma60")))
    else:
        anchor = _f(setup.get("ma60"), _f(setup.get("ma120")))
    buffer_atr = float(DOCTOR_HOLD_POLICY["profit_protect_atr_buffer"].get(horizon, 1.0))
    model_stop = _f(plan.get("invalidation"))
    candidates = [cost]
    if model_stop is not None:
        candidates.append(model_stop)
    if anchor is not None:
        candidates.append(anchor - buffer_atr * atr)
    protect = max(candidates)
    # A protection reference must stay below the current price; otherwise it is
    # already triggered and should be reported as such rather than shown as a
    # future stop.
    return round(float(min(protect, price - 0.25 * atr)), 2)


def holding_decision(
    stock: dict,
    horizon: str,
    phase: str,
    *,
    cost: float | None = None,
    original_stop: float | None = None,
    trailing_protection: float | None = None,
    thesis_state: str = "UNKNOWN",
) -> dict:
    """Classify an existing position as HOLD / PROTECT / REDUCE / SELL.

    Hard exits (explicit broken thesis or a breached user/model structural stop)
    always dominate the score.  New-entry timing is intentionally excluded from
    the holding score because a stock may be unattractive to buy *now* while an
    existing lower-cost position remains valid to hold.
    """
    live = phase == INTRADAY and _f((stock.get("intraday") or {}).get("price")) is not None
    intraday = stock.get("intraday") or {}
    block = (stock.get("horizons") or {}).get(horizon) or {}
    plan = block.get("plan") or {}
    if phase == INTRADAY and not live:
        reference_price = _f(stock.get("price"))
        model_stop = _f(plan.get("invalidation"))
        return {
            "action_code": "DATA",
            "headline": "官方即時價不足，暫不判定盤中續抱／賣出",
            "score": None,
            "price": None,
            "reference_price": reference_price,
            "price_source": "最近完整交易日收盤（僅參考，非盤中執行價）",
            "cost": cost,
            "pnl_pct": None,
            "model_stop": None if model_stop is None else round(model_stop, 2),
            "effective_stop": None if model_stop is None else round(model_stop, 2),
            "profit_protection": None,
            "trend_anchor": None,
            "remaining_expected_return": None,
            "positive_rate": None,
            "forecast_confidence": None,
            "sample_n": 0,
            "evidence_ok": False,
            "selection_score": _f(block.get("ranking_score")),
            "structure_score": None,
            "trend_score": None,
            "forecast_score": None,
            "reasons": ["盤中缺少已驗證的最新成交價，禁止用昨收替代目前價格做續抱／賣出判斷"],
            "hard_exit_reasons": [],
        }
    price = _f(intraday.get("price")) if live else _f(stock.get("price"))
    if price is None or price <= 0:
        return {"action_code": "DATA", "headline": "價格資料不足", "score": None, "reasons": ["無法取得有效價格"]}

    model_stop = _f(plan.get("invalidation"))
    atr = _f(plan.get("atr14"), max(price * 0.03, 0.01)) or max(price * 0.03, 0.01)
    selection = _f(block.get("ranking_score"), 50.0) or 50.0
    profile = expected_return_profile(stock, horizon, price)
    expected = _f(profile.get("conservative_expected_return"), _f(profile.get("remaining_expected_return")))
    positive = _f(profile.get("positive_rate"))
    confidence = _f(profile.get("forecast_confidence"), 0.0) or 0.0
    sample_n = int(_f(profile.get("sample_n"), 0) or 0)
    evidence_ok = confidence >= float(DOCTOR_HOLD_POLICY["min_forecast_confidence"]) and sample_n >= int(DOCTOR_HOLD_POLICY["min_sample_n"])

    # Structural score uses distance above the completed-day model invalidation.
    if model_stop is None:
        structure_score = 50.0
        structure_reason = "模型防守價不足，結構分數採中性"
    elif price <= model_stop:
        structure_score = 0.0
        structure_reason = "已跌破模型結構防守價"
    else:
        dist_atr = (price - model_stop) / max(atr, 0.01)
        structure_score = _clip_score(35.0 + min(55.0, max(0.0, dist_atr) * 18.0))
        structure_reason = f"距模型防守價約 {dist_atr:.1f} ATR"

    trend_score, trend_reason, trend_anchor = _trend_score(stock, horizon, price)

    if evidence_ok:
        expected_score = _scale(expected, -0.08, 0.16)
        positive_score = _scale(positive, 0.35, 0.70)
        forecast_score = 0.62 * expected_score + 0.38 * positive_score
        # Remaining-return evidence is shrunk toward neutral when confidence is
        # just above the action threshold.
        reliability = np.clip((confidence - DOCTOR_HOLD_POLICY["min_forecast_confidence"]) / 65.0, 0.0, 1.0)
        forecast_score = 50.0 + (forecast_score - 50.0) * float(0.45 + 0.55 * reliability)
        forecast_reason = "歷史案例仍支持正向" if forecast_score >= 55 else "歷史案例的剩餘報酬轉弱"
    else:
        forecast_score = 50.0
        forecast_reason = f"歷史案例信心不足（{confidence:.0f}/100，n={sample_n}），不以此強制賣出"

    weights = DOCTOR_HOLD_POLICY["weights"]
    score = (
        structure_score * float(weights["structure"])
        + trend_score * float(weights["trend"])
        + forecast_score * float(weights["forecast"])
        + selection * float(weights["selection"])
    )

    intraday_reason = None
    if live:
        state = str(intraday.get("state") or "")
        if state == "盤中轉弱":
            score -= float(DOCTOR_HOLD_POLICY.get("intraday_weak_penalty", 7.0))
            intraday_reason = "盤中相對轉弱，持股評分下修"
        elif state == "盤中強勢":
            score += float(DOCTOR_HOLD_POLICY.get("intraday_strong_bonus", 4.0))
            intraday_reason = "盤中仍維持相對強勢"
    score = _clip_score(score)

    user_stop = _f(original_stop)
    user_trail = _f(trailing_protection)
    thesis_broken = str(thesis_state).upper() in {"BROKEN", "FALSE", "已不成立"}
    hard_reasons: list[str] = []
    if thesis_broken:
        hard_reasons.append("你已確認原買進理由不成立")
    if user_stop is not None and user_stop > 0 and price <= user_stop:
        hard_reasons.append(f"已跌破你設定的原始防守價 {user_stop:.2f}")
    if user_trail is not None and user_trail > 0 and price <= user_trail:
        hard_reasons.append(f"已跌破你設定的獲利保護價 {user_trail:.2f}")
    if model_stop is not None and price <= model_stop:
        hard_reasons.append(f"已跌破模型結構防守價 {model_stop:.2f}")

    pnl_pct = (price / cost - 1.0) * 100.0 if cost is not None and cost > 0 else None
    protection = _model_profit_protection(stock, horizon, price, cost)

    if hard_reasons:
        action_code, headline = "SELL", "賣出／退出優先"
    elif score < float(DOCTOR_HOLD_POLICY["reduce_score"]) and evidence_ok:
        action_code, headline = "REDUCE", "持股優勢明顯轉弱，建議減碼"
    elif trend_score < 40 and score < float(DOCTOR_HOLD_POLICY["hold_score"]):
        action_code, headline = "REDUCE", "趨勢轉弱，建議減碼並提高防守"
    elif protection is not None and price <= protection:
        action_code, headline = "PROTECT", "已進入獲利保護區，建議先保護部分獲利"
    elif score >= float(DOCTOR_HOLD_POLICY["strong_hold_score"]):
        action_code, headline = "HOLD_STRONG", "結構仍強，可續抱"
    else:
        action_code, headline = "HOLD", "結構尚未失效，可續抱但持續觀察"

    reasons = list(hard_reasons)
    if not reasons:
        reasons.extend([structure_reason, trend_reason, forecast_reason])
        if intraday_reason:
            reasons.append(intraday_reason)
    reasons = reasons[:4]

    effective_stop = user_stop if user_stop is not None and user_stop > 0 else model_stop
    return {
        "action_code": action_code,
        "headline": headline,
        "score": round(score, 1),
        "price": price,
        "price_source": (str((stock.get("intraday") or {}).get("quote_source") or "官方即時報價") + " 即時價") if live else "最近完整交易日收盤",
        "cost": cost,
        "pnl_pct": None if pnl_pct is None else round(pnl_pct, 2),
        "model_stop": None if model_stop is None else round(model_stop, 2),
        "effective_stop": None if effective_stop is None else round(effective_stop, 2),
        "profit_protection": protection,
        "trend_anchor": None if trend_anchor is None else round(trend_anchor, 2),
        "remaining_expected_return": expected,
        "positive_rate": positive,
        "forecast_confidence": round(confidence, 1),
        "sample_n": sample_n,
        "evidence_ok": evidence_ok,
        "selection_score": round(selection, 1),
        "structure_score": round(structure_score, 1),
        "trend_score": round(trend_score, 1),
        "forecast_score": round(float(forecast_score), 1),
        "reasons": reasons,
        "hard_exit_reasons": hard_reasons,
    }
