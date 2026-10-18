"""Taiwan Alpha Radar V17 Goal-Complete - full-universe scan, rank, enrich and diagnose."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os
import copy

import numpy as np
import pandas as pd

from market_data import DailyPriceStore, ResearchDataClient, fetch_twse_universe, _taipei_timestamp
from policy_engine import generate_trade_plan, evaluate_entry_state, entry_timing_from_df, position_guidance, setup_from_df
from return_first_model import estimate_all_horizons
from industry_profile import fine_industry
from v18_multifactor import build_profile as build_v18_profile, price_signals as v18_price_signals, sector_context as v18_sector_context, score_horizon as v18_score_horizon
from trading_calendar import session_reference_policy, INTRADAY, POSTCLOSE, calendar_reference
from limitup_lab_v1820 import price_setup as limitup_price_setup, decorate_candidate as limitup_decorate, LAB_VERSION as LIMITUP_LAB_VERSION, SIGNAL_WEIGHTS as LIMITUP_FACTOR_WEIGHTS, BLEND_WEIGHTS as LIMITUP_BLEND_WEIGHTS
from pattern_chip_lab_v1821 import technical_factors as pattern_technicals, build_candidate as pattern_candidate, VERSION as PATTERN_VERSION, REVERSAL_WEIGHTS, BREAKOUT_WEIGHTS
from strategy_config import ARCHITECTURE_VERSION, SCAN_DEFAULTS, SELECTION_WEIGHTS, EMPIRICAL_RELIABILITY_POLICY, CANONICAL_MODEL_POLICY

OPERATIONS_VERSION = ARCHITECTURE_VERSION


@dataclass
class RunSettings:
    canonical_mode: bool = True
    reference_size: int = int(SCAN_DEFAULTS["reference_size"])
    candidate_size: int = int(SCAN_DEFAULTS["candidate_size"])
    research_pool_per_horizon: int = int(SCAN_DEFAULTS["research_pool_per_horizon"])
    history_period: str = str(SCAN_DEFAULTS["history_period"])
    model_family: str = "full"  # price_only | business_confirmed | flow_confirmed | full
    order_mode: str = "next_open"
    commission: float = 0.001425
    sell_tax: float = 0.003
    slippage: float = 0.0005
    notional: float = 100000.0
    min_ev_short: float = 0.0
    min_ev_mid: float = 0.0
    min_ev_long: float = 0.0
    min_price: float = float(SCAN_DEFAULTS["min_price"])
    min_avg_turnover: float = float(SCAN_DEFAULTS["min_avg_turnover"])


def load_dashboard(path: Path, include_features: bool = False) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compact_session_dashboard(snap: dict | None) -> dict:
    if not snap or not isinstance(snap, dict):
        return {}
    out = dict(snap)
    out["charts"] = {}
    return out


def compact_doctor_result(dr: dict | None) -> dict:
    return dict(dr) if isinstance(dr, dict) else {}


def remove_saved_dashboard(path: Path) -> bool:
    try:
        if path.exists():
            path.unlink()
        return True
    except Exception:
        return False


def snapshot_bytes(snap: dict) -> bytes:
    return json.dumps(snap, ensure_ascii=False, indent=2).encode("utf-8")


def chart_on_demand(snap: dict | None, ticker: str, data_dir: Path, allow_fetch: bool = False) -> dict | None:
    if not ticker:
        return None
    store = DailyPriceStore(data_dir / "daily_prices.sqlite")
    # Charts shown next to completed-day recommendations stop at the model's
    # reference date. This prevents a half-finished Yahoo daily bar from making
    # the chart disagree with the frozen baseline during the session.
    sess = (snap or {}).get("session") or {}
    if str(ticker).endswith(".TWO"):
        cutoff = str(sess.get("tpex_complete_date") or (snap or {}).get("price_date") or "") or None
    else:
        cutoff = str(sess.get("twse_complete_date") or (snap or {}).get("price_date") or "") or None
    df = store.get_prices(ticker, end_date=cutoff)
    if df.empty and allow_fetch:
        store.batch_fetch_and_update([ticker], period="1y")
        df = store.get_prices(ticker, end_date=cutoff)
    if df.empty:
        return None
    tail = df.tail(260)
    return {
        "dates": tail.index.strftime("%Y-%m-%d").tolist(),
        "ohlcv": tail[["Open", "High", "Low", "Close", "Volume"]].to_numpy().tolist(),
    }


def _market_context(store: DailyPriceStore, end_date: str | None = None) -> dict:
    df = store.get_prices("^TWII", end_date=end_date)
    if df.empty or len(df) < 65:
        return {"benchmark": "^TWII", "regime": "UNKNOWN", "ret20": 0.0, "price_date": ""}
    close = df["Close"]
    p = float(close.iloc[-1])
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())
    ret20 = p / float(close.iloc[-21]) - 1.0 if len(close) >= 21 else 0.0
    ret5 = p / float(close.iloc[-6]) - 1.0 if len(close) >= 6 else None
    day_change_pct = ((p / float(close.iloc[-2])) - 1.0) * 100.0 if len(close) >= 2 and float(close.iloc[-2]) > 0 else 0.0
    if p >= ma20 and ma20 >= ma60:
        regime = "BULL"
    elif p < ma20 and p >= ma60:
        regime = "NEUTRAL"
    else:
        regime = "BEAR"
    return {
        "benchmark": "^TWII",
        "regime": regime,
        "ret20": ret20,
        "ret5": ret5,
        "day_change_pct": round(float(day_change_pct), 3),
        "price": p,
        "ma20": ma20,
        "ma60": ma60,
        "price_date": str(df.index[-1].date()),
    }


def _pre_score(df: pd.DataFrame, market_ret20: float) -> float:
    if df.empty or len(df) < 130:
        return -999.0
    close = df["Close"]
    vol = df["Volume"]
    p = float(close.iloc[-1])
    r20 = p / float(close.iloc[-21]) - 1.0 if len(close) >= 21 else 0.0
    r60 = p / float(close.iloc[-61]) - 1.0 if len(close) >= 61 else r20
    r120 = p / float(close.iloc[-121]) - 1.0 if len(close) >= 121 else r60
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())
    vr = float(vol.tail(5).mean()) / max(1.0, float(vol.tail(20).mean()))
    rs = r20 - market_ret20
    trend = (1.0 if p > ma20 else 0.0) + (1.0 if ma20 > ma60 else 0.0)
    return 100.0 * (0.28 * np.tanh(rs * 5) + 0.24 * np.tanh(r60 * 2.5) + 0.18 * np.tanh(r120 * 1.5) + 0.12 * np.tanh((vr - 1) * 1.4) + 0.09 * trend)


def _fundamental_score(research: dict) -> tuple[float | None, dict]:
    if not research:
        return None, {}
    rev = research.get("monthly_revenue") or {}
    fin = research.get("financials") or {}
    val = research.get("valuation") or {}
    pieces = []
    detail = {}

    latest = rev.get("latest") or {}
    if rev.get("available") and latest:
        yoy = latest.get("yoy_pct")
        mom = latest.get("mom_pct")
        avg3 = rev.get("avg_yoy_3m_pct")
        rev_parts = []
        if yoy is not None:
            rev_parts.append(float(np.clip(50 + yoy * 1.1, 0, 100)))
        if avg3 is not None:
            rev_parts.append(float(np.clip(50 + avg3 * 1.0, 0, 100)))
        if mom is not None:
            rev_parts.append(float(np.clip(50 + mom * 1.2, 0, 100)))
        if rev_parts:
            score = float(np.mean(rev_parts))
            pieces.append((score, 0.45))
            detail["revenue_score"] = round(score, 1)

    if fin.get("available"):
        qs = fin.get("quarters") or []
        eps = [q.get("eps") for q in qs if q.get("eps") is not None]
        gm = [q.get("gross_margin_pct") for q in qs if q.get("gross_margin_pct") is not None]
        ep = 50.0
        if eps:
            ep = 65.0 if eps[-1] > 0 else 25.0
            if len(eps) >= 2:
                ep += 15.0 if eps[-1] > eps[-2] else -10.0
            if sum(eps) > 0:
                ep += 5.0
            ep = float(np.clip(ep, 0, 100))
            pieces.append((ep, 0.35))
            detail["earnings_score"] = round(ep, 1)
        else:
            # Official breadth snapshots expose cumulative/YTD EPS rather than
            # manufactured single-quarter EPS.  Use only its sign as a modest
            # profitability signal; richer trend scoring is reserved for the
            # comparable historical series when available.
            ytd = fin.get("latest_ytd_eps")
            try:
                ytd = float(ytd) if ytd is not None else None
            except Exception:
                ytd = None
            if ytd is not None and np.isfinite(ytd):
                ep = 60.0 if ytd > 0 else 25.0
                pieces.append((ep, 0.22))
                detail["earnings_score"] = round(ep, 1)
                detail["earnings_basis"] = "official_ytd"
        if len(gm) >= 2:
            gm_score = float(np.clip(50 + (gm[-1] - gm[0]) * 4.0, 20, 90))
            pieces.append((gm_score, 0.20))
            detail["margin_score"] = round(gm_score, 1)

    if not pieces:
        return None, detail
    wsum = sum(w for _, w in pieces)
    score = sum(s * w for s, w in pieces) / wsum

    # Only a mild valuation sanity adjustment; growth stocks are not penalized solely for a high PE.
    pe = (val or {}).get("pe")
    if pe is not None:
        if pe <= 0:
            score -= 8
        elif pe > 100:
            score -= 5
    return round(float(np.clip(score, 0, 100)), 1), detail


def _flow_score(research: dict, avg_volume_shares: float) -> tuple[float | None, dict]:
    if not research:
        return None, {}
    inst = research.get("institutional_flow") or {}
    if not inst.get("available"):
        return None, {}
    net_lots = float(inst.get("total_net_lots") or 0.0)
    monthly_lots = max(1.0, avg_volume_shares / 1000.0 * max(1, int(inst.get("sessions") or 20)))
    ratio = net_lots / monthly_lots
    raw_score = float(np.clip(50.0 + ratio * 180.0, 5.0, 95.0))
    sessions = max(1, int(inst.get("sessions") or 1))
    reliability = min(1.0, sessions / 10.0)
    # A one-day official breadth snapshot is intentionally shrunk strongly
    # toward neutral; a richer 10+ session history can express the full signal.
    score = 50.0 + (raw_score - 50.0) * (0.25 + 0.75 * reliability)
    detail = {
        "institutional_net_to_volume": round(ratio, 4),
        "institutional_sessions": sessions,
        "institutional_reliability": round(reliability, 3),
    }

    branch = research.get("main_force_proxy") or {}
    if branch.get("available"):
        proxy = float(branch.get("proxy_net_lots") or 0.0)
        branch_ratio = proxy / max(1.0, avg_volume_shares / 1000.0 * max(1, int(branch.get("sessions") or 10)))
        branch_score = float(np.clip(50.0 + branch_ratio * 120.0, 5.0, 95.0))
        score = 0.72 * score + 0.28 * branch_score
        detail["branch_proxy_net_to_volume"] = round(branch_ratio, 4)
    return round(float(np.clip(score, 0, 100)), 1), detail


def empirical_evidence_score(forecast: dict) -> float | None:
    """Confidence-shrunk historical-analog score used by production ranking.

    return_first_model already computes an empirical quality score that includes
    median/mean return, smoothed hit rate and downside-tail quality.  V16.4 then
    recomputed a second, slightly different empirical score here, which made the
    validation core and production selector drift apart.  V16.5 uses the single
    canonical empirical score and shrinks weak evidence toward neutral (50).
    """
    if not forecast or not forecast.get("estimate_available"):
        return None
    quality = forecast.get("empirical_quality_score")
    confidence = forecast.get("confidence_score")
    try:
        q = float(quality)
    except Exception:
        q = np.nan
    if not np.isfinite(q):
        # Backward-compatible fallback for old snapshots lacking the canonical
        # field.  New scans should normally not enter this branch.
        s = forecast.get("strategy") or {}
        median = float(s.get("median") or 0.0)
        mean = float(s.get("mean") or 0.0)
        p10 = float(s.get("p10") or 0.0)
        win = float(s.get("smoothed_positive_rate") or 0.5)
        q = (
            50.0
            + np.tanh(median * 8.0) * 24.0
            + np.tanh(mean * 6.0) * 12.0
            + np.tanh((win - 0.5) * 4.0) * 14.0
            - max(0.0, -p10 - 0.06) * 35.0
        )
    try:
        conf = float(confidence)
    except Exception:
        conf = 0.0
    conf = float(np.clip(conf if np.isfinite(conf) else 0.0, 0.0, 100.0)) / 100.0
    neutral = float(EMPIRICAL_RELIABILITY_POLICY["neutral_score"])
    floor = float(EMPIRICAL_RELIABILITY_POLICY["confidence_floor_multiplier"])
    reliability = floor + (1.0 - floor) * conf
    effective = neutral + (float(q) - neutral) * reliability
    return round(float(np.clip(effective, 0.0, 100.0)), 1)


def _forecast_score(forecast: dict) -> float | None:
    # Kept as a compatibility alias for older tests/callers.
    return empirical_evidence_score(forecast)


def _combined_ranking_score(stock: dict, horizon: str, model_family: str) -> None:
    block = stock["horizons"][horizon]
    forecast = block.get("forecast") or {}
    technical = float(forecast.get("technical_factor_score", forecast.get("composite_factor_score", 0.0)) or 0.0)
    empirical = _forecast_score(forecast)
    ranking_research = stock["ranking_research"] if "ranking_research" in stock else (stock.get("research") or {})
    fundamental, f_detail = _fundamental_score(ranking_research)
    flow, flow_detail = _flow_score(ranking_research, float(stock.get("avg_volume_20d") or 0.0))

    weights = SELECTION_WEIGHTS[horizon]
    allowed = {"technical", "empirical"}
    if model_family in {"business_confirmed", "full"}:
        allowed.add("fundamental")
    if model_family in {"flow_confirmed", "full"}:
        allowed.add("flow")

    values = {"technical": technical, "empirical": empirical, "fundamental": fundamental, "flow": flow}
    neutral = float(CANONICAL_MODEL_POLICY.get("research_neutral_score", 50.0))
    allowed_weight = sum(float(weights[k]) for k in allowed) or 1.0
    present_weight = sum(float(weights[k]) for k in allowed if values.get(k) is not None)

    # V17 fairness rule: missing evidence is neutral-imputed instead of silently
    # re-normalizing the remaining components upward.  Therefore two stocks are
    # always compared on the same evidence scale.  A small coverage penalty
    # prevents systematic data gaps from becoming an advantage.
    rank = sum(float(values[k] if values[k] is not None else neutral) * float(weights[k]) for k in allowed) / allowed_weight
    coverage = float(np.clip(present_weight / allowed_weight, 0.0, 1.0))
    penalty_max = float(CANONICAL_MODEL_POLICY.get("missing_evidence_penalty_max", 4.0))
    rank -= penalty_max * (1.0 - coverage)

    block["ranking_score"] = round(float(np.clip(rank, 0, 100)), 1)
    block["score_components"] = {
        "technical": round(technical, 1),
        "empirical": empirical,
        "empirical_raw": forecast.get("empirical_quality_score"),
        "forecast_confidence": forecast.get("confidence_score"),
        "analog_n": forecast.get("local_effective_n"),
        "fundamental": fundamental,
        "flow": flow,
        "evidence_coverage": round(coverage, 3),
        "missing_evidence_penalty": round(penalty_max * (1.0 - coverage), 2),
        **f_detail,
        **flow_detail,
    }
    if model_family == "full" and stock.get("v18_profile"):
        scores = v18_score_horizon(
            stock["v18_profile"], horizon, block["ranking_score"],
            forecast=forecast,
            financial_industry=("金融" in str(stock.get("industry") or "") or
                                "保險" in str(stock.get("industry") or "") or
                                str(stock.get("ticker") or "").split(".")[0].startswith("28")),
        )
        block["rank_v18"] = scores
        block["ranking_score"] = scores["score"]
        block["score_components"]["v18_evidence_coverage"] = stock["v18_profile"]["evidence_coverage"]
        block["score_components"]["v17_preblend"] = scores["score_v17"]
        block["score_components"]["data_quality_dq"] = scores["dq_score"]
        block["score_components"]["model_confidence_mc"] = scores["mc_score"]


def _enrich_tickers(stocks: list[dict], data_dir: Path, tickers: list[str], progress=None, progress_value: float = 0.88):
    pool = [t for t in dict.fromkeys(tickers) if t]
    if not pool:
        return
    by_ticker = {s["ticker"]: s for s in stocks}
    if progress:
        progress(f"補充公司與法人資料（{len(pool)} 檔）", progress_value)

    def one(ticker):
        client = ResearchDataClient(data_dir / "research_cache.sqlite")
        return ticker, client.stock_research(ticker, include_branch=False)

    workers = min(6, max(1, len(pool)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, ticker) for ticker in pool]
        for fut in as_completed(futures):
            try:
                ticker, research = fut.result()
                if ticker in by_ticker:
                    by_ticker[ticker]["research"] = research
            except Exception:
                continue




def _load_breadth_research(tickers: list[str], data_dir: Path, asof_date: str | None = None) -> dict[str, dict]:
    """Load comparable market-wide official evidence for canonical ranking."""
    if not tickers:
        return {}
    client = ResearchDataClient(Path(data_dir) / "research_cache.sqlite")
    return client.official_breadth_snapshot(tickers, asof_date=asof_date)

def _enrich_research(stocks: list[dict], data_dir: Path, settings: RunSettings, progress=None):
    if not stocks:
        return
    tickers = []
    for h in ["short", "mid", "long"]:
        ranked = sorted(
            stocks,
            key=lambda x: x.get("horizons", {}).get(h, {}).get("forecast", {}).get("composite_factor_score", 0),
            reverse=True,
        )[: max(5, int(settings.research_pool_per_horizon))]
        tickers.extend([x["ticker"] for x in ranked])
    _enrich_tickers(stocks, data_dir, tickers, progress=progress, progress_value=0.88)



def model_candidates(liquid: list[dict], settings: RunSettings) -> list[dict]:
    """Canonical V17 universe contract used by run_scan and tests.

    Production mode evaluates every liquid stock with sufficient completed-day
    history.  Candidate-size truncation exists only for explicit non-canonical
    research experiments.
    """
    if bool(getattr(settings, "canonical_mode", True)):
        return list(liquid)
    return list(liquid)[: max(50, int(getattr(settings, "candidate_size", 1000)))]

def _safe_cached_cutoff(store: DailyPriceStore, today: str, phase: str,
                        tickers: list[str] | None = None) -> str:
    """Newest ACTUALLY observed cross-market completed cache day.

    Never infer a completed-day candle from the calendar.  After the close,
    today is usable as a provisional research date ONLY if the index and a
    broad majority of BOTH listed and OTC prices are already present in the
    database.  The authoritative official-exchange date remains separate.
    """
    try:
        from market_data import _sqlite_connect
        listed_total = sum(str(t).endswith('.TW') for t in (tickers or []))
        otc_total = sum(str(t).endswith('.TWO') for t in (tickers or []))
        if listed_total and otc_total and hasattr(store, 'db_path'):
            with _sqlite_connect(store.db_path) as conn:
                rows = conn.execute("""
                    SELECT date,
                      SUM(CASE WHEN ticker LIKE '%.TW' THEN 1 ELSE 0 END),
                      SUM(CASE WHEN ticker LIKE '%.TWO' THEN 1 ELSE 0 END),
                      SUM(CASE WHEN ticker = '^TWII' THEN 1 ELSE 0 END)
                    FROM daily_prices
                    WHERE date <= ?
                    GROUP BY date ORDER BY date DESC LIMIT 14
                """, (str(today),)).fetchall()
            for day, listed_n, otc_n, index_n in rows:
                day = str(day)
                if phase in {'PREOPEN', 'INTRADAY'} and day >= str(today):
                    continue
                if (int(index_n or 0) >= 1
                    and int(listed_n or 0) >= max(1, int(listed_total * .90))
                    and int(otc_n or 0) >= max(1, int(otc_total * .90))):
                    return day
            # Actual database observed, but neither exchange had broad enough
            # session coverage. Do NOT elevate an index-only bar to a full
            # market day; keep the previous fully published snapshot instead.
            return ''
    except Exception:
        pass
    # Backward-compatible fallback for mocked stores and old caches.  Only a
    # REAL index price row can establish a fallback date, never the query date.
    try:
        raw = store.get_prices('^TWII')
    except Exception:
        return ''
    if raw is None or raw.empty:
        return ''
    dates = [str(x.date()) for x in raw.index]
    if phase in {'PREOPEN', 'INTRADAY'}:
        dates = [d for d in dates if d < str(today)]
    return max(dates) if dates else ''


def run_scan(data_dir: Path, settings: RunSettings, progress=None, now=None) -> dict:
    """Build a completed-session model at any time of day.

    V16.6 rule: the daily model NEVER consumes an unfinished intraday candle.
    During PREOPEN/INTRADAY it uses the latest exchange-confirmed EOD session.
    During POSTCLOSE it explicitly asks for today's official EOD; if today's
    exchange files are not ready yet, the previous completed session remains the
    baseline and the UI may layer a clearly-labelled closing-quote preview.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    # Canonical V17 production mode has one reproducible definition.  UI/debug
    # knobs cannot silently shrink the market or change the history window.
    if bool(getattr(settings, "canonical_mode", True)):
        settings.history_period = str(CANONICAL_MODEL_POLICY["history_period"])
        settings.model_family = str(CANONICAL_MODEL_POLICY["model_family"])
    now = now or _taipei_timestamp()
    session = session_reference_policy(now)
    phase = session.get("phase")
    today = str(session.get("today") or "")

    if progress:
        progress("載入上市櫃股票", 0.04)
    universe = fetch_twse_universe()
    # Archive the ACTUALLY observed listing membership; never backfill historical
    # universes using today's survivors. Archive is independent of model ranking.
    universe_pit = {'recorded': False}
    try:
        from universe_archive import UniverseArchive
        universe_pit = UniverseArchive(data_dir / 'universe_observations.sqlite').record(
            universe.to_dict('records'), observed_at=_taipei_timestamp(),
        )
        universe_pit['recorded'] = True
    except Exception as exc:
        universe_pit = {'recorded': False, 'warning': type(exc).__name__}
    tickers = universe["ticker"].astype(str).tolist()
    store = DailyPriceStore(data_dir / "daily_prices.sqlite")

    if progress:
        progress("更新歷史快取", 0.12)

    def price_progress(done, total, label):
        if progress:
            frac = float(done) / max(1.0, float(total))
            progress(str(label), 0.12 + 0.25 * max(0.0, min(1.0, frac)))

    try:
        dl = store.batch_fetch_and_update(
            tickers + ["^TWII"],
            period=settings.history_period,
            progress=price_progress,
            full_refresh_days=int(CANONICAL_MODEL_POLICY.get("full_history_refresh_days", 30)),
        )
    except TypeError:
        # Backward-compatible with test doubles / older data stores.
        dl = store.batch_fetch_and_update(
            tickers + ["^TWII"],
            period=settings.history_period,
            progress=price_progress,
        )

    # Determine the newest already-cached completed session before querying the
    # exchange.  During PREOPEN/INTRADAY this is a safe target hint (strictly
    # earlier than today) and prevents a stale "latest" OpenAPI response from
    # leaving one exchange one day behind the other.
    fallback_cutoff = _safe_cached_cutoff(store, today, str(phase), tickers)

    # EOD overlay is phase-aware.  Do not request today's unfinished daily bar
    # before/during trading.  After 13:30 request today's official close.
    official_target = session.get("official_target_date")
    if progress:
        progress("核對交易所完整收盤資料", 0.39)

    def official_progress(done, total, label):
        if progress:
            frac = float(done) / max(1.0, float(total))
            progress(str(label), 0.39 + 0.07 * max(0.0, min(1.0, frac)))

    official_request_target = str(official_target or fallback_cutoff or "") or None
    official_eod = store.overlay_official_eod(
        target_date=official_request_target,
        progress=official_progress,
    )

    # If the two exchanges disagree on their latest completed session, retry
    # once using the newer exchange date as an explicit date-specific target.
    # This directly handles the production failure where TWSE's "latest" cache
    # lagged one day while TPEx had already published the newer session.
    first_tw = str(official_eod.get("twse_date") or "")
    first_otc = str(official_eod.get("tpex_date") or "")
    first_idx = str(official_eod.get("index_date") or "")
    first_dates = [d for d in (first_tw, first_otc, first_idx) if d]
    if len(set(d for d in (first_tw, first_otc) if d)) > 1 and first_dates:
        retry_target = max(first_dates)
        if not today or retry_target <= today:
            if progress:
                progress(f"交易所日期不一致，重新核對 {retry_target}", 0.455)
            retried = store.overlay_official_eod(target_date=retry_target, progress=official_progress)
            retry_tw = str(retried.get("twse_date") or "")
            retry_otc = str(retried.get("tpex_date") or "")
            # Prefer the retry if it closes the gap or moves the lagging market
            # forward. Preserve first-call errors for diagnosis without letting
            # them erase a successful retry.
            first_gap = 0 if first_tw and first_otc and first_tw == first_otc else 1
            retry_gap = 0 if retry_tw and retry_otc and retry_tw == retry_otc else 1
            if retry_gap < first_gap or min([d for d in (retry_tw, retry_otc) if d], default="") > min([d for d in (first_tw, first_otc) if d], default=""):
                errs = list(official_eod.get("errors") or []) + list(retried.get("errors") or [])
                official_eod = dict(retried)
                official_eod["errors"] = list(dict.fromkeys(errs))
                official_eod["reconciled_retry"] = True
                official_eod["reconciled_target"] = retry_target

    # Never substitute a Yahoo/cache cutoff for the official-source date.
    # A missing exchange response must be visible to the data-health layer.
    raw_twse_cutoff = str(official_eod.get("twse_date") or "")
    raw_tpex_cutoff = str(official_eod.get("tpex_date") or "")
    raw_index_cutoff = str(official_eod.get("index_date") or "")

    # A cross-sectional ranking must never compare TWSE stocks from one session
    # with TPEx stocks from another.  First prefer the exchange-reconciled date;
    # if one source still lags, deliberately roll *all* markets back to the latest
    # common confirmed session rather than publishing a mixed-date model.
    synced = str(official_eod.get("synchronized_date") or "")
    candidates_for_sync = [d for d in (raw_twse_cutoff, raw_tpex_cutoff, raw_index_cutoff) if d]
    # On an unavailable endpoint, constrain the model to an actually cached
    # completed session; it remains explicitly UNVERIFIED in the health panel.
    if (not raw_twse_cutoff or not raw_tpex_cutoff) and fallback_cutoff:
        candidates_for_sync.append(fallback_cutoff)
    if not synced and candidates_for_sync:
        synced = min(candidates_for_sync)
    # If official clouds only serve an older date but independent cached
    # TWSE+TPEx OHLCV AND the index have a newer *completed* session, use the
    # recent data as a clearly labelled PROVISIONAL RESEARCH model. It must
    # never be labelled official/synchronized or used for an executable BUY.
    provisional_eod = False
    if (phase == POSTCLOSE and getattr(now, 'hour', 0) >= 16
            and hasattr(store, 'db_path') and fallback_cutoff
            and (not synced or fallback_cutoff > synced)):
        synced = fallback_cutoff
        provisional_eod = True
    if phase in {"PREOPEN", "INTRADAY"} and synced >= today:
        synced = fallback_cutoff
    if not synced:
        synced = fallback_cutoff
    if not synced:
        raise RuntimeError("交易所官方收盤與本機歷史快取均無可用的完整交易日；保留上一份已完成快照")

    twse_cutoff = synced
    tpex_cutoff = synced
    index_cutoff = synced
    official_eod["model_sync_date"] = synced
    official_eod["provisional_cache"] = bool(provisional_eod)
    official_eod["cache_observed_date"] = fallback_cutoff or ""
    official_eod["raw_twse_date"] = raw_twse_cutoff
    official_eod["raw_tpex_date"] = raw_tpex_cutoff
    official_eod["raw_index_date"] = raw_index_cutoff
    official_eod["date_provenance"] = {
        "twse": "exchange" if raw_twse_cutoff else "missing",
        "tpex": "exchange" if raw_tpex_cutoff else "missing",
        "index": "exchange" if raw_index_cutoff else "missing",
        "fallback_cache_date": fallback_cutoff or "",
        "model": "cross_market_cache_provisional" if provisional_eod else "official_exchange_date",
    }
    official_eod["model_date_aligned"] = bool(
        not provisional_eod and synced and raw_twse_cutoff == synced and raw_tpex_cutoff == synced and raw_index_cutoff >= synced
    )

    official_today_complete = bool(
        phase == POSTCLOSE
        and not provisional_eod
        and today
        and synced == today
        and raw_twse_cutoff == today
        and raw_tpex_cutoff == today
    )
    if phase == INTRADAY:
        analysis_mode = "INTRADAY_BASELINE"
    elif phase == POSTCLOSE and provisional_eod:
        analysis_mode = "POSTCLOSE_PROVISIONAL_RESEARCH_ONLY"
    elif phase == POSTCLOSE and official_today_complete:
        analysis_mode = "POSTCLOSE_FINAL"
    elif phase == POSTCLOSE:
        analysis_mode = "POSTCLOSE_PENDING"
    else:
        analysis_mode = "NEXT_SESSION_BASELINE"

    market = _market_context(store, end_date=index_cutoff or None)
    market_ret20 = float(market.get("ret20") or 0.0)
    # In the V18 peer-relative signal, an absent/stale benchmark is MISSING,
    # never a fabricated 0% index return.
    v18_index_ret20 = market_ret20 if str(market.get("price_date") or "") == str(index_cutoff or "") and str(index_cutoff or "") else None
    limitup_index_ret5 = market.get("ret5") if v18_index_ret20 is not None else None

    liquid = []
    valid_count = 0
    stale_count = 0
    history_insufficient_count = 0
    below_price_count = 0
    below_turnover_count = 0
    if progress:
        progress("篩選成交量足夠的股票", 0.48)
    universe_meta = {str(r.get("ticker")): r for r in universe.to_dict("records")}
    for ticker, raw_df in store.iter_prices(tickers, end_date=synced):
        cutoff = tpex_cutoff if str(ticker).endswith(".TWO") else twse_cutoff
        df = raw_df
        if cutoff and df is not None and not df.empty:
            df = df[df.index <= pd.Timestamp(cutoff)]
        if df is None or len(df) < int(CANONICAL_MODEL_POLICY.get("min_history_rows", 140)):
            history_insufficient_count += 1
            continue
        valid_count += 1
        latest_stock_date = str(df.index[-1].date())
        if cutoff and latest_stock_date < cutoff:
            stale_count += 1
            continue
        row = universe_meta.get(ticker, {})
        try:
            p = float(df["Close"].iloc[-1])
            avg_vol = float(df["Volume"].tail(20).mean())
            avg_turnover = float((df["Close"].tail(20) * df["Volume"].tail(20)).mean())
            if p < settings.min_price:
                below_price_count += 1
                continue
            baseline_turnover = float(CANONICAL_MODEL_POLICY.get("baseline_min_avg_turnover", settings.min_avg_turnover))
            if avg_turnover < baseline_turnover:
                below_turnover_count += 1
                continue
            liquid.append(
                {
                    "ticker": ticker,
                    "name": str(row.get("name", ticker)),
                    "industry": str(row.get("industry", "")),
                    "price": p,
                    "price_date": latest_stock_date,
                    "avg_volume_20d": avg_vol,
                    "avg_turnover_20d": avg_turnover,
                    "pre_score": _pre_score(df, market_ret20),
                    "fine_industry": fine_industry(ticker, str(row.get("name", ticker)), str(row.get("industry", ""))),
                    "v18_price": v18_price_signals(df, v18_index_ret20),
                    "limitup_price_setup": limitup_price_setup(
                        df, benchmark_ret5=limitup_index_ret5, benchmark_ret20=v18_index_ret20
                    ),
                    "pattern_technical": pattern_technicals(
                        df, benchmark_ret5=limitup_index_ret5, benchmark_ret20=v18_index_ret20
                    ),
                    "recent_sessions": [str(x.date()) for x in df.index[-20:]],
                    "df": df,
                }
            )
        except Exception:
            continue

    liquid.sort(key=lambda x: x["pre_score"], reverse=True)
    # Lightweight metadata for the one-minute full-market quote scan.  Keep the
    # historical DataFrame out of the snapshot; only the fields required to
    # normalize live volume/liquidity and identify the stock are persisted.
    intraday_universe = [
        {
            "ticker": c["ticker"],
            "name": c["name"],
            "industry": c["industry"],
            "price": c["price"],
            "price_date": c["price_date"],
            "avg_volume_20d": c["avg_volume_20d"],
            "avg_turnover_20d": c["avg_turnover_20d"],
            "pre_score": round(float(c["pre_score"]), 3),
        }
        for c in liquid
    ]
    candidates = model_candidates(liquid, settings)
    # Sector co-movement uses only the synchronized, investable completed-day
    # universe already in memory, with no additional network/API requests.
    v18_sector = v18_sector_context(liquid)
    v18_signals = {str(c["ticker"]): c for c in candidates}

    if progress:
        progress(f"比較走勢與歷史案例（{len(candidates)} 檔）", 0.66)
    evaluated: list[dict] = []
    benchmark_df = store.get_prices("^TWII", end_date=index_cutoff or None)
    market_day_change = float(market.get("day_change_pct") or 0.0)
    candidate_total = max(1, len(candidates))

    # Same completed session => the expensive price/analog model is identical.
    # Reuse the previous V17 completed-day core and recompute only ranking research.
    previous_by_ticker: dict[str, dict] = {}
    if bool(getattr(settings, "canonical_mode", True)):
        previous = load_dashboard(data_dir / "dashboard_snapshot.json")
        if isinstance(previous, dict) and previous.get("model_version") in {OPERATIONS_VERSION, "v18.20.0-limitup-shadow", "v18.12.0-short-return-tiers", "v17.1.0-synced-briefing"}:
            prev_date = str(previous.get("price_date") or "")
            current_dates = [str(c.get("price_date") or "") for c in candidates if c.get("price_date")]
            current_date = max(current_dates) if current_dates else ""
            if prev_date and current_date and prev_date == current_date:
                previous_by_ticker = {
                    str(x.get("ticker")): x for x in (previous.get("stocks") or [])
                    if isinstance(x, dict) and x.get("ticker") and str(x.get("price_date") or "") == current_date
                }
    reused_core_n = 0

    for idx, c in enumerate(candidates, 1):
        cached = previous_by_ticker.get(c["ticker"])
        if cached and isinstance(cached.get("horizons"), dict) and all(h in cached["horizons"] for h in ("short", "mid", "long")):
            stock = copy.deepcopy(cached)
            stock["name"] = c["name"]; stock["industry"] = c["industry"]
            stock["price"] = c["price"]; stock["price_date"] = c["price_date"]
            stock["avg_volume_20d"] = c["avg_volume_20d"]; stock["avg_turnover_20d"] = c["avg_turnover_20d"]
            stock["research"] = {}; stock["ranking_research"] = {}; stock["evidence"] = {}
            evaluated.append(stock)
            reused_core_n += 1
        else:
            df = c["df"]
            horizons = {}
            estimates = estimate_all_horizons(
                df, settings, twii_ret_20d=market_ret20, benchmark_df=benchmark_df
            )
            for h in ["short", "mid", "long"]:
                plan = generate_trade_plan(df, h)
                state = evaluate_entry_state(df, plan)
                est = estimates.get(h) or {}
                timing = entry_timing_from_df(df, plan, h, market_change_pct=market_day_change)
                horizons[h] = {
                    "plan": plan,
                    "entry_state": state,
                    "entry_timing": timing,
                    "position_guidance": position_guidance(market.get("regime"), h, timing),
                    "forecast": est,
                    "qualification": {
                        "research_qualified": bool(est.get("composite_factor_score", 0) >= 35),
                        "rank_first": True,
                    },
                }
            evaluated.append(
                {
                    "ticker": c["ticker"],
                    "name": c["name"],
                    "industry": c["industry"],
                    "fine_industry": fine_industry(c["ticker"], c["name"], c["industry"]),
                    "price": c["price"],
                    "price_date": c["price_date"],
                    "avg_volume_20d": c["avg_volume_20d"],
                    "avg_turnover_20d": c["avg_turnover_20d"],
                    "setup": setup_from_df(df),
                    "horizons": horizons,
                    "research": {},
                    "evidence": {},
                }
            )
        # All sector/price-only context is computed before this loop. Release
        # each 5y OHLCV frame after computing its short/mid/long forecasts;
        # never persist per-stock DataFrames in the JSON snapshot.
        c.pop("df", None)
        if progress and (idx == 1 or idx == candidate_total or idx % 25 == 0):
            progress(
                f"比較走勢與歷史案例（{idx}/{candidate_total} 檔；重用 {reused_core_n}）",
                0.66 + 0.19 * (idx / candidate_total),
            )

    # V17 comparable breadth evidence: one official market-wide snapshot is
    # attached to every evaluated stock before ranking.  Rich per-stock history
    # is fetched later for display only and cannot move the published ranking.
    breadth_asof = min([d for d in (twse_cutoff, tpex_cutoff) if d], default=latest_date if 'latest_date' in locals() else "")
    breadth: dict[str, dict] = {}
    if bool(CANONICAL_MODEL_POLICY.get("research_breadth", True)) and evaluated:
        if progress:
            progress(f"建立全市場可比公司/法人證據（{len(evaluated)} 檔）", 0.86)
        try:
            breadth = _load_breadth_research(
                [s["ticker"] for s in evaluated], data_dir, asof_date=breadth_asof or None
            )
        except Exception:
            breadth = {}

    def apply_scores(stock):
        for h in ["short", "mid", "long"]:
            _combined_ranking_score(stock, h, settings.model_family)

    for stock in evaluated:
        comparable = breadth.get(stock["ticker"]) or {}
        stock["ranking_research"] = comparable
        # Breadth data is immediately displayable; richer top-name detail may
        # replace only the display object below, never ranking_research.
        stock["research"] = comparable
        # Apply factors only to official market-wide comparable research. Rich
        # details for front stocks are display-only and cannot move ranking.
        core = v18_signals.get(str(stock["ticker"])) or {}
        stock["limitup_setup_raw"] = core.get("limitup_price_setup") or {}
        stock["pattern_technical_raw"] = core.get("pattern_technical") or {}
        stock["pattern_recent_sessions"] = core.get("recent_sessions") or []
        stock["v18_profile"] = build_v18_profile(
            core.get("v18_price") or {}, comparable,
            v18_sector.get(str(stock["ticker"])) or {},
            avg_vol_shares=float(stock.get("avg_volume_20d") or 0.0),
        )
        apply_scores(stock)

    # V18.4: all-market rank robustness, computed from the SAME snapshot without
    # additional provider requests. MC/eligibility change; Alpha ranking does not.
    from v18_rank_stability import apply_rank_stability
    rank_robustness = apply_rank_stability(evaluated)

    # Rich historical revenue/EPS/flow is loaded symmetrically for a wider front
    # set for cards and diagnosis.  It does not alter cross-sectional ranking.
    front = []
    top_detail = int(CANONICAL_MODEL_POLICY.get("research_detail_top_per_horizon", 20))
    for h in ["short", "mid", "long"]:
        front.extend([s["ticker"] for s in sorted(
            evaluated, key=lambda x: x["horizons"][h].get("ranking_score", -1), reverse=True
        )[:top_detail]])
    front = list(dict.fromkeys(front))
    if front:
        _enrich_tickers(evaluated, data_dir, front, progress=progress, progress_value=0.93)

    if os.getenv("ENABLE_BRANCH_FLOW", "0") == "1":
        top_tickers = []
        for h in ["short", "mid", "long"]:
            top_tickers.extend([s["ticker"] for s in sorted(evaluated, key=lambda x: x["horizons"][h].get("ranking_score", -1), reverse=True)[:5]])
        top_tickers = list(dict.fromkeys(top_tickers))
        by_ticker = {s["ticker"]: s for s in evaluated}
        if progress:
            progress(f"補充券商分點（{len(top_tickers)} 檔）", 0.96)
        def branch_one(ticker):
            client = ResearchDataClient(data_dir / "research_cache.sqlite")
            return ticker, client.branch_main_force_proxy(ticker.split(".")[0], max_sessions=20)
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(top_tickers)))) as ex:
            futs = [ex.submit(branch_one, t) for t in top_tickers]
            for fut in as_completed(futs):
                try:
                    ticker, branch = fut.result()
                    if ticker in by_ticker:
                        by_ticker[ticker].setdefault("research", {})["main_force_proxy"] = branch
                        apply_scores(by_ticker[ticker])
                except Exception:
                    continue

    for stock in evaluated:
        research = stock.get("research") or {}
        ranking_research = stock.get("ranking_research") or {}
        stock["evidence"] = {
            "revenue": bool((research.get("monthly_revenue") or {}).get("available")),
            "financials": bool((research.get("financials") or {}).get("available")),
            "valuation": bool((research.get("valuation") or {}).get("available")),
            "institutional_flow": bool((research.get("institutional_flow") or {}).get("available")),
            "main_force_proxy": bool((research.get("main_force_proxy") or {}).get("available")),
        }
        stock["ranking_evidence"] = {
            "revenue": bool((ranking_research.get("monthly_revenue") or {}).get("available")),
            "financials": bool((ranking_research.get("financials") or {}).get("available")),
            "valuation": bool((ranking_research.get("valuation") or {}).get("available")),
            "institutional_flow": bool((ranking_research.get("institutional_flow") or {}).get("available")),
            "breadth_comparable": bool(ranking_research.get("breadth_comparable")),
        }

    breadth_factor_counts = {
        "revenue": sum(1 for x in evaluated if ((x.get("ranking_research") or {}).get("monthly_revenue") or {}).get("available")),
        "financials": sum(1 for x in evaluated if ((x.get("ranking_research") or {}).get("financials") or {}).get("available")),
        "valuation": sum(1 for x in evaluated if ((x.get("ranking_research") or {}).get("valuation") or {}).get("available")),
        "institutional_flow": sum(1 for x in evaluated if ((x.get("ranking_research") or {}).get("institutional_flow") or {}).get("available")),
    }
    breadth_usable_n = sum(1 for x in evaluated if any(((x.get("ranking_evidence") or {}).get(k)) for k in ("revenue","financials","valuation","institutional_flow")))
    for stock in evaluated:
        stock["selection_quality"] = {}
        for h in ("short", "mid", "long"):
            ok, reasons = _selection_quality(stock, h)
            stock["selection_quality"][h] = {"eligible": bool(ok), "reasons": reasons}
        stock["limitup_shadow"] = limitup_decorate(
            stock, stock.pop("limitup_setup_raw", {}),
            sector=v18_sector.get(str(stock.get("ticker") or "")),
        )
        # Comparable all-market evidence only: rich per-stock detail cannot
        # retroactively move one candidate ahead of peers without that data.
        stock["pattern_chip_shadow"] = pattern_candidate(
            stock, stock.pop("pattern_technical_raw", {}),
            research=stock.get("ranking_research") or {},
            trading_dates=stock.pop("pattern_recent_sessions", []),
        )
    selection_eligible_n = selection_universe_stats(evaluated)

    if progress:
        progress("完成", 1.0)
    dates = [s["price_date"] for s in evaluated if s.get("price_date")]
    latest_date = max(dates) if dates else str(market.get("price_date") or "")
    common_complete_date = str(synced or latest_date)
    session_meta = dict(session)
    session_meta.update({
        "analysis_mode": analysis_mode,
        "twse_complete_date": twse_cutoff,
        "tpex_complete_date": tpex_cutoff,
        "index_complete_date": index_cutoff,
        "common_complete_date": common_complete_date,
        "official_today_complete": official_today_complete,
        "postclose_pending": bool(phase == POSTCLOSE and not official_today_complete),
        "provisional_cache": bool(provisional_eod),
    })
    snap = {
        "snapshot_id": f"snap_{_taipei_timestamp().strftime('%Y%m%d_%H%M%S')}",
        "generated_at": _taipei_timestamp().isoformat(timespec="seconds"),
        "price_date": latest_date,
        "market": market,
        "session": session_meta,
        "coverage": {
            "requested": len(universe),
            "twse_universe": int(getattr(universe, "attrs", {}).get("twse_count", 0) or sum(str(t).endswith(".TW") for t in tickers)),
            "tpex_universe": int(getattr(universe, "attrs", {}).get("tpex_count", 0) or sum(str(t).endswith(".TWO") for t in tickers)),
            "downloaded": int(dl.get("downloaded_tickers", 0)),
            "feature_valid": valid_count,
            "investable_model": len(liquid),
            "liquid": len(liquid),
            "history_insufficient": history_insufficient_count,
            "below_price_excluded": below_price_count,
            "below_turnover_excluded": below_turnover_count,
            "stale_excluded": stale_count,
            "expected_date": common_complete_date,
            "calendar_expected_date": official_target or "",
            "errors": dl.get("errors", []) + official_eod.get("errors", []),
            "model_core": {
                "evaluated_all_liquid": bool(getattr(settings, "canonical_mode", True)),
                "reused_same_date": int(reused_core_n),
                "newly_computed": max(0, len(evaluated) - int(reused_core_n)),
            },
            "history_refresh": {
                "mode": dl.get("mode"),
                "requested": int(dl.get("requested", 0) or 0),
                "downloaded_tickers": int(dl.get("downloaded_tickers", 0) or 0),
                "covered_ratio": dl.get("covered_ratio"),
                "last_full_refresh": dl.get("last_full_refresh"),
            },
            "official_eod": official_eod,
        },
        "candidate_n": len(evaluated),
        "full_universe_model_n": len(evaluated),
        "canonical_model": bool(getattr(settings, "canonical_mode", True)),
        "research_breadth_n": int(breadth_usable_n),
        "research_breadth_factor_counts": breadth_factor_counts,
        "selection_eligible_n": selection_eligible_n,
        "pattern_chip_summary": {
            "version": PATTERN_VERSION, "research_only": True,
            "reversal_weights": dict(REVERSAL_WEIGHTS),
            "breakout_weights": dict(BREAKOUT_WEIGHTS),
            "available_n": sum(bool((s.get("pattern_chip_shadow") or {}).get("available")) for s in evaluated),
            "eligible_n": sum(bool((s.get("pattern_chip_shadow") or {}).get("eligible_research")) for s in evaluated),
            "note": "K線雙底／KD／MACD／RSI／價量／營收／籌碼；起始權重尚未經樣本外校準，不提供買賣指令。",
        },
        "limitup_shadow_summary": {
            "version": LIMITUP_LAB_VERSION,
            "weights": dict(LIMITUP_FACTOR_WEIGHTS), "blend": dict(LIMITUP_BLEND_WEIGHTS),
            "research_only": True, "calibrated_probability": False,
            "raw_profiles": sum(bool((s.get("limitup_shadow") or {}).get("setup", {}).get("available")) for s in evaluated),
            "eligible_n": sum(bool((s.get("limitup_shadow") or {}).get("eligible_research")) for s in evaluated),
            "note": "短線1–5交易日漲停前兆觀察；獨立於正式Alpha排名與即時買點，尚無樣本外績效證明。",
        },
        "v18_factor_summary": {
            "model": "V18.4 integrated evidence/PIT blend",
            "rank_robustness": rank_robustness,
            "avg_dq_short": round(sum(float(((x.get('horizons') or {}).get('short') or {}).get('rank_v18', {}).get('dq_score') or 0) for x in evaluated) / max(1,len(evaluated)), 1),
            "pit_mode": "live_observation_not_historical_replay",
            "positive_factors": 13, "entry_risk_factors": 2,
            "avg_observed_factors": round(sum(int(s.get("v18_profile", {}).get("evidence_n") or 0) for s in evaluated)/max(1,len(evaluated)), 2),
            "stocks_with_sector_peers": sum(bool(v18_sector.get(str(s["ticker"]), {}).get("available")) for s in evaluated),
            "note": "未取得的大戶/融資/現金流證據不予虛構；宏觀與過熱僅進場風險提示",
        },
        "universe_pit_archive": universe_pit,
        "intraday_universe_n": len(intraday_universe),
        "intraday_universe": intraday_universe,
        "stocks": evaluated,
        "settings": asdict(settings),
        "source_type": "v18_full_universe_multifactor_evidence",
        "data_integrity_schema": "v18.5-source-provenance",
        "model_version": OPERATIONS_VERSION,
        "charts": {},
    }
    from v174_integrity import assess_snapshot
    audit = assess_snapshot(snap)
    if audit["status"] == "BLOCK":
        # Do not overwrite the previous complete snapshot with a zero-stock or
        # cross-date partial scan, even if the provider returned HTTP 200.
        raise RuntimeError("本次完整日線資料未通過核對：" + "；".join(audit["blockers"][:2]))
    previous_publication = load_dashboard(data_dir / "dashboard_snapshot.json")
    previous_day = str((previous_publication or {}).get("price_date") or "")
    if (previous_publication and previous_day and previous_day > str(snap.get("price_date") or "")
        and assess_snapshot(previous_publication).get("status") != "BLOCK"):
        raise RuntimeError("本次來源只取得較舊的交易日；為避免模型日期倒退，仍保留上一份完成快照")
    # If saving fails, surface an explicit job failure instead of telling the
    # user the refresh was complete while the UI still shows the old snapshot.
    from scan_job_manager import _atomic_json
    _atomic_json(data_dir / "dashboard_snapshot.json", compact_session_dashboard(snap))
    return snap


def _selection_quality(stock: dict, horizon: str) -> tuple[bool, list[str]]:
    """Data-quality/investability guardrail for published Top-N lists.

    This does not change a stock's ranking score. It prevents a very illiquid or
    evidence-poor name from being presented as a canonical "best" candidate for
    a horizon merely because the raw technical score is extreme.
    """
    reasons: list[str] = []
    if horizon not in {"short", "mid", "long"}:
        return False, ["UNKNOWN_HORIZON"]
    block = (stock.get("horizons") or {}).get(horizon) or {}
    comp = block.get("score_components") or {}
    forecast = block.get("forecast") or {}
    # Compatibility for old/synthetic snapshots that predate the V17 quality
    # metadata. Production V17.1 stocks always carry forecast/components.
    if not comp and not forecast:
        return True, []
    try:
        turnover = float(stock.get("avg_turnover_20d") or 0.0)
    except Exception:
        turnover = 0.0
    min_turnover = float((CANONICAL_MODEL_POLICY.get("horizon_min_avg_turnover") or {}).get(horizon, 0.0))
    if turnover < min_turnover:
        reasons.append("LIQUIDITY")
    try:
        coverage = float(comp.get("evidence_coverage") or 0.0)
    except Exception:
        coverage = 0.0
    min_cov = float((CANONICAL_MODEL_POLICY.get("horizon_min_evidence_coverage") or {}).get(horizon, 0.0))
    if coverage < min_cov:
        reasons.append("EVIDENCE_COVERAGE")
    if not forecast.get("estimate_available"):
        reasons.append("NO_EMPIRICAL_ESTIMATE")
    if horizon == "long" and bool(CANONICAL_MODEL_POLICY.get("long_require_company_evidence", True)):
        ev = stock.get("ranking_evidence") or {}
        company_ok = bool(ev.get("financials")) and bool(ev.get("revenue") or ev.get("valuation"))
        if not company_ok:
            reasons.append("COMPANY_EVIDENCE")
    return not reasons, reasons


def selection_universe_stats(stocks: list[dict]) -> dict[str, int]:
    return {h: sum(1 for s in stocks if _selection_quality(s, h)[0]) for h in ("short", "mid", "long")}

def select_market_best(snap: dict | None, horizon: str, n: int = 5) -> list:
    """Return canonical top-N after horizon-specific investability guardrails."""
    if not snap or not isinstance(snap, dict):
        return []
    stocks = [s for s in snap.get("stocks", []) if isinstance(s, dict)]
    ranked = sorted(
        stocks,
        key=lambda x: x.get("horizons", {}).get(horizon, {}).get("ranking_score", -1),
        reverse=True,
    )
    eligible = []
    for stock in ranked:
        ok, reasons = _selection_quality(stock, horizon)
        stock.setdefault("selection_quality", {})[horizon] = {"eligible": ok, "reasons": reasons}
        if ok:
            eligible.append(stock)
        if len(eligible) >= int(n):
            break
    return eligible[: max(0, int(n))]


def select_view(snap: dict | None, horizon: str, qualified: bool = True, n: int = 5, **kwargs) -> list:
    return select_market_best(snap, horizon, n)


def select_cross_horizon_shortlists(
    snap: dict | None,
    n: int = 5,
    max_appearances: int = 2,
    horizons: tuple[str, ...] = ("short", "mid", "long"),
) -> dict[str, list]:
    """Build diversified per-horizon shortlists without changing signal ranking.

    This is a portfolio-allocation helper only.  ``select_market_best`` remains
    the source of the true signal ranking.  A ticker may appear in at most
    ``max_appearances`` horizon lists; when a shared leader reaches that cap, the
    next-highest stock for the affected horizon is used.

    Horizons are filled round-robin with a rotating starting horizon so the
    de-duplication rule does not systematically privilege the short list.
    """
    hs = tuple(h for h in horizons if h in {"short", "mid", "long"})
    if not hs:
        return {}
    try:
        n = max(0, int(n))
        cap = max(1, int(max_appearances))
    except Exception:
        n, cap = 5, 2
    if n == 0 or not snap or not isinstance(snap, dict):
        return {h: [] for h in hs}

    stocks = [s for s in snap.get("stocks", []) if isinstance(s, dict)]
    ranked: dict[str, list] = {
        h: [
            s for s in sorted(
                stocks,
                key=lambda x: x.get("horizons", {}).get(h, {}).get("ranking_score", -1),
                reverse=True,
            )
            if _selection_quality(s, h)[0]
        ]
        for h in hs
    }
    selected: dict[str, list] = {h: [] for h in hs}
    selected_codes: dict[str, set[str]] = {h: set() for h in hs}
    appearances: dict[str, int] = {}

    for slot in range(n):
        # Rotate which horizon gets first choice at each rank slot.
        shift = slot % len(hs)
        order = hs[shift:] + hs[:shift]
        for h in order:
            if len(selected[h]) >= n:
                continue
            choice = None
            for stock in ranked[h]:
                code = str(stock.get("ticker") or "").strip()
                if not code or code in selected_codes[h]:
                    continue
                if appearances.get(code, 0) >= cap:
                    continue
                choice = stock
                break
            if choice is None:
                continue
            code = str(choice.get("ticker") or "").strip()
            selected[h].append(choice)
            selected_codes[h].add(code)
            appearances[code] = appearances.get(code, 0) + 1

    return selected


def shortlist_overlap_metrics(shortlists: dict | None) -> dict:
    """Summarize overlap in a cross-horizon allocation result.

    The helper is intentionally descriptive: it never feeds back into ranking.
    """
    if not isinstance(shortlists, dict):
        shortlists = {}
    counts: dict[str, int] = {}
    total_slots = 0
    for picks in shortlists.values():
        if not isinstance(picks, (list, tuple)):
            continue
        for stock in picks:
            if not isinstance(stock, dict):
                continue
            code = str(stock.get("ticker") or "").strip()
            if not code:
                continue
            total_slots += 1
            counts[code] = counts.get(code, 0) + 1

    unique = len(counts)
    duplicate_slots = max(0, total_slots - unique)
    multi = {k: v for k, v in counts.items() if v > 1}
    return {
        "total_slots": total_slots,
        "unique_tickers": unique,
        "duplicate_slots": duplicate_slots,
        "overlap_ratio": round(duplicate_slots / total_slots, 4) if total_slots else 0.0,
        "max_appearances": max(counts.values(), default=0),
        "multi_horizon_tickers": multi,
    }


def _evaluate_one(
    ticker: str,
    name: str,
    industry: str,
    df: pd.DataFrame,
    settings: RunSettings,
    market_ret20: float,
    research: dict,
    ranking_research: dict | None = None,
    market_regime: str = "UNKNOWN",
    market_day_change: float = 0.0,
    benchmark_df: pd.DataFrame | None = None,
) -> dict:
    price = float(df["Close"].iloc[-1])
    stock = {
        "ticker": ticker,
        "name": name,
        "industry": industry,
        "fine_industry": fine_industry(ticker, name, industry),
        "price": price,
        "price_date": str(df.index[-1].date()),
        "avg_volume_20d": float(df["Volume"].tail(20).mean()),
        "avg_turnover_20d": float((df["Close"].tail(20) * df["Volume"].tail(20)).mean()),
        "setup": setup_from_df(df),
        "horizons": {},
        "research": research or {},
        "ranking_research": ranking_research if ranking_research is not None else (research or {}),
    }
    estimates = estimate_all_horizons(df, settings, twii_ret_20d=market_ret20, benchmark_df=benchmark_df)
    for h in ["short", "mid", "long"]:
        plan = generate_trade_plan(df, h)
        est = estimates.get(h) or {}
        timing = entry_timing_from_df(df, plan, h, market_change_pct=market_day_change)
        stock["horizons"][h] = {
            "plan": plan,
            "entry_state": evaluate_entry_state(df, plan),
            "entry_timing": timing,
            "position_guidance": position_guidance(market_regime, h, timing),
            "forecast": est,
            "qualification": {"research_qualified": bool(est.get("composite_factor_score", 0) >= 35), "rank_first": True},
        }
        _combined_ranking_score(stock, h, settings.model_family)
    return stock


def research_on_demand(ticker: str, data_dir: Path, include_branch: bool = False) -> dict:
    """Fetch/cache research detail for one stock without rerunning the market model.

    This path is display-only.  It is intentionally separate from ranking so an
    on-demand UI click can never change the already-published short/mid/long
    scores.  ResearchDataClient itself keeps a SQLite cache and fails open when a
    provider is unavailable.
    """
    clean = str(ticker or "").strip()
    if not clean:
        return {"error": "EMPTY_TICKER", "source": "research unavailable"}
    try:
        client = ResearchDataClient(Path(data_dir) / "research_cache.sqlite")
        if hasattr(client, "stock_research_detail"):
            return client.stock_research_detail(clean, include_branch=bool(include_branch))
        return client.stock_research(clean, include_branch=bool(include_branch))
    except Exception as exc:
        return {
            "stock_id": clean.split(".")[0],
            "error": f"{type(exc).__name__}: {exc}",
            "monthly_revenue": {"available": False, "rows": []},
            "financials": {"available": False, "quarters": []},
            "valuation": {"available": False},
            "institutional_flow": {"available": False},
            "main_force_proxy": {"available": False, "reason": "NOT_REQUESTED"},
            "source": "research unavailable",
        }


def diagnose(code: str, snap: dict | None, data_dir: Path, *, quick_research: bool = False) -> dict:
    """Evaluate one stock on the same completed-day coordinate as the dashboard.

    Important V16.7 guardrail: a direct diagnosis never lets an unfinished
    intraday daily candle leak into the completed-day model.  When a dashboard
    snapshot exists its ``price_date`` is the authoritative cutoff.  Otherwise
    PREOPEN/INTRADAY fall back to the latest cached session strictly before today.
    """
    clean = str(code or "").strip()
    if not clean:
        return {"snapshot_id": "snap_none", "error": "EMPTY_CODE"}
    snap_id = snap.get("snapshot_id", "snap_unknown") if isinstance(snap, dict) else "snap_none"
    snapshot_cutoff = str((snap or {}).get("price_date") or "") if isinstance(snap, dict) else ""

    for item in ((snap or {}).get("stocks", []) if isinstance(snap, dict) else []):
        if isinstance(item, dict) and (item.get("ticker") == clean or str(item.get("ticker", "")).startswith(clean + ".")):
            return {
                "snapshot_id": snap_id,
                "stock": dict(item),
                "analysis_date": str(item.get("price_date") or snapshot_cutoff or ""),
                "source": "dashboard_snapshot",
            }

    ticker_options = [clean] if "." in clean else [f"{clean}.TW", f"{clean}.TWO"]
    store = DailyPriceStore(Path(data_dir) / "daily_prices.sqlite")

    # V18.17: on-demand doctor must NOT block on unrelated full-market EOD
    # provider probes when the dashboard already defines a fixed model date.
    # Prefer existing >=260-day price history of the same model date. A network
    # failure is not fatal when older cached history can still be labelled as
    # research; missing or mismatched dates will be denied live BUY by the UI.
    cached_complete = False
    if snapshot_cutoff:
        for t in ticker_options:
            try:
                cached = store.get_prices(t, end_date=snapshot_cutoff)
                if (cached is not None and len(cached) >= 260
                        and str(cached.index[-1].date()) == snapshot_cutoff):
                    cached_complete = True
                    break
            except Exception:
                continue
    if not cached_complete:
        try:
            store.batch_fetch_and_update(ticker_options, period="5y")
        except Exception:
            # UI can report an honest PRICE_DATA_UNAVAILABLE with existing cache
            # instead of killing the entire individual-stock diagnostic page.
            pass

    now = _taipei_timestamp()
    phase = session_reference_policy(now)
    official = {}
    if not snapshot_cutoff and phase.get("phase") == "POSTCLOSE":
        try:
            official = store.overlay_official_eod(target_date=str(phase.get("official_target_date") or "")) or {}
        except Exception:
            official = {}

    ticker = None
    df = pd.DataFrame()
    for t in ticker_options:
        trial = store.get_prices(t)
        if trial is None or trial.empty:
            continue
        trial = trial.sort_index()
        cutoff = snapshot_cutoff
        if not cutoff:
            dates = [str(x.date()) for x in trial.index]
            today = now.date().isoformat()
            if phase.get("phase") in {"PREOPEN", "INTRADAY"}:
                older = [d for d in dates if d < today]
                cutoff = max(older) if older else ""
            elif phase.get("phase") == "POSTCLOSE":
                exchange_date = str(official.get("tpex_date") if t.endswith(".TWO") else official.get("twse_date") or "")
                cutoff = exchange_date or (max(dates) if dates else "")
            else:
                cutoff = max(dates) if dates else ""
        if cutoff:
            trial = trial[trial.index <= pd.Timestamp(cutoff)]
        if not trial.empty:
            ticker, df = t, trial
            snapshot_cutoff = str(trial.index[-1].date())
            break
    if ticker is None or df.empty:
        return {"snapshot_id": snap_id, "error": "PRICE_DATA_UNAVAILABLE", "code": clean}

    settings_dict = snap.get("settings", {}) if isinstance(snap, dict) else {}
    try:
        allowed = set(RunSettings.__dataclass_fields__)
        settings = RunSettings(**{k: v for k, v in settings_dict.items() if k in allowed})
    except Exception:
        settings = RunSettings()
    market_ctx = (snap.get("market", {}) if isinstance(snap, dict) else {}) or {}
    market_ret20 = float(market_ctx.get("ret20") or 0.0)
    market_regime = str(market_ctx.get("regime") or "UNKNOWN")
    market_day_change = float(market_ctx.get("day_change_pct") or 0.0)
    research, ranking_research = {}, {}
    if not quick_research:
        try:
            client = ResearchDataClient(Path(data_dir) / "research_cache.sqlite")
            research = client.stock_research(
                ticker, include_branch=(os.getenv("ENABLE_BRANCH_FLOW", "0") == "1")
            )
            ranking_research = (client.official_breadth_snapshot([ticker], asof_date=snapshot_cutoff or None) or {}).get(ticker) or {}
        except Exception:
            research = {}
            ranking_research = {}
    # Quick on-demand doctor derives price/structure scores immediately from
    # the cached daily series; the detail toggle fetches company/flow reports.
    # It is deliberately distinct from ranked snapshot stocks, whose full
    # comparable PIT research already arrived in the frozen market model.

    # Network-only universe lookups are intentionally omitted here: the stock
    # code is a grounded identity, and no slow provider call should be required
    # just to show a display name. Model stocks already return by snapshot above.
    display_name, broad_industry = clean, "個股診斷"
    benchmark_for_doctor = store.get_prices("^TWII", end_date=snapshot_cutoff or None)
    stock = _evaluate_one(
        ticker, display_name, broad_industry, df, settings, market_ret20, research,
        ranking_research=ranking_research, market_regime=market_regime, market_day_change=market_day_change,
        benchmark_df=benchmark_for_doctor
    )
    return {
        "snapshot_id": snap_id,
        "stock": stock,
        "analysis_date": str(df.index[-1].date()),
        "source": "on_demand_cached_price_study" if quick_research else "on_demand_completed_day",
        "research_limited": bool(quick_research),
    }


def diagnose_with_live(code: str, snap: dict | None, data_dir: Path, horizon: str = "short", now=None) -> dict:
    """Single-stock diagnosis with a realtime overlay only during market hours."""
    dr = diagnose(code, snap, data_dir, quick_research=True)
    if dr.get("error") or not dr.get("stock"):
        return dr
    phase = session_reference_policy(now or _taipei_timestamp())
    dr["phase"] = phase
    dr["live_status"] = {"available": False, "reason": "NOT_INTRADAY", "source": "completed-day model"}
    if phase.get("phase") != "INTRADAY":
        return dr
    try:
        from intraday_engine import FinMindRealtimeClient, rerank_snapshot
        stock = dr["stock"]
        ticker = str(stock.get("ticker") or "")
        qdf, status = FinMindRealtimeClient().verified_snapshots([ticker])
        dr["live_status"] = {
            "available": bool(getattr(status, "available", False)),
            "verified": bool(getattr(status, "verified", False)),
            "verification": str(getattr(status, "verification", "") or ""),
            "reason": str(getattr(status, "reason", "") or ""),
            "source": str(getattr(status, "source", "") or ""),
            "fetched_at": str(getattr(status, "fetched_at", "") or ""),
            "quote_date": str(getattr(status, "quote_date", "") or ""),
        }
        # Only an official-verified quote may replace the completed-day price in
        # Doctor.  A provider-only quote can be shown as unavailable evidence,
        # but must never be labelled as an official intraday execution price.
        if getattr(status, "available", False) and getattr(status, "verified", False) and qdf is not None and not qdf.empty:
            mini = {
                "stocks": [stock],
                "market": ((snap or {}).get("market") or {}) if isinstance(snap, dict) else {},
                "snapshot_id": dr.get("snapshot_id", "doctor"),
            }
            live_rows = rerank_snapshot(mini, horizon, qdf, top_n=1, now=phase.get("now") and now)
            if live_rows:
                dr["stock"] = live_rows[0]
            else:
                # The MIS request itself may have succeeded while this specific
                # ticker had no latest matched price.  Do not leave a batch-level
                # verified flag that could make yesterday's close look live.
                dr["live_status"].update({
                    "available": False,
                    "verified": False,
                    "reason": "NO_EXECUTABLE_PRICE_FOR_TICKER",
                    "verification": "official_mis_no_ticker_price",
                })
    except Exception as exc:
        dr["live_status"] = {
            "available": False,
            "reason": f"{type(exc).__name__}",
            "source": "realtime unavailable",
        }
    return dr
