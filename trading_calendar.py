"""Trading-session phase and daily freshness helpers for Alpha Radar V16.6.

The trading-session helper deliberately separates four concerns:
- PREOPEN: use the latest *completed* daily session to prepare a watchlist.
- INTRADAY: keep the completed daily model frozen and overlay realtime quotes.
- POSTCLOSE: try to finalize today's official EOD data for tomorrow's plan.
- OFFDAY: reuse the latest completed session without pretending there is a new bar.

Holiday truth ultimately comes from the exchange EOD dates returned by the data
layer; weekday arithmetic here is only a UI/update-policy hint.
"""
from __future__ import annotations

from pathlib import Path
import datetime
import json

TZ_TAIPEI = datetime.timezone(datetime.timedelta(hours=8))
PREOPEN = "PREOPEN"
INTRADAY = "INTRADAY"
POSTCLOSE = "POSTCLOSE"
OFFDAY = "OFFDAY"


def _now(now: datetime.datetime | None = None) -> datetime.datetime:
    now = now or datetime.datetime.now(TZ_TAIPEI)
    if now.tzinfo is None:
        now = now.replace(tzinfo=TZ_TAIPEI)
    return now.astimezone(TZ_TAIPEI)


def market_phase(now: datetime.datetime | None = None) -> str:
    now = _now(now)
    if now.weekday() >= 5:
        return OFFDAY
    t = now.timetz().replace(tzinfo=None)
    if t < datetime.time(9, 0):
        return PREOPEN
    if t <= datetime.time(13, 30):
        return INTRADAY
    return POSTCLOSE


def session_reference_policy(now: datetime.datetime | None = None) -> dict:
    """Return update/render policy for the current Taiwan-market phase.

    `official_target_date` is intentionally today's date only after the close.
    Before/during the session we request the exchange's latest completed EOD
    snapshot instead of asking for today's unfinished bar.
    """
    now = _now(now)
    phase = market_phase(now)
    today = now.date().isoformat()
    if phase == PREOPEN:
        label = "盤前"
        update_label = "更新盤前基準與關注清單"
        notice = "以最近完整交易日分析今日盤前可關注標的。"
    elif phase == INTRADAY:
        label = "盤中"
        update_label = "更新完整基準資料（盤中買點自動即時）"
        notice = "完整日線模型固定在最近收盤；盤中另以即時價量尋找可執行買點。"
    elif phase == POSTCLOSE:
        label = "盤後"
        update_label = "更新盤後資料與下一交易日關注清單"
        notice = "優先納入今日官方收盤資料；若交易所尚未發布，先保留最近完整日線並標示待更新。"
    else:
        label = "休市"
        update_label = "更新最近完整交易日與下一交易日清單"
        notice = "休市期間不製造新日K，使用最近完整交易日規劃下一交易日。"
    return {
        "phase": phase,
        "label": label,
        "today": today,
        "now": now.isoformat(timespec="seconds"),
        "market_open": phase == INTRADAY,
        "official_target_date": today if phase == POSTCLOSE else None,
        "allow_realtime": phase == INTRADAY,
        "allow_postclose_quote_preview": phase == POSTCLOSE,
        "update_label": update_label,
        "notice": notice,
    }


def calendar_reference(data_dir: Path, now: datetime.datetime | None = None, allow_fetch: bool = False) -> dict:
    """Backward-compatible calendar object used by older callers/tests.

    `expected_date` is no longer treated as an authoritative completed-session
    date. The exchange EOD metadata is authoritative in V16.6.
    """
    p = session_reference_policy(now)
    return {
        "status": "VALID",
        "today": p["today"],
        "expected_date": p["official_target_date"] or "",
        "is_weekend": p["phase"] == OFFDAY,
        "next_open_passed": p["phase"] == POSTCLOSE,
        "phase": p["phase"],
        "operator_notice_count": 0,
        "warnings": [],
    }


def daily_freshness(price_date: str, calendar: dict) -> dict:
    if not price_date:
        return {"current_daily": False, "status": "MISSING"}
    # Freshness is primarily resolved by exchange EOD metadata in radar_service.
    # This helper therefore only marks presence and avoids declaring a pre-open
    # previous-session bar stale merely because the calendar date advanced.
    return {"current_daily": True, "status": "FRESH"}


def entry_review_allowed(price_date: str, plan: dict | None, calendar: dict) -> bool:
    return bool(plan is not None)


def save_closure_notice(data_dir: Path, closure_day: str, url: str, at_time: str, now: datetime.datetime | None = None, confirmed: bool = False):
    if not confirmed:
        raise ValueError("必須核對並勾選確認官方公告。")
    notice_file = Path(data_dir) / "closure_notices.json"
    notices = []
    if notice_file.exists():
        try:
            notices = json.loads(notice_file.read_text(encoding="utf-8"))
        except Exception:
            notices = []
    notices.append({"closure_day": closure_day, "url": url, "at_time": at_time})
    notice_file.write_text(json.dumps(notices, ensure_ascii=False, indent=2), encoding="utf-8")
