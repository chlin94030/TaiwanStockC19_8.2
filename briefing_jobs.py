"""Non-blocking, cross-browser-session market briefing cache for Alpha Radar."""
from __future__ import annotations

from pathlib import Path
import datetime as dt
import json
import os
import threading
import time

from global_context import fetch_global_context
from market_briefing import _only_recent_news, build_waiting_brief, human_global_summary
from scan_job_manager import _atomic_json, _json_file

_TZ = dt.timezone(dt.timedelta(hours=8))
_LOCK = threading.RLock()
_THREADS: dict[str, threading.Thread] = {}


def _now() -> dt.datetime:
    return dt.datetime.now(_TZ)


def _age_seconds(timestamp: object, now: dt.datetime) -> float:
    try:
        t = dt.datetime.fromisoformat(str(timestamp or ""))
        if t.tzinfo is None:
            t = t.replace(tzinfo=_TZ)
        return (now - t.astimezone(_TZ)).total_seconds()
    except (TypeError, ValueError):
        return float("inf")


def load_brief(data_dir: Path) -> dict:
    """Instant disk read. Do not fetch Yahoo or news here."""
    data_dir = Path(data_dir)
    obj = _json_file(data_dir / "market_briefing_snapshot.json")
    if not obj:
        g = _json_file(data_dir / "global_context.json")
        n = _json_file(data_dir / "market_news_digest.json")
        obj = {"global": g, "news": n, "built_at": g.get("fetched_at") or n.get("fetched_at") or ""}
    obj = dict(obj)
    news = obj.get("news") or {}
    if news:
        obj["news"] = _only_recent_news(news, now=_now(), max_age_hours=72)
    g = dict(obj.get("global") or {})
    age = _age_seconds(g.get("fetched_at"), _now())
    if g and age > 45 * 60:
        g["stale"] = True
    obj["global"] = g
    obj["global_summary"] = human_global_summary(g)
    with _LOCK:
        thread = _THREADS.get(str(data_dir.resolve()))
        obj["refreshing"] = bool(thread and thread.is_alive())
    obj["age_minutes"] = round(_age_seconds(obj.get("built_at"), _now()) / 60, 1) if obj.get("built_at") else None
    return obj


def ensure_brief(data_dir: Path, *, force: bool = False, ttl_seconds: int = 1500) -> dict:
    """Kick off a background refresh only when necessary, without blocking UI."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    key = str(data_dir.resolve())
    with _LOCK:
        thread = _THREADS.get(key)
        if thread and thread.is_alive():
            return {"started": False, "state": "running"}
        record = _json_file(data_dir / "market_briefing_snapshot.json")
        age = _age_seconds(record.get("built_at"), _now())
        if not force and 0 <= age < max(120, int(ttl_seconds)):
            return {"started": False, "state": "cached"}

        def worker():
            try:
                data = build_waiting_brief(data_dir, lambda: fetch_global_context(data_dir, max_age_minutes=20, force=force), force_news=force)
                data["built_at"] = _now().isoformat(timespec="seconds")
                # A failure to fetch network data must not erase a previously
                # useful market tape or news list. But expired news is filtered
                # on each read, including when fallbacks are stale.
                prior = _json_file(data_dir / "market_briefing_snapshot.json")
                if not (data.get("global") or {}).get("available") and (prior.get("global") or {}).get("available"):
                    old = dict(prior["global"])
                    old["stale"] = True
                    data["global"] = old
                if not (data.get("news") or {}).get("available") and (prior.get("news") or {}).get("available"):
                    data["news"] = dict(prior["news"])
                    data["news"]["stale"] = True
                _atomic_json(data_dir / "market_briefing_snapshot.json", data)
            except Exception:
                # Best-effort briefing must never block or destroy core model.
                pass
            finally:
                with _LOCK:
                    _THREADS.pop(key, None)

        thread = threading.Thread(target=worker, name="alpha-briefing-refresh", daemon=True)
        _THREADS[key] = thread
        thread.start()
        return {"started": True, "state": "running"}
