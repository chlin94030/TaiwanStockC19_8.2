"""V18.0 server-side scan manager.

A full market scan must never live inside a Streamlit widget run.  This module
keeps one process-local worker per data directory and persists its progress to
disk, so mobile WebSocket disconnect/reconnect does not restart the scan.  The
worker survives *browser* disconnects, not an app-server restart or eviction.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import threading
import time
from typing import Callable

from operational_tools import ScanBusyError, safe_error_text, scan_lock
from trading_calendar import POSTCLOSE, session_reference_policy
from v174_integrity import assess_snapshot

_TZ = dt.timezone(dt.timedelta(hours=8))
_GUARD = threading.RLock()
_JOBS: dict[str, threading.Thread] = {}
_STATUS_NAME = "market_scan_status.json"


def _now() -> dt.datetime:
    return dt.datetime.now(_TZ)


def _atomic_json(path: Path, payload: dict) -> None:
    """Atomic publication prevents readers from seeing truncated JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import uuid
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _json_file(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _pid_alive(pid: object) -> bool:
    try:
        value = int(pid)
        if value < 1:
            return False
        os.kill(value, 0)
        return True
    except PermissionError:
        return True
    except (ValueError, TypeError, ProcessLookupError, OSError):
        return False


def status(data_dir: Path) -> dict:
    """Read the persisted job report across sessions; never trigger a scan."""
    data_dir = Path(data_dir)
    p = data_dir / _STATUS_NAME
    obj = _json_file(p)
    if not obj:
        return {"state": "idle", "stage": "尚未開始更新", "percent": 0}
    if obj.get("state") == "running":
        key = str(data_dir.resolve())
        with _GUARD:
            thread = _JOBS.get(key)
            this_process = int(obj.get("pid") or -1) == os.getpid()
            thread_ok = bool(thread and thread.is_alive())
        # If a previous Streamlit server was restarted, its worker cannot
        # resume.  Never claim that the old progress is still active.
        if this_process and not thread_ok:
            obj["state"] = "interrupted"
            obj["stage"] = "伺服器工作執行緒已中斷，上一份完成的快照仍可使用"
        elif not this_process and not _pid_alive(obj.get("pid")):
            obj["state"] = "interrupted"
            obj["stage"] = "伺服器重啟後工作中斷，可重新啟動更新"
    return obj


def _recent_success(snapshot: dict | None, now: dt.datetime, cooldown_seconds: int) -> bool:
    """Only avoid a repeated refresh for a demonstrably sound *recent* model.

    This is not exchange freshness verification. POSTCLOSE without official EOD
    is never short-circuited; if a new Taiwan session starts, force a scan.
    """
    if not isinstance(snapshot, dict) or assess_snapshot(snapshot).get("status") != "OK":
        return False
    # Force a one-time provenance refresh for old V18.4 snapshots.  This does
    # not change the frozen alpha engine or discard the multi-year price cache.
    from strategy_config import ARCHITECTURE_VERSION
    if (snapshot.get("model_version") == ARCHITECTURE_VERSION
        and snapshot.get("data_integrity_schema") != "v18.5-source-provenance"):
        return False
    try:
        generated = dt.datetime.fromisoformat(str(snapshot["generated_at"]))
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=_TZ)
        age = (now - generated.astimezone(_TZ)).total_seconds()
    except (KeyError, ValueError, TypeError):
        return False
    if not (0 <= age <= max(30, int(cooldown_seconds))):
        return False
    phase = session_reference_policy(now)["phase"]
    meta = snapshot.get("session") or {}
    if phase == POSTCLOSE and (
        not meta.get("official_today_complete") or str(snapshot.get("price_date")) != now.date().isoformat()
    ):
        return False
    if generated.date() != now.date() and phase != "OFFDAY":
        return False
    return True


def start_scan(
    data_dir: Path,
    settings,
    run: Callable,
    *,
    snapshot: dict | None = None,
    force: bool = False,
    cooldown_seconds: int = 1200,
    now: dt.datetime | None = None,
    expected_model_version: str | None = None,
) -> dict:
    """Start one server worker and return immediately (no Streamlit APIs)."""
    now = now or _now()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    key = str(data_dir.resolve())
    with _GUARD:
        running = status(data_dir)
        if running.get("state") == "running":
            return {"state": "running", "started": False, "message": "更新正在進行，可離開本頁後再回來查看"}
        # A V17 snapshot may still be fresh on the same calendar day, but
        # must not prevent the first V18 calculation after deployment.
        snapshot_version_ok = (
            not expected_model_version or
            (isinstance(snapshot, dict) and snapshot.get("model_version") == expected_model_version)
        )
        if not force and snapshot_version_ok and _recent_success(snapshot, now, cooldown_seconds):
            return {"state": "reused", "started": False, "message": "近期已完成相同市場階段的模型更新；直接使用快照，不必重算"}

        run_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        state = {
            "state": "running", "run_id": run_id, "stage": "準備上市、上櫃官方資料",
            "percent": 0.0, "started_at": now.isoformat(timespec="seconds"),
            "updated_at": now.isoformat(timespec="seconds"), "pid": os.getpid(),
        }
        _atomic_json(data_dir / _STATUS_NAME, state)

        def worker():
            worker_start = time.monotonic()
            last_write = 0.0
            last_pct = -1.0

            def update(stage: str, value: float):
                nonlocal last_write, last_pct
                pct = min(100.0, max(0.0, float(value) * 100.0))
                mono = time.monotonic()
                if pct < 100 and mono - last_write < 0.6 and abs(pct - last_pct) < 2.0:
                    return
                last_write, last_pct = mono, pct
                state.update({"stage": str(stage)[:120], "percent": round(pct, 1), "updated_at": _now().isoformat(timespec="seconds")})
                _atomic_json(data_dir / _STATUS_NAME, state)

            try:
                with scan_lock(data_dir / "market_scan.lock"):
                    completed = run(data_dir, settings, progress=update, now=_now())
                persisted = _json_file(data_dir / "dashboard_snapshot.json")
                if str(persisted.get("snapshot_id") or "") != str(completed.get("snapshot_id") or ""):
                    raise RuntimeError("模型運算已完成，但快照儲存識別碼不一致；保留上一份可用資料，請檢查儲存權限")
                # Preserve what this completed model actually knew, without
                # changing the ranking or blocking a successful scan if the
                # optional prospective archive is not writable.
                try:
                    from prediction_journal_v188 import freeze_completed_ranking
                    ranking_archive = freeze_completed_ranking(data_dir, persisted)
                except Exception:
                    ranking_archive = {'status':'UNAVAILABLE','reason':'研究日誌無法寫入，日線模型仍可使用'}
                finished = _now()
                completed_at = finished.isoformat(timespec="seconds")
                audit = assess_snapshot(completed)
                model_date = str(completed.get('price_date') or '')
                postclose_lag = bool(finished.weekday() < 5 and finished.hour >= 17
                                     and model_date < finished.date().isoformat())
                if audit.get('sync_state') == 'PROVISIONAL':
                    freshness = 'provisional'
                    freshness_message = ('已取得較新兩市日線，但其中來源為快取備援；'
                                         '保留歷史預估，暫停即時可買判斷。')
                elif postclose_lag:
                    freshness = 'lagging'
                    freshness_message = ('本輪掃描完成，但資料仍未更新到今天；'
                                         '可能是來源回傳舊日、Yahoo更新失敗或雲端讀取受限。')
                else:
                    freshness = 'verified' if audit.get('sync_state') == 'ALIGNED' else 'unverified'
                    freshness_message = '交易所來源尚未全部核對，請檢查資料品質。' if freshness == 'unverified' else ''
                source_meta = ((completed.get('coverage') or {}).get('official_eod') or {})
                state.update({
                    'ranking_archive': ranking_archive,
                    'state': 'complete', 'stage': ('掃描完成，資料日期仍需確認' if freshness != 'verified' else '完整模型更新完成'),
                    'percent': 100.0,
                    'finished_at': completed_at, 'updated_at': completed_at,
                    'elapsed_seconds': round(time.monotonic() - worker_start, 1),
                    'snapshot_id': str(completed.get('snapshot_id') or ''),
                    'price_date': model_date,
                    'model_count': int(completed.get('full_universe_model_n') or 0),
                    'integrity': audit.get('status'),
                    'freshness_state': freshness, 'freshness_message': freshness_message,
                    'official_eod': {
                        'raw_twse_date': source_meta.get('raw_twse_date') or source_meta.get('twse_date') or '',
                        'raw_tpex_date': source_meta.get('raw_tpex_date') or source_meta.get('tpex_date') or '',
                        'source_errors_n': len(source_meta.get('errors') or []),
                    },
                })
            except ScanBusyError:
                state.update({"state": "busy", "stage": "另一個更新工作已在執行；可使用上一版快照", "updated_at": _now().isoformat(timespec="seconds")})
            except Exception as exc:
                secret = os.getenv("FINMIND_TOKEN", "")
                state.update({
                    "state": "failed", "stage": "本次更新未完成；仍保留上一版成功快照",
                    "error": safe_error_text(exc, mask_tokens=(secret,)),
                    "updated_at": _now().isoformat(timespec="seconds"),
                    "elapsed_seconds": round(time.monotonic() - worker_start, 1),
                })
            finally:
                _atomic_json(data_dir / _STATUS_NAME, state)
                with _GUARD:
                    _JOBS.pop(key, None)

        thread = threading.Thread(target=worker, name="alpha-market-" + run_id, daemon=True)
        _JOBS[key] = thread
        thread.start()
        return {"state": "running", "started": True, "message": "已在伺服器啟動更新；螢幕鎖定不會要求重新開始"}
