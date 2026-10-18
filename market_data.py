"""
Taiwan Alpha Radar V16 - market + research data layer with official MOPS fallback.

Price source: Yahoo Finance, with local SQLite cache.
Research source: FinMind v4 (monthly revenue, quarterly financials, PER,
institutional flows). Broker-branch/main-force proxy is optional because the
underlying branch dataset is a Sponsor feature and "main force" has no official
single definition.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import sqlite3
import time
import threading
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
try:
    import yfinance as yf
except ImportError:  # surfaced cleanly at runtime; dependency is in requirements.txt
    yf = None


FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_BRANCH_URL = "https://api.finmindtrade.com/api/v4/taiwan_stock_trading_daily_report"
TWSE_OPENAPI = "https://openapi.twse.com.tw/v1/opendata"
TPEX_OPENAPI = "https://www.tpex.org.tw/openapi/v1"


# Never share a live SQLite connection across Streamlit sessions/threads.
# WAL journal mode is persistent; issuing journal_mode=WAL for EVERY SELECT is
# expensive and can conflict with concurrent readers while changing modes.
_SQLITE_READY: set[str] = set()
_SQLITE_BOOTSTRAP_LOCK = threading.Lock()


@contextmanager
def _sqlite_connect(path: Path):
    """One transaction, one connection, guaranteed close (even on exceptions).

    sqlite3.Connection.__exit__ commits/rolls back but does NOT close the
    connection. This wrapper closes it deterministically to avoid WAL readers
    and filesystem descriptors lingering across Streamlit reruns.
    """
    db_path = Path(path)
    conn = sqlite3.connect(str(db_path), timeout=15.0)
    try:
        conn.execute("PRAGMA busy_timeout=15000")
        with _SQLITE_BOOTSTRAP_LOCK:
            if str(db_path.resolve()) not in _SQLITE_READY:
                # Cache only an actual WAL confirmation; do not label a
                # non-WAL filesystem as WAL-enabled.
                try:
                    mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()
                    if mode and str(mode[0]).lower() == "wal":
                        _SQLITE_READY.add(str(db_path.resolve()))
                except sqlite3.OperationalError as exc:
                    # Another process may be creating the DB at this instant.
                    # Let normal SQL and busy_timeout decide whether we can
                    # proceed; the next call will retry WAL setup.
                    if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                        raise
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    finally:
        conn.close()


def _sqlite_write_retry(path: Path, query: str, params, *, many: bool = False):
    """Retry transient SQLITE_BUSY on bounded writes, never hide failures."""
    for attempt in range(3):
        try:
            with _sqlite_connect(path) as conn:
                if many:
                    conn.executemany(query, params)
                else:
                    conn.execute(query, params)
            return
        except sqlite3.OperationalError as exc:
            if not any(k in str(exc).lower() for k in ("locked", "busy")) or attempt == 2:
                raise
            time.sleep(0.12 * (attempt + 1))


def _taipei_timestamp() -> datetime.datetime:
    tz = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(tz)


def _finmind_token() -> str:
    return os.getenv("FINMIND_TOKEN", "").strip()


def provider_runtime_status(store=None) -> dict:
    return {
        "sqlite_connected": (None if store is None else _sqlite_cache_healthy(store)),
        "yfinance_ready": yf is not None,
        "finmind_token_configured": bool(_finmind_token()),
        "branch_flow_enabled": os.getenv("ENABLE_BRANCH_FLOW", "0") == "1",
        "timestamp": _taipei_timestamp().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _sqlite_cache_healthy(store) -> bool:
    try:
        path = Path(getattr(store, "db_path"))
        with _sqlite_connect(path) as conn:
            return conn.execute("SELECT 1").fetchone()[0] == 1
    except Exception:
        return False


def stock_id_from_ticker(ticker: str) -> str:
    return str(ticker or "").split(".")[0].strip()


def _yf_history_args(period: str) -> dict:
    """Translate UI periods to yfinance-supported arguments.

    yfinance does not support arbitrary period strings such as 3y/8y, so those
    are converted to an explicit start date instead of silently failing.
    """
    valid = {"5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"}
    p = str(period or "5y")
    if p in valid:
        return {"period": p}
    if p.endswith("y") and p[:-1].isdigit():
        years = max(1, int(p[:-1]))
        start = (_taipei_timestamp().date() - datetime.timedelta(days=years * 366 + 10)).isoformat()
        return {"start": start}
    return {"period": "5y"}


INDUSTRY_CODE_LABELS = {
    "01": "水泥工業", "02": "食品工業", "03": "塑膠工業", "04": "紡織纖維",
    "05": "電機機械", "06": "電器電纜", "08": "玻璃陶瓷", "09": "造紙工業",
    "10": "鋼鐵工業", "11": "橡膠工業", "12": "汽車工業", "14": "建材營造",
    "15": "航運業", "16": "觀光餐旅", "17": "金融保險", "18": "貿易百貨",
    "19": "綜合", "20": "其他", "21": "化學工業", "22": "生技醫療業",
    "23": "油電燃氣業", "24": "半導體業", "25": "電腦及週邊設備業", "26": "光電業",
    "27": "通信網路業", "28": "電子零組件業", "29": "電子通路業", "30": "資訊服務業",
    "31": "其他電子業", "32": "文化創意業", "33": "農業科技業", "35": "綠能環保",
    "36": "數位雲端", "37": "運動休閒", "38": "居家生活", "80": "管理股票",
}


def _industry_label(value) -> str:
    raw = str(value or "").strip().replace("　", "")
    if not raw:
        return ""
    if raw in INDUSTRY_CODE_LABELS:
        return INDUSTRY_CODE_LABELS[raw]
    # Some sources occasionally return an integer-like code without zero padding.
    if raw.isdigit() and len(raw) == 1 and f"0{raw}" in INDUSTRY_CODE_LABELS:
        return INDUSTRY_CODE_LABELS[f"0{raw}"]
    # If an upstream provider already supplies a name, retain it.
    return raw


def fetch_twse_universe() -> pd.DataFrame:
    """Fetch a production-safe TWSE + TPEx common-stock universe.

    V16.1 hotfix:
    - TPEx's JSON endpoint can return an HTTP-200 HTML/redirect page on some
      cloud deployments.  A successful HTTP status is therefore *not* treated
      as a successful roster response unless the payload is actually a list.
    - TPEx has four official/fallback paths: primary OpenAPI, alternate OpenAPI,
      official MOPS CSV, and latest TPEx close-quote roster.
    - TWSE and TPEx are validated separately.  A missing whole market is never
      silently accepted just because the combined count is above a low floor.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AlphaRadar/16.6; +https://streamlit.io)",
        "Accept": "application/json,text/csv,text/plain,*/*",
    }

    twse_rows: list[tuple[str, str, str]] = []
    tpex_rows: list[tuple[str, str, str]] = []
    source_notes: list[str] = []

    def _valid_code(value: Any) -> str:
        code = str(value or "").strip()
        if len(code) == 4 and code.isdigit() and not code.startswith("0"):
            return code
        return ""

    def _safe_json(url: str, timeout: int = 12) -> Any:
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            if not r.ok:
                return None
            ctype = str(r.headers.get("content-type", "")).lower()
            # Some TPEx failure pages are HTML with HTTP 200.  Reject them
            # before calling .json() so the next fallback still gets a chance.
            head = (r.text or "")[:200].lstrip().lower()
            if "html" in ctype or head.startswith("<!doctype html") or head.startswith("<html"):
                return None
            js = r.json()
            return js
        except Exception:
            return None

    def _append_json_rows(payload: Any, market: str) -> int:
        if not isinstance(payload, list):
            return 0
        target = twse_rows if market == "TW" else tpex_rows
        before = len(target)
        for item in payload:
            if not isinstance(item, dict):
                continue
            code = _valid_code(
                item.get("公司代號")
                or item.get("SecuritiesCompanyCode")
                or item.get("Code")
                or item.get("公司代碼")
            )
            if not code:
                continue
            name = str(
                item.get("公司簡稱")
                or item.get("CompanyAbbreviation")
                or item.get("公司名稱")
                or item.get("CompanyName")
                or item.get("Company Name")
                or item.get("Name")
                or item.get("SecuritiesCompanyName")
                or ""
            ).strip()
            industry = _industry_label(
                item.get("產業別")
                or item.get("SecuritiesIndustryCode")
                or item.get("IndustryCode")
                or ""
            )
            suffix = ".TW" if market == "TW" else ".TWO"
            target.append((f"{code}{suffix}", name or code, industry or ("上市" if market == "TW" else "上櫃")))
        return len(target) - before

    def _append_csv_rows(url: str, market: str) -> int:
        target = twse_rows if market == "TW" else tpex_rows
        before = len(target)
        try:
            r = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            if not r.ok or not r.content:
                return 0
            # Official MOPS CSV is UTF-8 (often with BOM).  Decode defensively.
            raw = r.content.decode("utf-8-sig", errors="replace")
            if "公司代號" not in raw[:1000]:
                return 0
            from io import StringIO
            df = pd.read_csv(StringIO(raw), dtype=str)
            if df.empty:
                return 0
            for _, row in df.iterrows():
                code = _valid_code(row.get("公司代號"))
                if not code:
                    continue
                name = str(row.get("公司簡稱") or row.get("公司名稱") or code).strip()
                industry = _industry_label(row.get("產業別") or "")
                suffix = ".TW" if market == "TW" else ".TWO"
                target.append((f"{code}{suffix}", name or code, industry or ("上市" if market == "TW" else "上櫃")))
            return len(target) - before
        except Exception:
            return 0

    # ---------- TWSE company master ----------
    payload = _safe_json("https://openapi.twse.com.tw/v1/opendata/t187ap03_L")
    n = _append_json_rows(payload, "TW")
    if n:
        source_notes.append(f"TWSE master {n}")
    else:
        # Official MOPS CSV fallback preserves company industry metadata.
        n = _append_csv_rows("https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv", "TW")
        if n:
            source_notes.append(f"TWSE MOPS CSV {n}")
        else:
            # Last-resort official latest quote roster (industry unavailable).
            q = _safe_json("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL")
            n = _append_json_rows(q, "TW")
            if n:
                source_notes.append(f"TWSE daily roster {n}")

    # ---------- TPEx company master ----------
    # Primary endpoint is known to return a 200 HTML unavailable/redirect page
    # in some cloud environments.  We explicitly test payload type and continue.
    for url, label in [
        ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", "TPEx primary"),
        ("https://www.tpex.org.tw/openapi/v1/mopsfront_t187ap03_O", "TPEx alternate"),
    ]:
        payload = _safe_json(url)
        n = _append_json_rows(payload, "TWO")
        if n:
            source_notes.append(f"{label} {n}")
            break

    if not tpex_rows:
        # Official government open-data CSV.  This is the most important cloud
        # fallback and currently contains the full OTC company master.
        n = _append_csv_rows("https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv", "TWO")
        if n:
            source_notes.append(f"TPEx MOPS CSV {n}")

    if not tpex_rows:
        # Final official roster fallback.  It has no industry field, so the UI
        # will show broad '上櫃' until a company master is available, but the
        # stock is still allowed to participate in the market scan.
        payload = _safe_json("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes")
        n = _append_json_rows(payload, "TWO")
        if n:
            source_notes.append(f"TPEx daily roster {n}")

    # Deduplicate each market before validating counts.
    twse_df = pd.DataFrame(twse_rows, columns=["ticker", "name", "industry"]).drop_duplicates("ticker") if twse_rows else pd.DataFrame(columns=["ticker", "name", "industry"])
    tpex_df = pd.DataFrame(tpex_rows, columns=["ticker", "name", "industry"]).drop_duplicates("ticker") if tpex_rows else pd.DataFrame(columns=["ticker", "name", "industry"])

    twse_n = int(len(twse_df))
    tpex_n = int(len(tpex_df))

    # Conservative production floors.  They are deliberately below normal
    # market counts to tolerate listings/delistings, while still detecting the
    # catastrophic failure mode of losing an entire exchange.
    MIN_TWSE_COMMON = 850
    MIN_TPEX_COMMON = 650
    MIN_TOTAL_COMMON = 1600

    if twse_n < MIN_TWSE_COMMON or tpex_n < MIN_TPEX_COMMON or (twse_n + tpex_n) < MIN_TOTAL_COMMON:
        detail = "; ".join(source_notes) if source_notes else "no successful source"
        raise RuntimeError(
            "股票母體不完整："
            f"上市 {twse_n} 檔、上櫃 {tpex_n} 檔、合計 {twse_n + tpex_n} 檔。"
            "為避免漏掉整個市場後仍產生推薦，本次更新已中止並保留上一份成功快照。"
            f" 資料來源：{detail}"
        )

    out = pd.concat([twse_df, tpex_df], ignore_index=True).drop_duplicates("ticker")
    # Attach diagnostics without changing the public dataframe schema used by
    # radar_service.py.  This is useful for interactive debugging if needed.
    out.attrs["twse_count"] = twse_n
    out.attrs["tpex_count"] = tpex_n
    out.attrs["total_count"] = int(len(out))
    out.attrs["sources"] = source_notes
    return out


class DailyPriceStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_cache_error: str | None = None
        self.download_errors: list[str] = []
        self._init_db()

    def _init_db(self):
        with _sqlite_connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
                    PRIMARY KEY (ticker, date)
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS store_meta (
                    meta_key TEXT PRIMARY KEY, value TEXT
                )"""
            )

    def clear(self) -> bool:
        try:
            with _sqlite_connect(self.db_path) as conn:
                conn.execute("DELETE FROM daily_prices")
                conn.execute("DELETE FROM store_meta")
            return True
        except Exception as exc:
            self.last_cache_error = str(exc)
            return False

    def _meta_get(self, key: str, default: str = "") -> str:
        try:
            with _sqlite_connect(self.db_path) as conn:
                row = conn.execute("SELECT value FROM store_meta WHERE meta_key=?", (key,)).fetchone()
            return str(row[0]) if row else default
        except Exception:
            return default

    def _meta_set(self, key: str, value: str) -> None:
        try:
            with _sqlite_connect(self.db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO store_meta(meta_key,value) VALUES(?,?)", (key, str(value)))
        except Exception:
            pass

    @staticmethod
    def _period_years(period: str) -> int:
        p = str(period or "5y").lower()
        if p == "max":
            return 99
        if p.endswith("y") and p[:-1].isdigit():
            return max(1, int(p[:-1]))
        return 5

    def _coverage_counts(self, tickers: list[str]) -> dict[str, int]:
        if not tickers:
            return {}
        try:
            # Query once without a giant IN (...) placeholder list; some SQLite
            # builds cap bound variables near 999 while the Taiwan universe is
            # >2,000 symbols. Filter the small aggregate result in Python.
            wanted = set(map(str, tickers))
            with _sqlite_connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT ticker, COUNT(*) FROM daily_prices GROUP BY ticker"
                ).fetchall()
            return {str(t): int(n) for t, n in rows if str(t) in wanted}
        except Exception:
            return {}

    @staticmethod
    def _ticker_frame(data: pd.DataFrame, ticker: str, chunk_len: int) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()
        try:
            if isinstance(data.columns, pd.MultiIndex):
                lvl0 = set(map(str, data.columns.get_level_values(0)))
                lvl1 = set(map(str, data.columns.get_level_values(1)))
                if ticker in lvl0:
                    out = data[ticker].copy()
                elif ticker in lvl1:
                    out = data.xs(ticker, axis=1, level=1).copy()
                else:
                    return pd.DataFrame()
            else:
                out = data.copy() if chunk_len == 1 else pd.DataFrame()
            wanted = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
            return out[wanted].dropna(how="all") if wanted else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def batch_fetch_and_update(
        self,
        tickers: list[str],
        period: str = "5y",
        chunk_size: int = 80,
        progress=None,
        force_full: bool = False,
        full_refresh_days: int = 7,
    ) -> dict:
        """Refresh price history efficiently.

        Cold start / weekly maintenance downloads the requested history.  Normal
        daily updates reuse the local multi-year history and only backfill new or
        insufficient tickers; the official TWSE/TPEx overlay then supplies the
        newest complete session.  This preserves the model input while avoiding
        downloading ~2,000 five-year histories on every button click.
        """
        if yf is None:
            raise RuntimeError("缺少 yfinance；請先執行 pip install -r requirements.txt")
        unique = list(dict.fromkeys([t for t in tickers if t]))
        self.download_errors = []
        if not unique:
            return {"downloaded_tickers": 0, "errors": [], "mode": "none", "requested": 0}

        coverage = self._coverage_counts(unique)
        cached_period = self._period_years(self._meta_get("history_period", "0y"))
        requested_period = self._period_years(period)
        last_full_text = self._meta_get("last_full_refresh", "")
        last_full = None
        try:
            last_full = datetime.datetime.fromisoformat(last_full_text).date() if last_full_text else None
        except Exception:
            last_full = None
        age_days = 999 if last_full is None else (_taipei_timestamp().date() - last_full).days
        covered_ratio = sum(1 for t in unique if coverage.get(t, 0) >= 140) / max(1, len(unique))
        full_refresh = bool(
            force_full
            or cached_period < requested_period
            or age_days >= int(full_refresh_days)
            or covered_ratio < 0.70
        )

        if full_refresh:
            fetch_list = unique
            fetch_period = period
            mode = "full"
        else:
            # V18.14 fix: an established five-year cache still needs NEW daily
            # closes.  The old path refreshed only new/short-history tickers;
            # if an exchange endpoint was unreachable, every mature stock could
            # remain stuck on 10/08 indefinitely.  A bounded 1-month overlay is
            # cheap compared with another five-year download, and is followed
            # by the authoritative official EOD overlay in radar_service.
            now_tpe = _taipei_timestamp()
            stage = ("POSTCLOSE" if now_tpe.weekday() < 5 and
                     now_tpe.hour >= 15 else "OTHER")
            daily_key = now_tpe.strftime("%Y-%m-%d") + ":" + stage
            last_key = self._meta_get("last_daily_overlay_key", "")
            if daily_key != last_key:
                fetch_list = unique
                fetch_period = "1mo"
                mode = "recent_daily"
            else:
                fetch_list = [t for t in unique if coverage.get(t, 0) < 140]
                fetch_period = period
                mode = "incremental_cached"

        if progress:
            progress(0, max(1, len(fetch_list)), "完整歷史更新" if full_refresh else "沿用歷史快取")

        downloaded = 0
        total_chunks = max(1, (len(fetch_list) + chunk_size - 1) // chunk_size) if fetch_list else 0
        for i in range(0, len(fetch_list), chunk_size):
            chunk = fetch_list[i:i + chunk_size]
            chunk_no = i // chunk_size + 1
            if progress:
                progress(i, len(fetch_list), f"行情批次 {chunk_no}/{total_chunks}（{min(i+len(chunk),len(fetch_list))}/{len(fetch_list)} 檔）")
            data = None
            exc_last = None
            for attempt in range(2):
                try:
                    history_args = _yf_history_args(fetch_period)
                    data = yf.download(
                        chunk,
                        **history_args,
                        group_by="ticker",
                        auto_adjust=True,
                        actions=False,
                        progress=False,
                        threads=True,
                        timeout=20,
                    )
                    break
                except Exception as exc:
                    exc_last = exc
                    if progress:
                        progress(i, len(fetch_list), f"行情批次 {chunk_no}/{total_chunks} 重試 {attempt+1}/2")
                    time.sleep(0.6 * (attempt + 1))
            if data is None or getattr(data, "empty", True):
                self.download_errors.append(f"chunk {chunk_no}: {exc_last or 'empty response'}")
                continue

            records: list[tuple] = []
            parsed_tickers = 0
            for ticker in chunk:
                frame = self._ticker_frame(data, ticker, len(chunk))
                if frame.empty:
                    continue
                frame = frame.copy()
                frame.index = pd.to_datetime(frame.index).tz_localize(None)
                close = pd.to_numeric(frame.get("Close"), errors="coerce")
                if close is None:
                    continue
                valid = close.notna() & np.isfinite(close) & (close > 0)
                if not bool(valid.any()):
                    continue
                f = frame.loc[valid].copy()
                cl = pd.to_numeric(f["Close"], errors="coerce")
                op = pd.to_numeric(f.get("Open"), errors="coerce").fillna(cl) if "Open" in f.columns else cl
                hi = pd.to_numeric(f.get("High"), errors="coerce").fillna(cl) if "High" in f.columns else cl
                lo = pd.to_numeric(f.get("Low"), errors="coerce").fillna(cl) if "Low" in f.columns else cl
                vol = pd.to_numeric(f.get("Volume"), errors="coerce").fillna(0.0) if "Volume" in f.columns else pd.Series(0.0, index=f.index)
                # Never store contradictory OHLC or negative-volume records.
                # Keep valid bars; let official EOD overlay still be authoritative.
                good = (np.isfinite(op) & (op > 0) & np.isfinite(hi) & (hi > 0)
                        & np.isfinite(lo) & (lo > 0) & np.isfinite(vol) & (vol >= 0)
                        & (hi >= np.maximum(op, cl) * 0.995)
                        & (lo <= np.minimum(op, cl) * 1.005)
                        & (hi >= lo))
                if not bool(good.any()):
                    continue
                cl, op, hi, lo, vol = (series.loc[good] for series in (cl, op, hi, lo, vol))
                dates = cl.index.strftime("%Y-%m-%d").tolist()
                records.extend(zip(
                    [ticker] * len(f), dates,
                    op.astype(float).tolist(), hi.astype(float).tolist(), lo.astype(float).tolist(),
                    cl.astype(float).tolist(), vol.astype(float).tolist(),
                ))
                parsed_tickers += 1

            if records:
                try:
                    _sqlite_write_retry(self.db_path,
                        """INSERT OR REPLACE INTO daily_prices
                            (ticker, date, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)""", records, many=True)
                    downloaded += parsed_tickers
                except Exception as exc:
                    self.download_errors.append(f"sqlite write: {exc}")
            if progress:
                progress(min(i + len(chunk), len(fetch_list)), len(fetch_list), f"行情批次 {chunk_no}/{total_chunks} 完成")

        if full_refresh and downloaded >= max(5, int(len(fetch_list) * 0.70)):
            self._meta_set("history_period", period)
            self._meta_set("last_full_refresh", _taipei_timestamp().isoformat(timespec="seconds"))
        if mode == "recent_daily" and downloaded >= max(1, int(len(fetch_list) * 0.65)):
            # On an ordinary after-close weekday a provider can return thousands
            # of *old* bars successfully. Success must mean broad data for the
            # requested day, not merely HTTP 200 / parsed ticker count.
            freshness_confirmed = True
            if stage == "POSTCLOSE":
                target_day = now_tpe.date().isoformat()
                try:
                    with _sqlite_connect(self.db_path) as conn:
                        available = {str(t) for (t,) in conn.execute(
                            "SELECT ticker FROM daily_prices WHERE date=?", (target_day,)).fetchall()}
                    tw_codes = [t for t in unique if str(t).endswith('.TW')]
                    otc_codes = [t for t in unique if str(t).endswith('.TWO')]
                    freshness_confirmed = bool(
                        '^TWII' in available and
                        (not tw_codes or sum(t in available for t in tw_codes) >= max(1,int(len(tw_codes)*.90))) and
                        (not otc_codes or sum(t in available for t in otc_codes) >= max(1,int(len(otc_codes)*.90))))
                except Exception:
                    freshness_confirmed = False
            if freshness_confirmed:
                self._meta_set("last_daily_overlay_key", daily_key)
                self._meta_set("last_daily_overlay_at", _taipei_timestamp().isoformat(timespec="seconds"))
            else:
                self.download_errors.append("日線近日資料未達今日兩市場覆蓋門檻，未標為更新成功，可稍後補抓。")

        if progress and not fetch_list:
            progress(1, 1, "歷史資料已在快取，直接更新最新交易日")
        return {
            "downloaded_tickers": downloaded,
            "errors": self.download_errors[:],
            "mode": mode,
            "requested": len(fetch_list),
            "covered_ratio": round(covered_ratio, 3),
            "last_full_refresh": last_full_text,
        }

    @staticmethod
    def _num(value):
        try:
            text = str(value).replace(",", "").strip()
            if text in {"", "--", "---", "X", "-"}:
                return None
            v = float(text)
            return v if np.isfinite(v) else None
        except Exception:
            return None

    @staticmethod
    def _roc_date(value):
        """Return ISO date for ROC/Gregorian source dates, never guess invalid days.

        Accepts 1151009, 115/10/9, 115.10.09, 115-10-09,
        20261009, 2026-10-09, and Chinese year/month/day separators.
        """
        if value is None:
            return ""
        raw = str(value).strip().replace("\u3000", "")
        if not raw:
            return ""
        match = re.fullmatch(r"(\d{2,4})[\-/\.\u5e74](\d{1,2})[\-/\.\u6708](\d{1,2})\u65e5?", raw)
        if match:
            year, month, day = map(int, match.groups())
        else:
            digits = raw.replace("/", "").replace("-", "")
            if len(digits) == 7 and digits.isdigit():
                year, month, day = int(digits[:3]), int(digits[3:5]), int(digits[5:7])
            elif len(digits) == 8 and digits.isdigit():
                year, month, day = int(digits[:4]), int(digits[4:6]), int(digits[6:8])
            else:
                return ""
        if year < 1912:
            year += 1911
        # Guard against broken upstream dates instead of labelling them official.
        if not (1990 <= year <= 2100):
            return ""
        try:
            return datetime.date(year, month, day).isoformat()
        except ValueError:
            return ""

    @classmethod
    def _official_payload_date(cls, payload: dict | None) -> str:
        """Use an actual source date, never the date we merely *asked* for.

        An official endpoint can respond with its latest available trading
        session even when queried for a newer date. If we labelled those rows
        with the requested day, the dashboard would falsely claim synchronized
        current-day closing data.
        """
        if not isinstance(payload, dict):
            return ""
        for key in ("date", "Date", "tradeDate", "TradeDate", "reportDate", "ReportDate", "queryDate", "QueryDate", "資料日期", "查詢日期"):
            value = cls._roc_date(payload.get(key))
            if value:
                return value
        for key in ("title", "Title", "stat"):
            label = str(payload.get(key) or "")
            for match in re.finditer(r"\d{2,4}[年/.-]\d{1,2}[月/.-]\d{1,2}日?", label):
                value = cls._roc_date(match.group(0))
                if value:
                    return value
        return ""

    def overlay_official_eod(self, target_date: str | None = None, progress=None) -> dict:
        """Overlay the newest official TWSE/TPEx EOD bars.

        Source order is deliberately redundant because cloud hosts sometimes
        reach Yahoo but fail one of the Taiwan exchange OpenAPI domains:
          1) official latest-snapshot OpenAPI
          2) official date-specific TWSE/TPEx web endpoint for ``target_date``

        The function returns source/date metadata so the UI can show exactly
        what was updated instead of silently presenting a stale Yahoo session.
        """
        records: list[tuple] = []
        counts = {
            "twse": 0, "tpex": 0, "index": 0, "errors": [],
            "target_date": str(target_date or ""),
            "twse_date": "", "tpex_date": "", "index_date": "",
            "twse_source": "", "tpex_source": "", "index_source": "",
            "twse_fresh": False, "tpex_fresh": False, "index_fresh": False,
            "twse_target_response": False, "twse_target_has_data": False,
            "tpex_target_response": False, "tpex_target_has_data": False,
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AlphaRadar/16.6; +https://github.com/)",
            "Accept": "application/json,text/plain,*/*",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

        def get_json(url: str, *, params: dict | None = None, label: str = "", extra_headers: dict | None = None):
            try:
                req_headers = dict(headers)
                if extra_headers:
                    req_headers.update(extra_headers)
                r = requests.get(url, params=params, headers=req_headers, timeout=18)
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                counts["errors"].append(f"{label or url}: {type(exc).__name__}: {exc}")
                return None

        def newest_date(rows: list[tuple], suffix: str) -> str:
            ds = [str(x[1]) for x in rows if str(x[0]).endswith(suffix) and x[1]]
            return max(ds) if ds else ""

        # ---------- TWSE: latest snapshot ----------
        twse_rows: list[tuple] = []
        payload = get_json(
            "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
            label="TWSE OpenAPI latest",
        )
        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                code = str(row.get("Code", "")).strip()
                if len(code) != 4 or not code.isdigit():
                    continue
                date = self._roc_date(row.get("Date"))
                op = self._num(row.get("OpeningPrice")); hi = self._num(row.get("HighestPrice"))
                lo = self._num(row.get("LowestPrice")); cl = self._num(row.get("ClosingPrice"))
                vol = self._num(row.get("TradeVolume"))
                if date and cl is not None and cl > 0:
                    twse_rows.append((f"{code}.TW", date, op or cl, hi or cl, lo or cl, cl, vol or 0.0))
            if twse_rows:
                counts["twse_source"] = "TWSE OpenAPI"

        twse_date = newest_date(twse_rows, ".TW")

        # ---------- TWSE fallback: date-specific RWD ----------
        # Important for Streamlit/GitHub cloud environments where the OpenAPI
        # hostname may be blocked or where Yahoo is one session behind.
        if target_date and (not twse_date or twse_date < target_date):
            ymd = str(target_date).replace("-", "")
            fb = get_json(
                "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY_ALL",
                params={"response": "json", "date": ymd},
                label=f"TWSE RWD {target_date}",
            )
            fb_rows: list[tuple] = []
            if isinstance(fb, dict):
                counts["twse_target_response"] = True
                fb_date = self._official_payload_date(fb)
                if fb_date != str(target_date):
                    counts["errors"].append(
                        f"TWSE RWD date mismatch: requested {target_date}, actual {fb_date or 'missing'}")
                data_rows = fb.get("data") if isinstance(fb.get("data"), list) and fb_date == str(target_date) else []
                # RWD STOCK_DAY_ALL row layout:
                # code,name,volume,value,open,high,low,close,change,transactions
                for row in data_rows:
                    if not isinstance(row, (list, tuple)) or len(row) < 8:
                        continue
                    code = str(row[0]).strip()
                    if len(code) != 4 or not code.isdigit():
                        continue
                    op, hi, lo, cl = map(self._num, [row[4], row[5], row[6], row[7]])
                    vol = self._num(row[2])
                    if cl is not None and cl > 0:
                        fb_rows.append((f"{code}.TW", fb_date, op or cl, hi or cl, lo or cl, cl, vol or 0.0))
            if fb_rows:
                counts["twse_target_has_data"] = True
                twse_rows.extend(fb_rows)
                twse_date = newest_date(fb_rows, ".TW")
                counts["twse_source"] = "TWSE RWD date-specific"

        records.extend(twse_rows)
        counts["twse"] = len({r[0] for r in twse_rows if r[1] == (twse_date or r[1])}) if twse_rows else 0
        counts["twse_date"] = twse_date
        counts["twse_fresh"] = bool(twse_date and (not target_date or twse_date >= target_date))
        if progress:
            progress(1, 3, f"官方行情：上市 {twse_date or '未取得'}")

        # ---------- TPEx: latest snapshot ----------
        tpex_rows: list[tuple] = []
        payload = get_json(
            "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
            label="TPEx OpenAPI latest",
        )
        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                code = str(row.get("SecuritiesCompanyCode") or row.get("Code") or "").strip()
                if len(code) != 4 or not code.isdigit():
                    continue
                date = self._roc_date(row.get("Date"))
                op = self._num(row.get("Open") or row.get("OpeningPrice"))
                hi = self._num(row.get("High") or row.get("HighestPrice"))
                lo = self._num(row.get("Low") or row.get("LowestPrice"))
                cl = self._num(row.get("Close") or row.get("ClosingPrice"))
                vol = self._num(row.get("TradingShares") or row.get("TradeVolume"))
                if date and cl is not None and cl > 0:
                    tpex_rows.append((f"{code}.TWO", date, op or cl, hi or cl, lo or cl, cl, vol or 0.0))
            if tpex_rows:
                counts["tpex_source"] = "TPEx OpenAPI"

        tpex_date = newest_date(tpex_rows, ".TWO")

        # ---------- TPEx fallback: date-specific official dailyQuotes ----------
        if target_date and (not tpex_date or tpex_date < target_date):
            tpex_referer = {"Referer": "https://www.tpex.org.tw/zh-tw/mainboard/trading/info/pricing.html"}
            fb = get_json(
                "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes",
                params={"response": "json", "date": str(target_date).replace("-", "/")},
                label=f"TPEx dailyQuotes {target_date}",
                extra_headers=tpex_referer,
            )
            # Some TPEx deployments expose the same JSON through the older
            # o=json query form. Retry it only when the first variant did not
            # return a table, keeping both paths official.
            if not (isinstance(fb, dict) and isinstance(fb.get("tables"), list) and fb.get("tables")):
                fb2 = get_json(
                    "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes",
                    params={"l": "zh-tw", "s": "0,asc,0", "o": "json", "date": str(target_date).replace("-", "/")},
                    label=f"TPEx dailyQuotes alt {target_date}",
                    extra_headers=tpex_referer,
                )
                if isinstance(fb2, dict):
                    fb = fb2
            fb_rows: list[tuple] = []
            if isinstance(fb, dict):
                counts["tpex_target_response"] = True
                fb_date = self._official_payload_date(fb)
                if fb_date != str(target_date):
                    counts["errors"].append(
                        f"TPEx dailyQuotes date mismatch: requested {target_date}, actual {fb_date or 'missing'}")
                tables = fb.get("tables") if isinstance(fb.get("tables"), list) and fb_date == str(target_date) else []
                for table in tables:
                    if not isinstance(table, dict):
                        continue
                    fields = [str(x).replace(" ", "").strip() for x in (table.get("fields") or [])]
                    data_rows = table.get("data") if isinstance(table.get("data"), list) else []
                    if not data_rows:
                        continue

                    def idx(names: tuple[str, ...], default: int | None = None):
                        for name in names:
                            if name in fields:
                                return fields.index(name)
                        return default

                    # The current official table order is code,name,close,change,
                    # open,high,low,avg,volume,...; field-based lookup is preferred.
                    i_code = idx(("代號", "證券代號"), 0)
                    i_close = idx(("收盤", "收盤價"), 2)
                    i_open = idx(("開盤", "開盤價"), 4)
                    i_high = idx(("最高", "最高價"), 5)
                    i_low = idx(("最低", "最低價"), 6)
                    i_vol = idx(("成交股數", "成交量"), 8)
                    for row in data_rows:
                        if not isinstance(row, (list, tuple)):
                            continue
                        need = max(x for x in [i_code, i_close, i_open, i_high, i_low, i_vol] if x is not None)
                        if len(row) <= need:
                            continue
                        code = str(row[i_code]).strip() if i_code is not None else ""
                        if len(code) != 4 or not code.isdigit():
                            continue
                        cl = self._num(row[i_close]) if i_close is not None else None
                        op = self._num(row[i_open]) if i_open is not None else None
                        hi = self._num(row[i_high]) if i_high is not None else None
                        lo = self._num(row[i_low]) if i_low is not None else None
                        vol = self._num(row[i_vol]) if i_vol is not None else None
                        if cl is not None and cl > 0:
                            fb_rows.append((f"{code}.TWO", fb_date, op or cl, hi or cl, lo or cl, cl, vol or 0.0))
            if fb_rows:
                counts["tpex_target_has_data"] = True
                tpex_rows.extend(fb_rows)
                tpex_date = newest_date(fb_rows, ".TWO")
                counts["tpex_source"] = "TPEx dailyQuotes date-specific"

        # Additional documented TPEx official historical fallback. The newer
        # dailyQuotes API may respond with tables but no source date; such a
        # payload is *not* verifiable and must not be tagged with target_date.
        # The legacy 14:30 quote report contains its own ROC reportDate, so it
        # can safely restore a missing OTC daily close when cloud OpenAPI fails.
        if target_date and (not tpex_date or tpex_date < target_date):
            try:
                date_obj = datetime.date.fromisoformat(str(target_date))
                roc_report_day = f"{date_obj.year - 1911}/{date_obj.month:02d}/{date_obj.day:02d}"
                legacy = get_json(
                    "https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php",
                    params={"l": "zh-tw", "d": roc_report_day, "se": "EW"},
                    label=f"TPEx legacy EOD {target_date}",
                )
                report_day = self._official_payload_date(legacy)
                legacy_rows = []
                if isinstance(legacy, dict) and report_day == str(target_date):
                    for row in (legacy.get("aaData") or []):
                        if not isinstance(row, (tuple, list)) or len(row) < 9:
                            continue
                        code = re.sub(r"<[^>]+>", "", str(row[0])).strip()
                        if len(code) != 4 or not code.isdigit():
                            continue
                        cl, op, hi, lo, vol = (self._num(row[i]) for i in (2, 4, 5, 6, 8))
                        if cl is not None and cl > 0:
                            legacy_rows.append((f"{code}.TWO", report_day, op or cl,
                                                hi or cl, lo or cl, cl, vol or 0.0))
                if len(legacy_rows) >= 50:
                    tpex_rows.extend(legacy_rows)
                    tpex_date = report_day
                    counts["tpex_source"] = "TPEx official historical 14:30"
                elif isinstance(legacy, dict) and legacy.get("aaData") and report_day != target_date:
                    counts["errors"].append("TPEx legacy source date absent/mismatched; rejected")
            except (ValueError, TypeError) as exc:
                counts["errors"].append(f"TPEx legacy date parse: {type(exc).__name__}")

        records.extend(tpex_rows)
        counts["tpex"] = len({r[0] for r in tpex_rows if r[1] == (tpex_date or r[1])}) if tpex_rows else 0
        counts["tpex_date"] = tpex_date
        counts["tpex_fresh"] = bool(tpex_date and (not target_date or tpex_date >= target_date))
        if progress:
            progress(2, 3, f"官方行情：上櫃 {tpex_date or '未取得'}")

        # ---------- TAIEX close: OpenAPI then date-specific RWD fallback ----------
        index_row = None
        payload = get_json(
            "https://openapi.twse.com.tw/v1/exchangeReport/MI_INDEX",
            label="TWSE index OpenAPI latest",
        )
        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("指數") or row.get("Index") or row.get("Name") or "")
                if "發行量加權股價指數" not in name and "TAIEX" not in name.upper():
                    continue
                date = self._roc_date(row.get("日期") or row.get("Date"))
                cl = self._num(row.get("收盤指數") or row.get("ClosingIndex") or row.get("Close"))
                if date and cl is not None and cl > 0:
                    index_row = ("^TWII", date, cl, cl, cl, cl, 0.0)
                    counts["index_source"] = "TWSE index OpenAPI"
                    break

        if target_date and (index_row is None or str(index_row[1]) < target_date):
            ymd = str(target_date).replace("-", "")
            fb = get_json(
                "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
                params={"response": "json", "date": ymd, "type": "ALLBUT0999"},
                label=f"TWSE index RWD {target_date}",
            )
            if isinstance(fb, dict):
                fb_index_date = self._official_payload_date(fb)
                if fb_index_date != str(target_date):
                    counts["errors"].append(
                        f"TWSE index RWD date mismatch: wanted {target_date}, source {fb_index_date or 'unknown'}")
                tables = (fb.get("tables") if isinstance(fb.get("tables"), list) else []) if fb_index_date == str(target_date) else []
                for table in tables:
                    if not isinstance(table, dict):
                        continue
                    fields = [str(x).strip() for x in (table.get("fields") or [])]
                    data_rows = table.get("data") if isinstance(table.get("data"), list) else []
                    if not fields or not data_rows:
                        continue
                    try:
                        name_i = next(i for i, x in enumerate(fields) if "指數" in x and "收盤" not in x)
                        close_i = next(i for i, x in enumerate(fields) if "收盤指數" in x)
                    except StopIteration:
                        continue
                    for row in data_rows:
                        if not isinstance(row, (list, tuple)) or len(row) <= max(name_i, close_i):
                            continue
                        name = str(row[name_i])
                        if "發行量加權股價指數" not in name:
                            continue
                        cl = self._num(row[close_i])
                        if cl is not None and cl > 0:
                            index_row = ("^TWII", fb_index_date, cl, cl, cl, cl, 0.0)
                            counts["index_source"] = "TWSE index RWD date-specific"
                            break
                    if index_row is not None and str(index_row[1]) >= str(target_date):
                        break

        # FMTQIK monthly official market turnover ledger is a documented
        # independent TAIEX-close source. MI_INDEX is sometimes an older
        # response on Streamlit Cloud; never let that stale index hold all
        # TWSE/TPEx securities back when a date-specific official close exists.
        # The returned month MAY contain an older date: only the actual row
        # whose date matches target_date may be published as the target day.
        if target_date and (index_row is None or str(index_row[1]) < str(target_date)):
            ledger = get_json(
                "https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK",
                params={"response": "json", "date": str(target_date).replace("-", "")},
                label=f"TWSE FMTQIK {target_date}",
            )
            if isinstance(ledger, dict) and isinstance(ledger.get('data'), list):
                headers_market = [str(x).strip() for x in (ledger.get('fields') or [])]
                try:
                    d_idx = next(i for i, name in enumerate(headers_market) if name in ('日期', 'Date'))
                    c_idx = next(i for i, name in enumerate(headers_market) if '發行量加權股價指數' in name)
                except StopIteration:
                    d_idx, c_idx = None, None
                if d_idx is not None and c_idx is not None:
                    for values in ledger['data']:
                        if not isinstance(values, (list, tuple)) or len(values) <= max(d_idx, c_idx):
                            continue
                        actual_day = self._roc_date(values[d_idx])
                        close_value = self._num(values[c_idx])
                        if actual_day == str(target_date) and close_value is not None and close_value > 0:
                            # FMTQIK only supplies the daily close, not OHLC.
                            # This is sufficient for historical benchmark
                            # returns but never claims a verified intraday high.
                            index_row = ('^TWII', actual_day, close_value, close_value,
                                         close_value, close_value, 0.0)
                            counts['index_source'] = 'TWSE FMTQIK official EOD ledger'
                            break
                if index_row is None or str(index_row[1]) < str(target_date):
                    counts['errors'].append('FMTQIK 尚無要求日期之實際指數收盤列；不得以當月其他日期代替。')

        if index_row is not None:
            records.append(index_row)
            counts["index"] = 1
            counts["index_date"] = str(index_row[1])
            counts["index_fresh"] = bool(not target_date or str(index_row[1]) >= target_date)
        if progress:
            progress(3, 3, f"官方行情：大盤 {counts.get('index_date') or '未取得'}")

        if records:
            try:
                _sqlite_write_retry(self.db_path,
                    """INSERT OR REPLACE INTO daily_prices
                        (ticker, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    records, many=True)
            except Exception as exc:
                counts["errors"].append(f"official EOD sqlite: {type(exc).__name__}: {exc}")

        counts["all_markets_fresh"] = bool(
            counts["twse_fresh"] and counts["tpex_fresh"] and counts["index_fresh"]
        )
        return counts

    @staticmethod
    def _price_rows_to_frame(rows) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame(rows, columns=["date", "Open", "High", "Low", "Close", "Volume"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()
        return df

    @staticmethod
    def _select_price_frame(conn: sqlite3.Connection, ticker: str,
                            limit: int | None = None,
                            end_date: str | None = None) -> pd.DataFrame:
        """Push date filtering and LIMIT into SQLite; keep 5y bars in order.

        Filtering AFTER LIMIT used to return fewer than requested history bars
        whenever the DB contained newer/partial sessions than the model cutoff.
        """
        args: list = [str(ticker)]
        where = "WHERE ticker = ?"
        if end_date:
            cutoff = pd.Timestamp(str(end_date)).strftime("%Y-%m-%d")
            where += " AND date <= ?"
            args.append(cutoff)
        fields = "date, open as Open, high as High, low as Low, close as Close, volume as Volume"
        if limit is not None and int(limit) > 0:
            sql = (f"SELECT {fields} FROM (SELECT {fields} FROM daily_prices "
                   f"{where} ORDER BY date DESC LIMIT ?) ORDER BY date ASC")
            args.append(int(limit))
        else:
            sql = f"SELECT {fields} FROM daily_prices {where} ORDER BY date ASC"
        df = pd.read_sql_query(sql, conn, params=tuple(args))
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()
        return df

    def iter_prices(self, tickers: list[str], limit: int | None = None, end_date: str | None = None):
        """Stream cached prices with one short-lived reader connection.

        The caller must consume or close the iterator to release its snapshot;
        normal full-universe scans exhaust it, then the connection closes.
        """
        with _sqlite_connect(self.db_path) as conn:
            for ticker in tickers:
                try:
                    yield ticker, self._select_price_frame(conn, ticker, limit, end_date)
                except Exception as exc:
                    self.last_cache_error = f"read {ticker}: {type(exc).__name__}: {exc}"
                    yield ticker, pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    def get_prices(self, ticker: str, limit: int | None = None, end_date: str | None = None) -> pd.DataFrame:
        with _sqlite_connect(self.db_path) as conn:
            return self._select_price_frame(conn, ticker, limit, end_date)


class ResearchDataClient:
    """Cached research-data client for fields shown next to recommendations."""

    def __init__(self, db_path: Path, token: str | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.token = (token if token is not None else _finmind_token()).strip()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 AlphaRadar/16.6"})
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        self._init_cache()

    def _init_cache(self):
        with _sqlite_connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS research_cache (
                    cache_key TEXT PRIMARY KEY,
                    fetched_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )"""
            )

    def _cache_get(self, key: str, ttl_hours: float) -> Any | None:
        try:
            with _sqlite_connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT fetched_at, payload FROM research_cache WHERE cache_key=?", (key,)
                ).fetchone()
            if not row:
                return None
            ts = datetime.datetime.fromisoformat(row[0])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=datetime.timezone.utc)
            age = datetime.datetime.now(datetime.timezone.utc) - ts.astimezone(datetime.timezone.utc)
            if age.total_seconds() > ttl_hours * 3600:
                return None
            return json.loads(row[1])
        except Exception:
            return None

    def _cache_set(self, key: str, payload: Any):
        try:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            _sqlite_write_retry(self.db_path,
                "INSERT OR REPLACE INTO research_cache(cache_key,fetched_at,payload) VALUES(?,?,?)",
                (key, now, text))
        except Exception:
            pass

    def _finmind(self, dataset: str, stock_id: str, start_date: str, end_date: str | None = None, ttl_hours: float = 6.0) -> list[dict]:
        key = f"fm:{dataset}:{stock_id}:{start_date}:{end_date or ''}"
        cached = self._cache_get(key, ttl_hours)
        if isinstance(cached, list):
            return cached
        params = {"dataset": dataset, "data_id": stock_id, "start_date": start_date}
        if end_date:
            params["end_date"] = end_date
        if self.token:
            params["token"] = self.token
        try:
            r = self.session.get(FINMIND_URL, params=params, timeout=12)
            r.raise_for_status()
            js = r.json()
            if isinstance(js, dict) and js.get("status") not in (None, 200, "200"):
                return []
            data = js.get("data", []) if isinstance(js, dict) else []
            if isinstance(data, list):
                self._cache_set(key, data)
                return data
        except Exception:
            pass
        return []

    @staticmethod
    def _num(value):
        try:
            if value is None:
                return None
            s = str(value).strip().replace(",", "").replace("%", "")
            if s in {"", "-", "--", "N/A", "nan", "None"}:
                return None
            x = float(s)
            return x if np.isfinite(x) else None
        except Exception:
            return None

    @staticmethod
    def _roc_month(value) -> str | None:
        s = "".join(ch for ch in str(value or "") if ch.isdigit())
        if len(s) < 5:
            return None
        try:
            roc_year = int(s[:-2])
            month = int(s[-2:])
            if not 1 <= month <= 12:
                return None
            return f"{roc_year + 1911:04d}-{month:02d}"
        except Exception:
            return None

    def _public_json_snapshot(self, cache_key: str, url: str, ttl_hours: float = 8.0) -> list[dict]:
        cached = self._cache_get(cache_key, ttl_hours)
        if isinstance(cached, list):
            return cached
        try:
            r = self.session.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0 AlphaRadar/16.6"})
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                self._cache_set(cache_key, data)
                return data
        except Exception:
            pass
        return []

    @staticmethod
    def _flatten_mops_table(df: pd.DataFrame) -> pd.DataFrame:
        """Flatten MOPS HTML headers so parsers are resilient to multi-row headings."""
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            cols = []
            for col in out.columns:
                parts = []
                for part in col:
                    text = str(part).strip()
                    if text and not text.lower().startswith("unnamed") and text not in parts:
                        parts.append(text)
                cols.append("".join(parts))
            out.columns = cols
        else:
            out.columns = [str(c).strip() for c in out.columns]
        return out

    @staticmethod
    def _header_norm(value: Any) -> str:
        return (
            str(value or "")
            .replace(" ", "")
            .replace("\n", "")
            .replace("\r", "")
            .replace("　", "")
            .replace("（", "(")
            .replace("）", ")")
        )

    @classmethod
    def _find_table_col(cls, df: pd.DataFrame, include: tuple[str, ...], exclude: tuple[str, ...] = ()) -> str | None:
        for col in df.columns:
            text = cls._header_norm(col)
            if all(cls._header_norm(x) in text for x in include) and not any(cls._header_norm(x) in text for x in exclude):
                return col
        return None

    @staticmethod
    def _clean_stock_code(value: Any) -> str:
        text = str(value or "").strip().replace("=", "").replace('"', "")
        if text.endswith(".0"):
            text = text[:-2]
        return text

    @classmethod
    def _parse_monthly_revenue_html(cls, html: str, stock_id: str, greg_year: int, month: int) -> dict | None:
        """Parse one official MOPS historical monthly-revenue HTML file."""
        try:
            tables = pd.read_html(StringIO(str(html)))
        except Exception:
            return None
        for raw in tables:
            df = cls._flatten_mops_table(raw)
            code_col = cls._find_table_col(df, ("公司代號",)) or cls._find_table_col(df, ("公司代碼",))
            if code_col is None:
                continue
            mask = df[code_col].map(cls._clean_stock_code).eq(str(stock_id))
            if not bool(mask.any()):
                continue
            row = df.loc[mask].iloc[0]
            current_col = cls._find_table_col(df, ("當月營收",), ("去年",))
            prev_col = cls._find_table_col(df, ("上月營收",))
            last_year_col = cls._find_table_col(df, ("去年當月營收",))
            mom_col = cls._find_table_col(df, ("上月比較", "增減"))
            yoy_col = cls._find_table_col(df, ("去年同月", "增減"))
            current = cls._num(row.get(current_col)) if current_col else None
            prev = cls._num(row.get(prev_col)) if prev_col else None
            last_year = cls._num(row.get(last_year_col)) if last_year_col else None
            mom = cls._num(row.get(mom_col)) if mom_col else None
            yoy = cls._num(row.get(yoy_col)) if yoy_col else None
            if current is None:
                continue
            if mom is None and prev not in (None, 0):
                mom = (current / prev - 1.0) * 100.0
            if yoy is None and last_year not in (None, 0):
                yoy = (current / last_year - 1.0) * 100.0
            return {
                "month": f"{int(greg_year):04d}-{int(month):02d}",
                "revenue_billion": round(float(current) / 100000.0, 2),
                "mom_pct": round(float(mom), 2) if mom is not None else None,
                "yoy_pct": round(float(yoy), 2) if yoy is not None else None,
            }
        return None

    def _official_monthly_revenue_history_month(self, stock_id: str, ticker: str, greg_year: int, month: int) -> dict | None:
        market = "otc" if str(ticker).endswith(".TWO") else "sii"
        roc_year = int(greg_year) - 1911
        key = f"mops:monthly_hist:{market}:{stock_id}:{roc_year}:{int(month)}"
        cached = self._cache_get(key, 24 * 45)
        if isinstance(cached, dict) and cached.get("month"):
            return cached
        for variant in (0, 1):
            url = f"https://mopsov.twse.com.tw/nas/t21/{market}/t21sc03_{roc_year}_{int(month)}_{variant}.html"
            try:
                r = self.session.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0 AlphaRadar/16.6.2"})
                r.raise_for_status()
                payload = r.content
                text = None
                for encoding in ("big5", "cp950", "utf-8"):
                    try:
                        text = payload.decode(encoding)
                        break
                    except Exception:
                        continue
                if text is None:
                    text = payload.decode("big5", errors="ignore")
                parsed = self._parse_monthly_revenue_html(text, stock_id, greg_year, month)
                if parsed:
                    self._cache_set(key, parsed)
                    return parsed
            except Exception:
                continue
        return None

    def _official_monthly_revenue_history(self, stock_id: str, ticker: str, months: int = 12) -> dict:
        """Display-only official monthly history; latest OpenAPI remains the anchor."""
        latest = self._official_monthly_revenue_latest(stock_id, ticker)
        latest_row = latest.get("latest") or {}
        label = str(latest_row.get("month") or "")
        try:
            anchor = pd.Period(label, freq="M")
        except Exception:
            anchor = pd.Period((_taipei_timestamp().date().replace(day=1) - datetime.timedelta(days=1)), freq="M")
        out = []
        for offset in range(max(1, int(months)) - 1, -1, -1):
            period = anchor - offset
            ym = str(period)
            if ym == label and latest_row.get("revenue_billion") is not None:
                row = dict(latest_row)
            else:
                row = self._official_monthly_revenue_history_month(stock_id, ticker, period.year, period.month)
            if row and row.get("revenue_billion") is not None:
                out.append(row)
        # Deduplicate after merging latest OpenAPI and static history.
        dedup = {str(r.get("month")): r for r in out if r.get("month")}
        rows = [dedup[k] for k in sorted(dedup)][-max(1, int(months)):]
        if not rows:
            return latest
        yoy_vals = [r.get("yoy_pct") for r in rows[-3:] if r.get("yoy_pct") is not None]
        return {
            "available": True,
            "latest": rows[-1],
            "rows": rows,
            "avg_yoy_3m_pct": round(float(np.mean(yoy_vals)), 2) if yoy_vals else None,
            "history_limited": len(rows) < 6,
            "coverage_count": len(rows),
            "coverage_label": f"近 {len(rows)} 個月（官方）" if len(rows) > 1 else "最新 1 個月（官方）",
            "source": "MOPS official historical monthly revenue" if len(rows) > 1 else latest.get("source", "official_openapi"),
        }

    @classmethod
    def _parse_quarterly_income_html(cls, html: str, stock_id: str, greg_year: int, quarter: int) -> dict | None:
        """Parse one MOPS cumulative income-statement period by header name."""
        try:
            tables = pd.read_html(StringIO(str(html)))
        except Exception:
            return None
        for raw in tables:
            df = cls._flatten_mops_table(raw)
            code_col = cls._find_table_col(df, ("公司代號",)) or cls._find_table_col(df, ("公司代碼",))
            if code_col is None:
                continue
            mask = df[code_col].map(cls._clean_stock_code).eq(str(stock_id))
            if not bool(mask.any()):
                continue
            row = df.loc[mask].iloc[0]
            eps_col = cls._find_table_col(df, ("基本每股盈餘",))
            revenue_col = cls._find_table_col(df, ("營業收入",))
            gross_col = cls._find_table_col(df, ("營業毛利",))
            eps = cls._num(row.get(eps_col)) if eps_col else None
            revenue = cls._num(row.get(revenue_col)) if revenue_col else None
            gross = cls._num(row.get(gross_col)) if gross_col else None
            if eps is None and revenue is None and gross is None:
                continue
            return {
                "quarter": f"{int(greg_year)}Q{int(quarter)}",
                "ytd_eps": round(float(eps), 4) if eps is not None else None,
                "ytd_revenue": float(revenue) if revenue is not None else None,
                "ytd_gross": float(gross) if gross is not None else None,
            }
        return None

    def _official_quarter_cumulative(self, stock_id: str, ticker: str, greg_year: int, quarter: int) -> dict | None:
        market = "otc" if str(ticker).endswith(".TWO") else "sii"
        roc_year = int(greg_year) - 1911
        key = f"mops:quarterly_hist:{market}:{stock_id}:{roc_year}:Q{int(quarter)}"
        cached = self._cache_get(key, 24 * 120)
        if isinstance(cached, dict) and cached.get("quarter"):
            return cached
        data = {
            "encodeURIComponent": 1,
            "step": 1,
            "firstin": 1,
            "off": 1,
            "isQuery": "Y",
            "TYPEK": market,
            "year": str(roc_year),
            "season": f"{int(quarter):02d}",
        }
        try:
            r = self.session.post(
                "https://mops.twse.com.tw/mops/web/ajax_t163sb04",
                data=data,
                timeout=12,
                headers={"User-Agent": "Mozilla/5.0 AlphaRadar/16.6.2", "Referer": "https://mops.twse.com.tw/mops/web/t163sb04"},
            )
            r.raise_for_status()
            r.encoding = r.encoding or "utf-8"
            parsed = self._parse_quarterly_income_html(r.text, stock_id, greg_year, quarter)
            if parsed:
                self._cache_set(key, parsed)
                return parsed
        except Exception:
            return None
        return None

    @staticmethod
    def _quarter_index(year: int, quarter: int) -> int:
        return int(year) * 4 + int(quarter) - 1

    @staticmethod
    def _quarter_from_index(index: int) -> tuple[int, int]:
        return int(index // 4), int(index % 4 + 1)

    def _official_financial_history(self, stock_id: str, ticker: str, periods: int = 4) -> dict:
        """Build single-quarter EPS/margin from official cumulative MOPS statements."""
        latest = self._official_financial_latest(stock_id, ticker)
        label = str(latest.get("latest_period") or "")
        try:
            year = int(label[:4]); quarter = int(label[-1])
            if quarter not in (1, 2, 3, 4):
                raise ValueError
        except Exception:
            now = _taipei_timestamp().date()
            # Conservative disclosed-period guess: previous calendar quarter.
            idx = self._quarter_index(now.year, ((now.month - 1) // 3) + 1) - 1
            year, quarter = self._quarter_from_index(idx)
        latest_idx = self._quarter_index(year, quarter)
        # Five cumulative observations are sufficient to reconstruct the last four single quarters.
        keys = [self._quarter_from_index(latest_idx - i) for i in range(max(1, int(periods)), -1, -1)]
        cumulative: dict[tuple[int, int], dict] = {}
        for y, q in keys:
            row = self._official_quarter_cumulative(stock_id, ticker, y, q)
            if row:
                cumulative[(y, q)] = row
        history = []
        target_keys = keys[-max(1, int(periods)):]
        for y, q in target_keys:
            row = cumulative.get((y, q))
            if not row:
                continue
            prev = cumulative.get((y, q - 1)) if q > 1 else None
            def single(field: str):
                cur = row.get(field)
                if cur is None:
                    return None
                if q == 1:
                    return float(cur)
                old = prev.get(field) if prev else None
                return float(cur) - float(old) if old is not None else None
            # Revenue/gross profit are additive flow measures, so a single-quarter
            # margin can be reconstructed by differencing cumulative values. EPS
            # is *not* safely additive/subtractive because the weighted-average
            # share denominator can change (stock dividends/splits, IAS 33).
            # Therefore official MOPS fallback keeps Q2/Q3/Q4 EPS as cumulative
            # YTD and never fabricates a single-quarter EPS by subtraction.
            qrev = single("ytd_revenue")
            qgross = single("ytd_gross")
            gm = qgross / qrev * 100.0 if qgross is not None and qrev not in (None, 0) else None
            ytd_eps = round(float(row.get("ytd_eps")), 2) if row.get("ytd_eps") is not None else None
            single_eps = ytd_eps if q == 1 else None
            history.append({
                "quarter": f"{y}Q{q}",
                "date": f"{y}-{q*3:02d}-01",
                "eps": single_eps,
                "eps_ytd_sum_unadjusted": ytd_eps,
                "eps_ytd_official": ytd_eps,
                "gross_margin_pct": round(gm, 2) if gm is not None and np.isfinite(gm) else None,
            })
        if not history:
            return latest
        history = history[-max(1, int(periods)):]
        gm_vals = [x.get("gross_margin_pct") for x in history if x.get("gross_margin_pct") is not None]
        return {
            "available": True,
            "quarters": history[-4:],
            "history_quarters": history,
            # Official OpenAPI/MOPS fallback is cumulative EPS. Do not manufacture
            # a four-quarter sum from incompatible cumulative denominators.
            "eps_4q_sum": None,
            "gross_margin_4q_avg_pct": round(float(np.mean(gm_vals[-4:])), 2) if gm_vals else None,
            "gross_margin_latest_pct": gm_vals[-1] if gm_vals else latest.get("gross_margin_latest_pct"),
            "latest_ytd_eps": history[-1].get("eps_ytd_sum_unadjusted"),
            "latest_period": history[-1].get("quarter"),
            "eps_ytd_basis": "official_cumulative",
            "history_limited": len(history) < 4,
            "coverage_count": len(history),
            "coverage_label": f"近 {len(history)} 季（官方）",
            "source": "MOPS official historical quarterly income statements",
        }

    @staticmethod
    def _parse_twse_t86_payload(payload: dict, stock_id: str, date_label: str) -> dict | None:
        if not isinstance(payload, dict):
            return None
        fields = [str(x).strip() for x in (payload.get("fields") or [])]
        rows = payload.get("data") or []
        if not fields or not isinstance(rows, list):
            return None
        def n(s):
            return ResearchDataClient._header_norm(s)
        field_map = {n(name): idx for idx, name in enumerate(fields)}
        aliases = {
            "code": ["證券代號"],
            "foreign": ["外陸資買賣超股數(不含外資自營商)", "外資及陸資買賣超股數(不含外資自營商)"],
            "trust": ["投信買賣超股數"],
            "dealer": ["自營商買賣超股數"],
            "total": ["三大法人買賣超股數"],
        }
        idxs = {}
        for key, names in aliases.items():
            idxs[key] = next((field_map.get(n(x)) for x in names if n(x) in field_map), None)
        if idxs.get("code") is None:
            return None
        for raw in rows:
            if not isinstance(raw, (list, tuple)) or idxs["code"] >= len(raw):
                continue
            if ResearchDataClient._clean_stock_code(raw[idxs["code"]]) != str(stock_id):
                continue
            def value(key):
                i = idxs.get(key)
                return ResearchDataClient._num(raw[i]) if i is not None and i < len(raw) else None
            f, t, d, total = value("foreign"), value("trust"), value("dealer"), value("total")
            if total is None and any(x is not None for x in (f, t, d)):
                total = sum(x or 0.0 for x in (f, t, d))
            if not any(x is not None for x in (f, t, d, total)):
                return None
            return {
                "date": str(date_label),
                "foreign_net_lots": round(float(f or 0.0) / 1000.0, 1),
                "trust_net_lots": round(float(t or 0.0) / 1000.0, 1),
                "dealer_net_lots": round(float(d or 0.0) / 1000.0, 1),
                "total_net_lots": round(float(total or 0.0) / 1000.0, 1),
            }
        return None

    @staticmethod
    def _parse_tpex_institutional_rows(rows: list[dict], stock_id: str) -> dict | None:
        if not isinstance(rows, list):
            return None
        row = None
        for candidate in rows:
            if not isinstance(candidate, dict):
                continue
            normalized = {ResearchDataClient._header_norm(k): v for k, v in candidate.items()}
            code = normalized.get(ResearchDataClient._header_norm("SecuritiesCompanyCode"))
            if code is None:
                code = normalized.get(ResearchDataClient._header_norm("股票代號"))
            if ResearchDataClient._clean_stock_code(code) == str(stock_id):
                row = candidate
                break
        if not isinstance(row, dict):
            return None
        norm_row = {ResearchDataClient._header_norm(k): v for k, v in row.items()}
        def first(*keys):
            for key in keys:
                value = norm_row.get(ResearchDataClient._header_norm(key))
                if value is not None and str(value).strip() not in {"", "-", "--"}:
                    return value
            return None
        f = ResearchDataClient._num(first(
            "Foreign Investors include Mainland Area Investors (Foreign Dealers excluded)-Difference",
            "ForeignInvestorsincludeMainlandAreaInvestors(ForeignDealersexcluded)-Difference",
        ))
        t = ResearchDataClient._num(first("SecuritiesInvestmentTrustCompanies-Difference"))
        d = ResearchDataClient._num(first("Dealers-Difference"))
        total = ResearchDataClient._num(first("TotalDifference"))
        if total is None and any(x is not None for x in (f, t, d)):
            total = sum(x or 0.0 for x in (f, t, d))
        date_raw = "".join(ch for ch in str(first("Date", "日期") or "") if ch.isdigit())
        if len(date_raw) == 7:
            date_label = f"{int(date_raw[:3]) + 1911:04d}-{date_raw[3:5]}-{date_raw[5:7]}"
        elif len(date_raw) == 8:
            date_label = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
        else:
            date_label = None  # Unknown source date; never invent the scan date.
        if not any(x is not None for x in (f, t, d, total)):
            return None
        return {
            "date": date_label,
            "foreign_net_lots": round(float(f or 0.0) / 1000.0, 1),
            "trust_net_lots": round(float(t or 0.0) / 1000.0, 1),
            "dealer_net_lots": round(float(d or 0.0) / 1000.0, 1),
            "total_net_lots": round(float(total or 0.0) / 1000.0, 1),
        }

    def _official_institutional_detail(self, stock_id: str, ticker: str, max_sessions: int = 8) -> dict:
        """Official fail-open institutional fallback for display detail only."""
        if str(ticker).endswith(".TWO"):
            rows = self._public_json_snapshot(
                "official:tpex:3insti_daily", f"{TPEX_OPENAPI}/tpex_3insti_daily_trading", ttl_hours=1.0
            )
            parsed = self._parse_tpex_institutional_rows(rows, stock_id)
            if not parsed:
                return {"available": False, "source": "TPEx official institutional OpenAPI", "reason": "NO_DATA"}
            vals = [parsed]
            source = "TPEx official institutional OpenAPI"
        else:
            vals = []
            today = _taipei_timestamp().date()
            checked_weekdays = 0
            offset = 0
            # Bound calls so expanding a card cannot hang the app.  Cache makes repeat clicks cheap.
            while len(vals) < max(1, int(max_sessions)) and checked_weekdays < max(12, int(max_sessions) + 5) and offset < 32:
                day = today - datetime.timedelta(days=offset)
                offset += 1
                if day.weekday() >= 5:
                    continue
                checked_weekdays += 1
                label = day.isoformat()
                key = f"official:t86:{stock_id}:{label}"
                ttl = 0.5 if day == today else 24 * 30
                parsed = self._cache_get(key, ttl)
                if not isinstance(parsed, dict):
                    try:
                        r = self.session.get(
                            "https://www.twse.com.tw/rwd/zh/fund/T86",
                            params={"response": "json", "date": day.strftime("%Y%m%d"), "selectType": "ALLBUT0999"},
                            timeout=10,
                            headers={"User-Agent": "Mozilla/5.0 AlphaRadar/16.6.2"},
                        )
                        r.raise_for_status()
                        parsed = self._parse_twse_t86_payload(r.json(), stock_id, label)
                        if parsed:
                            self._cache_set(key, parsed)
                    except Exception:
                        parsed = None
                if isinstance(parsed, dict) and parsed.get("date"):
                    vals.append(parsed)
            vals = sorted(vals, key=lambda x: x.get("date", ""))[-max(1, int(max_sessions)):]
            source = "TWSE official T86"
        if not vals:
            return {"available": False, "source": source, "reason": "NO_DATA"}
        def agg(field):
            return round(float(sum(float(x.get(field) or 0.0) for x in vals)), 1)
        total = agg("total_net_lots")
        foreign = agg("foreign_net_lots")
        trust = agg("trust_net_lots")
        dealer = agg("dealer_net_lots")
        sessions = len(vals)
        return {
            "available": True,
            "sessions": sessions,
            "foreign_net_lots": foreign,
            "trust_net_lots": trust,
            "dealer_net_lots": dealer,
            "total_net_lots": total,
            "daily_rows": vals,
            "scope_label": f"最近 {sessions} 個交易日（官方）" if sessions > 1 else "最新 1 個交易日（官方）",
            "source": source,
            "summary": f"{source}：{sessions} 個交易日合計 {total:+,.0f} 張（外資 {foreign:+,.0f}、投信 {trust:+,.0f}、自營商 {dealer:+,.0f}）",
        }

    def _official_monthly_revenue_latest(self, stock_id: str, ticker: str) -> dict:
        otc = str(ticker).endswith(".TWO")
        url = (f"{TPEX_OPENAPI}/mopsfin_t187ap05_O" if otc else f"{TWSE_OPENAPI}/t187ap05_L")
        rows = self._public_json_snapshot(f"official:monthly_revenue:{'O' if otc else 'L'}", url, ttl_hours=8)
        row = next((r for r in rows if str(r.get("公司代號") or r.get("公司代碼") or "").strip() == stock_id), None)
        if not row:
            return {"available": False, "rows": [], "source": "official_openapi"}
        current = self._num(row.get("營業收入-當月營收"))
        prev = self._num(row.get("營業收入-上月營收"))
        last_year = self._num(row.get("營業收入-去年當月營收"))
        mom = self._num(row.get("營業收入-上月比較增減(%)"))
        yoy = self._num(row.get("營業收入-去年同月增減(%)"))
        if mom is None and current is not None and prev not in (None, 0):
            mom = (current / prev - 1) * 100
        if yoy is None and current is not None and last_year not in (None, 0):
            yoy = (current / last_year - 1) * 100
        month = self._roc_month(row.get("資料年月"))
        latest = {
            "month": month,
            # Official MOPS amount unit is NTD thousand; 100,000 thousand = 1 億.
            "revenue_billion": round(current / 100000.0, 2) if current is not None else None,
            "mom_pct": round(mom, 2) if mom is not None else None,
            "yoy_pct": round(yoy, 2) if yoy is not None else None,
        }
        return {
            "available": current is not None,
            "latest": latest,
            "rows": [latest] if current is not None else [],
            "avg_yoy_3m_pct": None,
            "history_limited": True,
            "source": "TWSE/TPEx official monthly revenue OpenAPI",
        }

    def _official_financial_latest(self, stock_id: str, ticker: str) -> dict:
        otc = str(ticker).endswith(".TWO")
        suffixes = ["ci", "mim", "basi", "fh", "ins", "bd"]
        row = None
        for suffix in suffixes:
            if otc:
                url = f"{TPEX_OPENAPI}/mopsfin_t187ap06_O_{suffix}"
                key = f"official:income:O:{suffix}"
            else:
                url = f"{TWSE_OPENAPI}/t187ap06_L_{suffix}"
                key = f"official:income:L:{suffix}"
            rows = self._public_json_snapshot(key, url, ttl_hours=24)
            row = next((r for r in rows if str(r.get("公司代號") or r.get("公司代碼") or "").strip() == stock_id), None)
            if row:
                break
        if not row:
            return {"available": False, "quarters": [], "source": "official_openapi"}
        year = self._num(row.get("年度"))
        quarter = self._num(row.get("季別"))
        greg_year = int(year + 1911) if year is not None and year < 1900 else int(year) if year is not None else None
        qn = int(quarter) if quarter is not None else None
        eps = self._num(row.get("基本每股盈餘（元）") or row.get("基本每股盈餘(元)") or row.get("基本每股盈餘"))
        revenue = self._num(row.get("營業收入") or row.get("收入") or row.get("收益"))
        gross = self._num(row.get("營業毛利（毛損）淨額") or row.get("營業毛利（毛損）") or row.get("營業毛利(毛損)淨額") or row.get("營業毛利(毛損)"))
        gm = gross / revenue * 100 if gross is not None and revenue not in (None, 0) else None
        label = f"{greg_year}Q{qn}" if greg_year and qn else None
        return {
            "available": eps is not None or revenue is not None,
            "quarters": [],
            "eps_4q_sum": None,
            "gross_margin_4q_avg_pct": None,
            "gross_margin_latest_pct": round(gm, 2) if gm is not None and np.isfinite(gm) else None,
            "latest_ytd_eps": round(eps, 2) if eps is not None else None,
            "latest_period": label,
            "history_limited": True,
            "source": "TWSE/TPEx official income-statement OpenAPI",
        }


    def _official_valuation_latest(self, stock_id: str, ticker: str) -> dict:
        """Latest official valuation snapshot when FinMind valuation is unavailable.

        TWSE exposes BWIBBU_ALL. TPEx exposes mainboard PER/PBR analysis.  Field
        aliases are intentionally permissive because the English/Chinese labels
        differ slightly between markets and occasionally across endpoint revisions.
        """
        otc = str(ticker).endswith(".TWO")
        if otc:
            url = f"{TPEX_OPENAPI}/tpex_mainboard_peratio_analysis"
            rows = self._public_json_snapshot("official:valuation:O", url, ttl_hours=4)
            code_keys = ["SecuritiesCompanyCode", "SecuritiesCompanyCode ", "股票代號", "證券代號", "公司代號", "Code"]
            pe_keys = ["PriceEarningRatio", "PriceEarningRatio ", "本益比", "PEratio", "P/E"]
            pbr_keys = ["PriceBookRatio", "PriceBookRatio ", "股價淨值比", "PBratio", "P/B"]
            yld_keys = ["DividendYield", "DividendYield ", "殖利率(%)", "殖利率", "DividendYield(%)"]
        else:
            # BWIBBU_ALL is an exchange-report endpoint rather than opendata.
            url = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
            rows = self._public_json_snapshot("official:valuation:L", url, ttl_hours=4)
            code_keys = ["Code", "公司代號", "股票代號"]
            pe_keys = ["PEratio", "PriceEarningRatio", "本益比"]
            pbr_keys = ["PBratio", "PriceBookRatio", "股價淨值比"]
            yld_keys = ["DividendYield", "DividendYield(%)", "殖利率(%)", "殖利率"]

        def first(obj, keys):
            for k in keys:
                if k in obj and str(obj.get(k)).strip() not in {"", "-", "--"}:
                    return obj.get(k)
            return None

        row = next((r for r in rows if str(first(r, code_keys) or "").strip() == stock_id), None)
        if not row:
            return {"available": False, "source": "official_openapi"}
        pe = self._num(first(row, pe_keys))
        pbr = self._num(first(row, pbr_keys))
        yld = self._num(first(row, yld_keys))
        date_val = first(row, ["Date", "日期", "資料日期", "ReportDate"])
        return {
            "available": bool(pe is not None or pbr is not None or yld is not None),
            "date": str(date_val or _taipei_timestamp().date())[:10],
            "pe": round(pe, 2) if pe is not None else None,
            "pbr": round(pbr, 2) if pbr is not None else None,
            "dividend_yield_pct": round(yld, 2) if yld is not None else None,
            "source": "TWSE/TPEx official valuation OpenAPI",
        }

    @staticmethod
    def _date_ago(days: int) -> str:
        return (_taipei_timestamp().date() - datetime.timedelta(days=days)).isoformat()

    def monthly_revenue(self, stock_id: str) -> dict:
        rows = self._finmind("TaiwanStockMonthRevenue", stock_id, self._date_ago(800), ttl_hours=24)
        if not rows:
            return {"available": False, "rows": []}
        df = pd.DataFrame(rows)
        if df.empty or "revenue" not in df.columns:
            return {"available": False, "rows": []}
        for c in ["revenue", "revenue_year", "revenue_month"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["revenue", "revenue_year", "revenue_month"])
        if df.empty:
            return {"available": False, "rows": []}
        df["period"] = pd.to_datetime(
            df["revenue_year"].astype(int).astype(str) + "-" + df["revenue_month"].astype(int).astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
        df = df.dropna(subset=["period"]).sort_values("period").drop_duplicates("period", keep="last")
        df["mom"] = df["revenue"].pct_change()
        period_map = {p.to_period("M"): r for p, r in zip(df["period"], df["revenue"])}
        df["yoy"] = [
            (rev / period_map.get(p.to_period("M") - 12) - 1.0)
            if period_map.get(p.to_period("M") - 12) not in (None, 0) else np.nan
            for p, rev in zip(df["period"], df["revenue"])
        ]
        tail = df.tail(24)
        out_rows = [
            {
                "month": r.period.strftime("%Y-%m"),
                "revenue_billion": round(float(r.revenue) / 1e8, 2),  # 億元 = 1e8 NTD
                "mom_pct": round(float(r.mom) * 100.0, 2) if pd.notna(r.mom) else None,
                "yoy_pct": round(float(r.yoy) * 100.0, 2) if pd.notna(r.yoy) else None,
            }
            for r in tail.itertuples()
        ]
        latest = out_rows[-1] if out_rows else {}
        yoy_vals = [r["yoy_pct"] for r in out_rows[-3:] if r.get("yoy_pct") is not None]
        return {
            "available": bool(out_rows),
            "latest": latest,
            "rows": out_rows,
            "avg_yoy_3m_pct": round(float(np.mean(yoy_vals)), 2) if yoy_vals else None,
        }

    def financials(self, stock_id: str) -> dict:
        rows = self._finmind("TaiwanStockFinancialStatements", stock_id, self._date_ago(650), ttl_hours=96)
        if not rows:
            return {"available": False, "quarters": []}
        df = pd.DataFrame(rows)
        if df.empty or not {"date", "type", "value"}.issubset(df.columns):
            return {"available": False, "quarters": []}
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value", "type"])
        if df.empty:
            return {"available": False, "quarters": []}
        p = df.pivot_table(index="date", columns="type", values="value", aggfunc="last").sort_index()
        if p.empty:
            return {"available": False, "quarters": []}

        rev_col = next((c for c in ["Revenue", "OperatingRevenue", "TotalRevenue", "Income"] if c in p.columns), None)
        gp_col = "GrossProfit" if "GrossProfit" in p.columns else None
        eps_col = "EPS" if "EPS" in p.columns else None

        q = p.tail(8)
        history_quarters = []
        for dt, row in q.iterrows():
            eps = float(row[eps_col]) if eps_col and pd.notna(row.get(eps_col)) else None
            rev = float(row[rev_col]) if rev_col and pd.notna(row.get(rev_col)) else None
            gp = float(row[gp_col]) if gp_col and pd.notna(row.get(gp_col)) else None
            gm = (gp / rev * 100.0) if gp is not None and rev not in (None, 0) else None
            history_quarters.append({
                "quarter": f"{dt.year}Q{((dt.month - 1) // 3) + 1}",
                "date": dt.strftime("%Y-%m-%d"),
                "eps": round(eps, 2) if eps is not None else None,
                "gross_margin_pct": round(gm, 2) if gm is not None and np.isfinite(gm) else None,
            })
        running_by_year = {}
        for item in history_quarters:
            year = int(item["quarter"][:4])
            if item.get("eps") is not None:
                running_by_year[year] = running_by_year.get(year, 0.0) + float(item["eps"])
                item["eps_ytd_sum_unadjusted"] = round(running_by_year[year], 2)
            else:
                item["eps_ytd_sum_unadjusted"] = None
        quarters = history_quarters[-4:]
        eps_vals = [x["eps"] for x in quarters if x.get("eps") is not None]
        gm_vals = [x["gross_margin_pct"] for x in quarters if x.get("gross_margin_pct") is not None]
        return {
            "available": bool(quarters),
            "quarters": quarters,
            "history_quarters": history_quarters,
            "eps_4q_sum": round(float(np.sum(eps_vals)), 2) if eps_vals else None,
            "gross_margin_4q_avg_pct": round(float(np.mean(gm_vals)), 2) if gm_vals else None,
            "gross_margin_latest_pct": gm_vals[-1] if gm_vals else None,
        }

    def valuation(self, stock_id: str) -> dict:
        rows = self._finmind("TaiwanStockPER", stock_id, self._date_ago(50), ttl_hours=6)
        if not rows:
            return {"available": False}
        df = pd.DataFrame(rows)
        if df.empty:
            return {"available": False}
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date")
        row = df.iloc[-1]
        pe = pd.to_numeric(pd.Series([row.get("PER")]), errors="coerce").iloc[0]
        pbr = pd.to_numeric(pd.Series([row.get("PBR")]), errors="coerce").iloc[0]
        yld = pd.to_numeric(pd.Series([row.get("dividend_yield")]), errors="coerce").iloc[0]
        return {
            "available": bool(pd.notna(pe) or pd.notna(pbr)),
            "date": str(row.get("date", ""))[:10],
            "pe": round(float(pe), 2) if pd.notna(pe) else None,
            "pbr": round(float(pbr), 2) if pd.notna(pbr) else None,
            "dividend_yield_pct": round(float(yld), 2) if pd.notna(yld) else None,
        }

    def institutional_flow(self, stock_id: str) -> dict:
        rows = self._finmind("TaiwanStockInstitutionalInvestorsBuySellWide", stock_id, self._date_ago(45), ttl_hours=2)
        if not rows:
            # Compatibility fallback to long table.
            long_rows = self._finmind("TaiwanStockInstitutionalInvestorsBuySell", stock_id, self._date_ago(45), ttl_hours=2)
            return self._institutional_long(long_rows)
        df = pd.DataFrame(rows)
        if df.empty:
            return {"available": False}
        for c in df.columns:
            if c.endswith("_buy") or c.endswith("_sell"):
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").tail(22)

        def net(buy, sell):
            if buy in df.columns and sell in df.columns:
                return float((df[buy] - df[sell]).sum())
            return 0.0

        foreign = net("Foreign_Investor_buy", "Foreign_Investor_sell") + net("Foreign_Dealer_Self_buy", "Foreign_Dealer_Self_sell")
        trust = net("Investment_Trust_buy", "Investment_Trust_sell")
        if "Dealer_buy" in df.columns and "Dealer_sell" in df.columns:
            dealer = net("Dealer_buy", "Dealer_sell")
        else:
            dealer = net("Dealer_self_buy", "Dealer_self_sell") + net("Dealer_Hedging_buy", "Dealer_Hedging_sell")
        total = foreign + trust + dealer
        daily_rows = []
        if "date" in df.columns:
            for _, r in df.iterrows():
                def rnet(buy, sell):
                    return float(r.get(buy, 0) or 0) - float(r.get(sell, 0) or 0) if buy in df.columns and sell in df.columns else 0.0
                f = rnet("Foreign_Investor_buy", "Foreign_Investor_sell") + rnet("Foreign_Dealer_Self_buy", "Foreign_Dealer_Self_sell")
                t = rnet("Investment_Trust_buy", "Investment_Trust_sell")
                if "Dealer_buy" in df.columns and "Dealer_sell" in df.columns:
                    d = rnet("Dealer_buy", "Dealer_sell")
                else:
                    d = rnet("Dealer_self_buy", "Dealer_self_sell") + rnet("Dealer_Hedging_buy", "Dealer_Hedging_sell")
                dt = r.get("date")
                daily_rows.append({
                    "date": str(pd.Timestamp(dt).date()) if pd.notna(dt) else "",
                    "foreign_net_lots": round(f/1000.0,1),
                    "trust_net_lots": round(t/1000.0,1),
                    "dealer_net_lots": round(d/1000.0,1),
                    "total_net_lots": round((f+t+d)/1000.0,1),
                })
        return {
            "available": True,
            "sessions": int(len(df)),
            "foreign_net_lots": round(foreign / 1000.0, 1),
            "trust_net_lots": round(trust / 1000.0, 1),
            "dealer_net_lots": round(dealer / 1000.0, 1),
            "total_net_lots": round(total / 1000.0, 1),
            "daily_rows": daily_rows[-22:],
            "summary": self._flow_text(total / 1000.0, foreign / 1000.0, trust / 1000.0, dealer / 1000.0),
        }

    def _institutional_long(self, rows: list[dict]) -> dict:
        if not rows:
            return {"available": False}
        df = pd.DataFrame(rows)
        if df.empty or not {"buy", "sell", "name"}.issubset(df.columns):
            return {"available": False}
        df["buy"] = pd.to_numeric(df["buy"], errors="coerce").fillna(0.0)
        df["sell"] = pd.to_numeric(df["sell"], errors="coerce").fillna(0.0)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            latest_dates = sorted(df["date"].dropna().unique())[-22:]
            df = df[df["date"].isin(latest_dates)]
        nets = df.assign(net=df["buy"] - df["sell"]).groupby("name")["net"].sum()
        foreign = float(nets[[i for i in nets.index if "Foreign" in str(i) or "外資" in str(i)]].sum())
        trust = float(nets[[i for i in nets.index if "Investment_Trust" in str(i) or "投信" in str(i)]].sum())
        dealer = float(nets[[i for i in nets.index if "Dealer" in str(i) or "自營" in str(i)]].sum())
        total = foreign + trust + dealer
        daily_rows=[]
        if "date" in df.columns:
            for dt, g in df.assign(net=df["buy"]-df["sell"]).groupby("date"):
                gn=g.groupby("name")["net"].sum()
                f=float(gn[[i for i in gn.index if "Foreign" in str(i) or "外資" in str(i)]].sum())
                t=float(gn[[i for i in gn.index if "Investment_Trust" in str(i) or "投信" in str(i)]].sum())
                d=float(gn[[i for i in gn.index if "Dealer" in str(i) or "自營" in str(i)]].sum())
                daily_rows.append({"date":str(pd.Timestamp(dt).date()),"foreign_net_lots":round(f/1000,1),"trust_net_lots":round(t/1000,1),"dealer_net_lots":round(d/1000,1),"total_net_lots":round((f+t+d)/1000,1)})
        return {
            "available": True,
            "sessions": int(df["date"].nunique()) if "date" in df.columns else None,
            "foreign_net_lots": round(foreign / 1000.0, 1),
            "trust_net_lots": round(trust / 1000.0, 1),
            "dealer_net_lots": round(dealer / 1000.0, 1),
            "total_net_lots": round(total / 1000.0, 1),
            "daily_rows": daily_rows[-22:],
            "summary": self._flow_text(total / 1000.0, foreign / 1000.0, trust / 1000.0, dealer / 1000.0),
        }

    @staticmethod
    def _flow_text(total, foreign, trust, dealer) -> str:
        direction = "偏買超" if total > 0 else "偏賣超" if total < 0 else "大致中性"
        return f"近月三大法人{direction} {total:+,.0f} 張（外資 {foreign:+,.0f}、投信 {trust:+,.0f}、自營商 {dealer:+,.0f}）"

    def branch_main_force_proxy(self, stock_id: str, max_sessions: int = 10, force: bool = False) -> dict:
        """
        Optional broker-branch concentration proxy.

        Disabled by default.  It is intentionally labelled as a proxy because
        'main force' is not an official investor category.  FinMind's branch data
        is Sponsor-only and one trading date is requested at a time.
        """
        if (not force) and os.getenv("ENABLE_BRANCH_FLOW", "0") != "1":
            return {
                "available": False,
                "reason": "DISABLED",
                "summary": "未啟用券商分點資料；主力屬代理指標，不以三大法人冒充。",
            }
        if not self.token:
            return {
                "available": False,
                "reason": "TOKEN_REQUIRED",
                "summary": "主力代理需 FinMind Sponsor 分點權限；目前未設定 FINMIND_TOKEN。",
            }

        end = _taipei_timestamp().date()
        days = []
        d = end
        while len(days) < max_sessions and (end - d).days < max(20, max_sessions * 3):
            if d.weekday() < 5:
                days.append(d)
            d -= datetime.timedelta(days=1)

        frames = []
        for day in days:
            key = f"branch:{stock_id}:{day.isoformat()}"
            cached = self._cache_get(key, 4)
            if isinstance(cached, list):
                rows = cached
            else:
                params = {"data_id": stock_id, "start_date": day.isoformat(), "token": self.token}
                try:
                    r = self.session.get(FINMIND_BRANCH_URL, params=params, timeout=10)
                    r.raise_for_status()
                    js = r.json()
                    rows = js.get("data", []) if isinstance(js, dict) else []
                    if isinstance(rows, list):
                        self._cache_set(key, rows)
                except Exception:
                    rows = []
            if rows:
                frames.append(pd.DataFrame(rows))

        if not frames:
            return {
                "available": False,
                "reason": "NO_BRANCH_DATA",
                "summary": "分點資料目前無可用回傳（可能為權限、休市或資料尚未更新）。",
            }
        df = pd.concat(frames, ignore_index=True)
        if not {"securities_trader", "buy", "sell"}.issubset(df.columns):
            return {"available": False, "reason": "SCHEMA_MISMATCH", "summary": "分點資料欄位格式不符。"}
        df["buy"] = pd.to_numeric(df["buy"], errors="coerce").fillna(0.0)
        df["sell"] = pd.to_numeric(df["sell"], errors="coerce").fillna(0.0)
        df["net"] = df["buy"] - df["sell"]
        by_branch = df.groupby("securities_trader")["net"].sum().sort_values()
        top_buy = by_branch.tail(5).sort_values(ascending=False)
        top_sell = by_branch.head(5)
        # Concentration proxy: top five net buys + top five net sells, expressed separately.
        buy_lots = float(top_buy.clip(lower=0).sum()) / 1000.0
        sell_lots = float((-top_sell.clip(upper=0)).sum()) / 1000.0
        proxy = buy_lots - sell_lots
        daily_rows = []
        if "date" in df.columns:
            ddf = df.copy()
            ddf["date"] = pd.to_datetime(ddf["date"], errors="coerce")
            ddf = ddf.dropna(subset=["date"])
            for dt, grp in ddf.groupby("date"):
                daily_rows.append({
                    "date": str(pd.Timestamp(dt).date()),
                    "net_lots": round(float(grp["net"].sum()) / 1000.0, 1),
                })
            daily_rows = sorted(daily_rows, key=lambda x: x["date"])[-max_sessions:]
        return {
            "available": True,
            "sessions": len(frames),
            "proxy_net_lots": round(proxy, 1),
            "top_buy_lots": round(buy_lots, 1),
            "top_sell_lots": round(sell_lots, 1),
            "top_buy_branches": [{"name": str(k), "net_lots": round(float(v) / 1000.0, 1)} for k, v in top_buy.items()],
            "top_sell_branches": [{"name": str(k), "net_lots": round(float(v) / 1000.0, 1)} for k, v in top_sell.items()],
            "daily_rows": daily_rows,
            "summary": f"近 {len(frames)} 個交易日分點代理：前5大淨買與前5大淨賣差額 {proxy:+,.0f} 張（非官方『主力』分類）",
        }

    def holding_distribution(self, stock_id: str, lookback_days: int = 180) -> dict:
        """Optional shareholder concentration snapshot (Backer/Sponsor).

        Large holder is approximated as >=1,000 lots (1,000,000 shares); retail
        is <=400 lots (400,000 shares). This is display-only evidence and is not
        used in ranking, keeping the production model stable if entitlement is
        unavailable.
        """
        if not self.token:
            return {"available": False, "reason": "TOKEN_REQUIRED"}
        rows = self._finmind("TaiwanStockHoldingSharesPer", stock_id, self._date_ago(lookback_days), ttl_hours=24)
        if not rows:
            return {"available": False, "reason": "NO_DATA"}
        df=pd.DataFrame(rows)
        if df.empty or not {"date","HoldingSharesLevel","percent"}.issubset(df.columns):
            return {"available": False, "reason": "SCHEMA_MISMATCH"}
        df["date"]=pd.to_datetime(df["date"],errors="coerce")
        df["percent"]=pd.to_numeric(df["percent"],errors="coerce")
        df=df.dropna(subset=["date","percent"]).copy()
        if df.empty: return {"available":False,"reason":"NO_DATA"}
        import re
        def bounds(label):
            lab=str(label or "").replace(",","")
            if lab.lower()=="total": return (None,None)
            nums=[int(x) for x in re.findall(r"\d+",lab)]
            if not nums: return (None,None)
            if any(k in lab for k in ["以上",">", "+"]): return (nums[0],None)
            if len(nums)>=2: return (min(nums[0],nums[1]),max(nums[0],nums[1]))
            return (nums[0],nums[0])
        hist=[]
        for dt,g in df.groupby("date"):
            large=retail=0.0
            for _,r in g.iterrows():
                lo,hi=bounds(r.get("HoldingSharesLevel")); pct=float(r.get("percent") or 0)
                if lo is None: continue
                if lo>=1_000_000: large += pct
                if hi is not None and hi<=400_000: retail += pct
            hist.append({"date":str(pd.Timestamp(dt).date()),"large_holder_pct":round(large,2),"retail_pct":round(retail,2)})
        hist=sorted(hist,key=lambda x:x["date"])[-16:]
        if not hist: return {"available":False,"reason":"NO_DATA"}
        return {"available":True,"latest_date":hist[-1]["date"],"large_holder_pct":hist[-1]["large_holder_pct"],"retail_pct":hist[-1]["retail_pct"],"history":hist}

    def official_breadth_snapshot(self, tickers: list[str], asof_date: str | None = None,
                                 *, decision_at: str | None = None, historical_replay: bool = False) -> dict[str, dict]:
        """Return one comparable, low-cost official research snapshot for a stock universe.

        V17 uses this *breadth* dataset for ranking so every liquid stock is
        compared on the same evidence basis.  Rich FinMind/MOPS history remains
        available for display/on-demand detail, but it no longer changes the
        already-published cross-sectional ranking merely because one ticker had
        a richer provider response than another.

        The method intentionally uses market-wide official endpoints and indexes
        them once, avoiding thousands of per-stock HTTP requests.
        """
        tickers = [str(t) for t in dict.fromkeys(tickers or []) if str(t)]
        wanted = {stock_id_from_ticker(t): t for t in tickers if stock_id_from_ticker(t)}
        if not wanted:
            return {}
        from pit_data_manager import load_breadth_replay
        # Historical decisions may ONLY read versions actually observed before
        # that time. The current-day OpenAPI endpoints cannot be rewound.
        if historical_replay:
            if not decision_at:
                raise ValueError('historical PIT replay needs explicit decision_at')
            return load_breadth_replay(tickers, self.db_path.parent / 'pit_observations.sqlite', decision_at)

        # Never allow a live latest-data API call to masquerade as evidence
        # captured on an old decision date. Historical times MUST use archive
        # replay; asof_date is an EOD price reference, not a first-seen time.
        if decision_at is not None:
            from pit_data_manager import _dt, TZ
            from datetime import datetime as _clock_datetime
            requested = _dt(decision_at)
            actual_seen = _clock_datetime.now(TZ)
            if requested is None or abs((actual_seen-requested).total_seconds())>300:
                raise ValueError('live breadth does not accept a backdated/future decision_at; use historical_replay=True')

        out: dict[str, dict] = {
            t: {
                "stock_id": stock_id_from_ticker(t),
                "monthly_revenue": {"available": False, "rows": [], "source": "official breadth"},
                "financials": {"available": False, "quarters": [], "source": "official breadth"},
                "valuation": {"available": False, "source": "official breadth"},
                "institutional_flow": {"available": False, "source": "official breadth"},
                "main_force_proxy": {"available": False, "reason": "NOT_IN_BREADTH_MODEL"},
                "source": "TWSE/TPEx official breadth snapshot",
                "fetched_at": _taipei_timestamp().isoformat(timespec="seconds"),
                "breadth_comparable": True,
            }
            for t in tickers
        }

        def code_of(row: dict) -> str:
            for key in ("公司代號", "公司代碼", "Code", "SecuritiesCompanyCode", "股票代號", "證券代號"):
                if key in row:
                    c = self._clean_stock_code(row.get(key))
                    if c:
                        return c
            # fallback through normalized headers
            norm = {self._header_norm(k): v for k, v in (row or {}).items()}
            for key in ("公司代號", "公司代碼", "Code", "SecuritiesCompanyCode", "股票代號", "證券代號"):
                v = norm.get(self._header_norm(key))
                c = self._clean_stock_code(v)
                if c:
                    return c
            return ""

        # ---- monthly revenue: one bulk endpoint per market ----
        for otc, url, key in [
            (False, f"{TWSE_OPENAPI}/t187ap05_L", "official:monthly_revenue:L"),
            (True, f"{TPEX_OPENAPI}/mopsfin_t187ap05_O", "official:monthly_revenue:O"),
        ]:
            rows = self._public_json_snapshot(key, url, ttl_hours=8)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                code = code_of(row)
                ticker = wanted.get(code)
                if not ticker or bool(ticker.endswith(".TWO")) != otc:
                    continue
                current = self._num(row.get("營業收入-當月營收"))
                prev = self._num(row.get("營業收入-上月營收"))
                last_year = self._num(row.get("營業收入-去年當月營收"))
                mom = self._num(row.get("營業收入-上月比較增減(%)"))
                yoy = self._num(row.get("營業收入-去年同月增減(%)"))
                if mom is None and current is not None and prev not in (None, 0):
                    mom = (current / prev - 1.0) * 100.0
                if yoy is None and current is not None and last_year not in (None, 0):
                    yoy = (current / last_year - 1.0) * 100.0
                month = self._roc_month(row.get("資料年月"))
                latest = {
                    "month": month,
                    "revenue_billion": round(current / 100000.0, 2) if current is not None else None,
                    "mom_pct": round(mom, 2) if mom is not None else None,
                    "yoy_pct": round(yoy, 2) if yoy is not None else None,
                }
                out[ticker]["monthly_revenue"] = {
                    "available": current is not None,
                    "latest": latest,
                    "rows": [latest] if current is not None else [],
                    "avg_yoy_3m_pct": None,
                    "history_limited": True,
                    "source": "TWSE/TPEx official monthly revenue OpenAPI",
                }

        # ---- latest financial statement: six industry schemas, one pass each ----
        for otc in (False, True):
            for suffix in ("ci", "mim", "basi", "fh", "ins", "bd"):
                if otc:
                    url = f"{TPEX_OPENAPI}/mopsfin_t187ap06_O_{suffix}"
                    key = f"official:income:O:{suffix}"
                else:
                    url = f"{TWSE_OPENAPI}/t187ap06_L_{suffix}"
                    key = f"official:income:L:{suffix}"
                rows = self._public_json_snapshot(key, url, ttl_hours=24)
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    code = code_of(row)
                    ticker = wanted.get(code)
                    if not ticker or bool(ticker.endswith(".TWO")) != otc:
                        continue
                    # The same code should appear in one schema.  Keep the first
                    # usable record so later endpoints cannot overwrite it with
                    # an empty variant.
                    if (out[ticker].get("financials") or {}).get("available"):
                        continue
                    year = self._num(row.get("年度")); quarter = self._num(row.get("季別"))
                    greg_year = int(year + 1911) if year is not None and year < 1900 else int(year) if year is not None else None
                    qn = int(quarter) if quarter is not None else None
                    eps = self._num(row.get("基本每股盈餘（元）") or row.get("基本每股盈餘(元)") or row.get("基本每股盈餘"))
                    revenue = self._num(row.get("營業收入") or row.get("收入") or row.get("收益"))
                    gross = self._num(row.get("營業毛利（毛損）淨額") or row.get("營業毛利（毛損）") or row.get("營業毛利(毛損)淨額") or row.get("營業毛利(毛損)"))
                    gm = gross / revenue * 100.0 if gross is not None and revenue not in (None, 0) else None
                    label = f"{greg_year}Q{qn}" if greg_year and qn else None
                    out[ticker]["financials"] = {
                        "available": eps is not None or revenue is not None,
                        "quarters": [],
                        "eps_4q_sum": None,
                        "gross_margin_4q_avg_pct": None,
                        "gross_margin_latest_pct": round(gm, 2) if gm is not None and np.isfinite(gm) else None,
                        "latest_ytd_eps": round(eps, 2) if eps is not None else None,
                        "latest_period": label,
                        "eps_ytd_basis": "official_cumulative",
                        "history_limited": True,
                        "source": "TWSE/TPEx official income-statement OpenAPI",
                    }

        # ---- official valuation: one endpoint per market ----
        market_specs = [
            (False, "official:valuation:L", "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL",
             ["Code", "公司代號", "股票代號"], ["PEratio", "PriceEarningRatio", "本益比"],
             ["PBratio", "PriceBookRatio", "股價淨值比"], ["DividendYield", "DividendYield(%)", "殖利率(%)", "殖利率"]),
            (True, "official:valuation:O", f"{TPEX_OPENAPI}/tpex_mainboard_peratio_analysis",
             ["SecuritiesCompanyCode", "SecuritiesCompanyCode ", "股票代號", "證券代號", "公司代號", "Code"],
             ["PriceEarningRatio", "PriceEarningRatio ", "本益比", "PEratio", "P/E"],
             ["PriceBookRatio", "PriceBookRatio ", "股價淨值比", "PBratio", "P/B"],
             ["DividendYield", "DividendYield ", "殖利率(%)", "殖利率", "DividendYield(%)"]),
        ]
        def first(obj, keys):
            for k in keys:
                if k in obj and str(obj.get(k)).strip() not in {"", "-", "--"}:
                    return obj.get(k)
            norm = {self._header_norm(k): v for k, v in obj.items()}
            for k in keys:
                v = norm.get(self._header_norm(k))
                if v is not None and str(v).strip() not in {"", "-", "--"}:
                    return v
            return None
        for otc, key, url, code_keys, pe_keys, pbr_keys, yld_keys in market_specs:
            rows = self._public_json_snapshot(key, url, ttl_hours=4)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                code = self._clean_stock_code(first(row, code_keys))
                ticker = wanted.get(code)
                if not ticker or bool(ticker.endswith(".TWO")) != otc:
                    continue
                pe = self._num(first(row, pe_keys)); pbr = self._num(first(row, pbr_keys)); yld = self._num(first(row, yld_keys))
                out[ticker]["valuation"] = {
                    "available": bool(pe is not None or pbr is not None or yld is not None),
                    "date": str(first(row, ["Date", "日期", "資料日期", "ReportDate"]) or "")[:10] or None,
                    "pe": round(pe, 2) if pe is not None else None,
                    "pbr": round(pbr, 2) if pbr is not None else None,
                    "dividend_yield_pct": round(yld, 2) if yld is not None else None,
                    "source": "TWSE/TPEx official valuation OpenAPI",
                }

        # ---- institutional flow: TPEx latest bulk + TWSE latest completed T86 ----
        tpex_rows = self._public_json_snapshot(
            "official:tpex:3insti_daily", f"{TPEX_OPENAPI}/tpex_3insti_daily_trading", ttl_hours=1.0
        )
        for row in tpex_rows:
            if not isinstance(row, dict):
                continue
            code = code_of(row); ticker = wanted.get(code)
            if not ticker or not ticker.endswith(".TWO"):
                continue
            parsed = self._parse_tpex_institutional_rows([row], code)
            if parsed:
                out[ticker]["institutional_flow"] = {
                    "available": True, "sessions": 1,
                    "foreign_net_lots": parsed.get("foreign_net_lots"),
                    "trust_net_lots": parsed.get("trust_net_lots"),
                    "dealer_net_lots": parsed.get("dealer_net_lots"),
                    "total_net_lots": parsed.get("total_net_lots"),
                    "daily_rows": [parsed], "date": parsed.get("date"), "scope_label": "最新 1 個交易日（官方）",
                    "source": "TPEx official institutional OpenAPI",
                }

        try:
            base_day = datetime.date.fromisoformat(str(asof_date)) if asof_date else _taipei_timestamp().date()
        except Exception:
            base_day = _taipei_timestamp().date()
        payload = None; label = ""
        # At most seven weekday probes; successful payload is cached as a whole,
        # so this is one market-wide lookup rather than one request per ticker.
        probed = 0; offset = 0
        probe_limit = 1 if asof_date else 5
        while probed < probe_limit and offset < 10 and payload is None:
            day = base_day - datetime.timedelta(days=offset); offset += 1
            if day.weekday() >= 5:
                continue
            probed += 1
            label = day.isoformat()
            cache_key = f"official:t86:bulk:{label}"
            cached = self._cache_get(cache_key, 0.5 if day == _taipei_timestamp().date() else 24 * 30)
            if isinstance(cached, dict) and (cached.get("data") or []):
                payload = cached
                break
            try:
                r = self.session.get(
                    "https://www.twse.com.tw/rwd/zh/fund/T86",
                    params={"response": "json", "date": day.strftime("%Y%m%d"), "selectType": "ALLBUT0999"},
                    timeout=6,
                    headers={"User-Agent": "Mozilla/5.0 AlphaRadar/17.0"},
                )
                r.raise_for_status(); js = r.json()
                if isinstance(js, dict) and isinstance(js.get("data"), list) and js.get("data"):
                    payload = js; self._cache_set(cache_key, js); break
            except Exception:
                pass
        if isinstance(payload, dict):
            fields = [str(x).strip() for x in (payload.get("fields") or [])]
            rows = payload.get("data") or []
            norm = lambda x: self._header_norm(x)
            fmap = {norm(name): idx for idx, name in enumerate(fields)}
            def fidx(names):
                return next((fmap.get(norm(x)) for x in names if norm(x) in fmap), None)
            idxs = {
                "code": fidx(["證券代號"]),
                "foreign": fidx(["外陸資買賣超股數(不含外資自營商)", "外資及陸資買賣超股數(不含外資自營商)"]),
                "trust": fidx(["投信買賣超股數"]),
                "dealer": fidx(["自營商買賣超股數"]),
                "total": fidx(["三大法人買賣超股數"]),
            }
            for raw in rows:
                if not isinstance(raw, (list, tuple)) or idxs["code"] is None or idxs["code"] >= len(raw):
                    continue
                code = self._clean_stock_code(raw[idxs["code"]]); ticker = wanted.get(code)
                if not ticker or ticker.endswith(".TWO"):
                    continue
                def val(name):
                    i = idxs.get(name); return self._num(raw[i]) if i is not None and i < len(raw) else None
                f = val("foreign"); t = val("trust"); d = val("dealer"); total = val("total")
                if total is None and any(x is not None for x in (f, t, d)):
                    total = sum(x or 0.0 for x in (f, t, d))
                if not any(x is not None for x in (f, t, d, total)):
                    continue
                parsed = {
                    "date": label,
                    "foreign_net_lots": round(float(f or 0.0) / 1000.0, 1),
                    "trust_net_lots": round(float(t or 0.0) / 1000.0, 1),
                    "dealer_net_lots": round(float(d or 0.0) / 1000.0, 1),
                    "total_net_lots": round(float(total or 0.0) / 1000.0, 1),
                }
                out[ticker]["institutional_flow"] = {
                    "available": True, "sessions": 1,
                    "foreign_net_lots": parsed["foreign_net_lots"],
                    "trust_net_lots": parsed["trust_net_lots"],
                    "dealer_net_lots": parsed["dealer_net_lots"],
                    "total_net_lots": parsed["total_net_lots"],
                    "daily_rows": [parsed], "date": parsed.get("date"), "scope_label": "最新 1 個交易日（官方）",
                    "source": "TWSE official T86",
                }
        from pit_data_manager import annotate_breadth_live
        return annotate_breadth_live(
            out, self.db_path.parent / 'pit_observations.sqlite',
            decision_at=decision_at,
        )

    def stock_research_detail(self, ticker: str, include_branch: bool = False) -> dict:
        """Richer on-demand display data without changing production ranking inputs.

        FinMind remains preferred when it supplies a useful history.  When it does
        not, this path backfills official MOPS history and official institutional
        data.  It is called only from the UI detail toggle.
        """
        base = self.stock_research(ticker, include_branch=include_branch)
        stock_id = stock_id_from_ticker(ticker)
        rev = dict(base.get("monthly_revenue") or {})
        fin = dict(base.get("financials") or {})
        inst = dict(base.get("institutional_flow") or {})
        warnings = []
        try:
            if len(rev.get("rows") or []) < 6 or rev.get("history_limited"):
                upgraded = self._official_monthly_revenue_history(stock_id, ticker, months=12)
                if len(upgraded.get("rows") or []) > len(rev.get("rows") or []):
                    rev = upgraded
        except Exception as exc:
            warnings.append(f"monthly_history:{type(exc).__name__}")
        try:
            if len(fin.get("history_quarters") or fin.get("quarters") or []) < 4 or fin.get("history_limited"):
                upgraded = self._official_financial_history(stock_id, ticker, periods=4)
                if len(upgraded.get("history_quarters") or upgraded.get("quarters") or []) > len(fin.get("history_quarters") or fin.get("quarters") or []):
                    fin = upgraded
        except Exception as exc:
            warnings.append(f"quarterly_history:{type(exc).__name__}")
        try:
            if not inst.get("available") or not (inst.get("daily_rows") or []):
                upgraded = self._official_institutional_detail(stock_id, ticker, max_sessions=8)
                if upgraded.get("available"):
                    inst = upgraded
        except Exception as exc:
            warnings.append(f"institutional:{type(exc).__name__}")
        result = dict(base)
        result["monthly_revenue"] = rev
        result["financials"] = fin
        result["institutional_flow"] = inst
        sources = []
        for obj in [rev, fin, result.get("valuation") or {}, inst]:
            if obj.get("source") and str(obj.get("source")) not in sources:
                sources.append(str(obj.get("source")))
        result["source"] = " + ".join(sources) or str(base.get("source") or "research unavailable")
        result["fetched_at"] = _taipei_timestamp().isoformat(timespec="seconds")
        if warnings:
            result["detail_warnings"] = warnings
        return result

    def stock_research(self, ticker: str, include_branch: bool = False) -> dict:
        stock_id = stock_id_from_ticker(ticker)
        rev = self.monthly_revenue(stock_id)
        fin = self.financials(stock_id)
        val = self.valuation(stock_id)
        inst = self.institutional_flow(stock_id)

        # FinMind is preferred for history, but production display must not go
        # blank just because a token/rate-limit/network call fails. Official
        # TWSE/TPEx snapshots provide the latest revenue and latest cumulative EPS.
        if not rev.get("available"):
            rev = self._official_monthly_revenue_latest(stock_id, ticker)
        if not fin.get("available"):
            fin = self._official_financial_latest(stock_id, ticker)
        if not val.get("available"):
            val = self._official_valuation_latest(stock_id, ticker)

        branch = self.branch_main_force_proxy(stock_id) if include_branch else {
            "available": False,
            "reason": "NOT_REQUESTED",
            "summary": "券商分點尚未載入，可於個股展開區按鈕載入。",
        }
        sources = []
        for obj in [rev, fin, val]:
            if obj.get("source"):
                sources.append(str(obj.get("source")))
        if (val.get("available") and not val.get("source")) or inst.get("available"):
            sources.append("FinMind v4")
        return {
            "stock_id": stock_id,
            "monthly_revenue": rev,
            "financials": fin,
            "valuation": val,
            "institutional_flow": inst,
            "main_force_proxy": branch,
            "source": " + ".join(dict.fromkeys(sources)) or "research unavailable",
            "fetched_at": _taipei_timestamp().isoformat(timespec="seconds"),
        }
