"""Point-in-time evidence contract and immutable, observed-at snapshots.

Never backdate an observation to a statutory deadline.  A backtest may read
only a version actually archived by its decision timestamp.  Legal deadlines
are informative lower/upper bounds, not evidence of an actual publication.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import sqlite3
from typing import Any

TZ = ZoneInfo('Asia/Taipei')


def _dt(value: Any) -> datetime | None:
    if value is None or str(value).strip() == '':
        return None
    try:
        if isinstance(value, datetime):
            out = value
        elif isinstance(value, date):
            out = datetime.combine(value, time(23, 59, 59))
        else:
            raw = str(value).strip()
            if len(raw) == 10:
                out = datetime.combine(date.fromisoformat(raw), time(23, 59, 59))
            else:
                out = datetime.fromisoformat(raw.replace('Z', '+00:00'))
        return out.replace(tzinfo=TZ) if out.tzinfo is None else out.astimezone(TZ)
    except (ValueError, TypeError):
        return None


def asof_check(*, decision_at: Any, period_end: Any = None,
               published_at: Any = None, first_seen_at: Any = None,
               require_observation: bool = True) -> dict:
    """Fail closed: known future dates are LOOKAHEAD; unverifiable data UNKNOWN.

    Date-only published/seen values are conservatively interpreted as 23:59:59
    Taipei time, so a same-day pre-close decision never gets an un-timed item.
    """
    decision = _dt(decision_at)
    period = _dt(period_end)
    published = _dt(published_at)
    seen = _dt(first_seen_at)
    if decision is None:
        return {'valid': False, 'status': 'INVALID', 'reason': '缺少有效決策時間'}
    for label, instant in (('資料期末', period), ('公告時間', published), ('首次取得時間', seen)):
        if instant is not None and instant > decision:
            return {'valid': False, 'status': 'LOOKAHEAD', 'reason': f'{label}晚於決策時間'}
    if published_at is not None and published is None:
        return {'valid': False, 'status': 'INVALID', 'reason': '公告時間無法解析'}
    if first_seen_at is not None and seen is None:
        return {'valid': False, 'status': 'INVALID', 'reason': '首次取得時間無法解析'}
    if period_end is not None and period is None:
        return {'valid': False, 'status': 'INVALID', 'reason': '資料期末無法解析'}
    if require_observation and seen is None:
        return {'valid': False, 'status': 'UNVERIFIED_TIME', 'reason': '無當時可取得的觀測紀錄'}
    if not published and not seen:
        return {'valid': False, 'status': 'UNVERIFIED_TIME', 'reason': '既無公告時間亦無實際觀測時間'}
    if published is not None and seen is not None and seen < published:
        return {'valid': False, 'status': 'INVALID', 'reason': '取得時間早於正式公告時間，需人工核實'}
    return {'valid': True, 'status': 'VERIFIED' if published else 'OBSERVED_ONLY',
            'reason': '公開時間與觀測時間核對' if published else '已於決策前觀測；原始公告時間未證實'}


def check_pit_validity(data_period_end: str, published_at: str | None,
                       first_seen_at: str | None, decision_at: str,
                       data_type: str = 'monthly_revenue') -> tuple[bool, str]:
    """Gemini-compatible API with fail-closed semantics (no invented deadlines)."""
    result = asof_check(decision_at=decision_at, period_end=data_period_end,
                        published_at=published_at, first_seen_at=first_seen_at)
    return bool(result['valid']), str(result['status'] + ': ' + result['reason'])


def observable_period_end(data_type: str, raw: str | None) -> str | None:
    """Normalize only known report periods, never invent a publication date."""
    if not raw:
        return None
    try:
        if data_type == 'monthly_revenue' and len(str(raw)) == 7:
            import calendar
            year, month = (int(x) for x in raw.split('-'))
            return date(year, month, calendar.monthrange(year, month)[1]).isoformat()
        if data_type == 'financials' and 'Q' in str(raw):
            year, quarter = str(raw).split('Q')
            q = int(quarter)
            return date(int(year), q * 3, [31,30,30,31][q - 1]).isoformat() if q in (1,2,3,4) else None
        return date.fromisoformat(str(raw)[:10]).isoformat()
    except (ValueError, TypeError):
        return None


class PITLedger:
    """Append-only local evidence versions; does not provide historical data retroactively.

    A duplicate payload gets its ORIGINAL observed timestamp back, not the new
    scan time.  Different provider revisions create distinct rows. Read-only
    replays must not call present-day bulk OpenAPI fallbacks.
    """
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS pit_observations (
                id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, data_type TEXT NOT NULL,
                period_end TEXT NOT NULL, source TEXT NOT NULL, payload_hash TEXT NOT NULL,
                observed_at TEXT NOT NULL, published_at TEXT, payload_json TEXT NOT NULL,
                UNIQUE(ticker, data_type, period_end, source, payload_hash)
            )''')
            db.execute('''CREATE INDEX IF NOT EXISTS pit_asof ON pit_observations
                (ticker, data_type, observed_at)''')

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=20)
        db.execute('PRAGMA busy_timeout=20000')
        return db

    @staticmethod
    def _key(payload: dict) -> tuple[str, str]:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)
        return hashlib.sha256(raw.encode('utf-8')).hexdigest(), raw

    def record_many(self, rows: list[dict], observed_at: Any = None) -> dict:
        observed = _dt(observed_at or datetime.now(TZ))
        if observed is None:
            raise ValueError('invalid observation time')
        stamp = observed.isoformat(timespec='seconds')
        insert_rows = []
        for row in rows:
            payload = row.get('payload')
            if not isinstance(payload, dict):
                continue
            period = observable_period_end(str(row.get('data_type')), row.get('period_end'))
            if not period or period > observed.date().isoformat():
                continue
            key, raw = self._key(payload)
            pub = _dt(row.get('published_at'))
            # Never record future publications as observed evidence.
            if pub is not None and pub > observed:
                continue
            insert_rows.append((str(row['ticker']), str(row['data_type']), period,
                                str(row.get('source') or 'unknown'), key, stamp,
                                pub.isoformat(timespec='seconds') if pub else None, raw))
        if not insert_rows:
            return {'seen': 0, 'new': 0}
        with self._connect() as db:
            before = db.total_changes
            db.executemany('''INSERT OR IGNORE INTO pit_observations
                (ticker,data_type,period_end,source,payload_hash,observed_at,published_at,payload_json)
                VALUES (?,?,?,?,?,?,?,?)''', insert_rows)
            new = db.total_changes - before
        return {'seen': len(insert_rows), 'new': new}

    def latest_asof(self, ticker: str, data_type: str, decision_at: Any) -> dict | None:
        cutoff = _dt(decision_at)
        if cutoff is None:
            return None
        with self._connect() as db:
            row = db.execute('''SELECT period_end,source,observed_at,published_at,payload_json
                FROM pit_observations WHERE ticker=? AND data_type=? AND observed_at <= ?
                AND (published_at IS NULL OR published_at <= ?)
                ORDER BY period_end DESC,observed_at DESC,id DESC LIMIT 1''',
                (ticker, data_type, cutoff.isoformat(timespec='seconds'),
                 cutoff.isoformat(timespec='seconds'))).fetchone()
        if row is None:
            return None
        period, source, first_seen, published, raw = row
        check = asof_check(decision_at=cutoff, period_end=period,
                           published_at=published, first_seen_at=first_seen)
        if not check['valid']:
            return None
        return {'period_end': period, 'source': source, 'first_seen_at': first_seen,
                'published_at': published, 'payload': json.loads(raw), 'pit': check}


# Heuristic conservative staleness, not a substitution for official deadlines.
# Monthly/quarterly fields are valid between filings; short-lived flows are not.
FRESHNESS_LIMIT_DAYS = {'monthly_revenue': 50, 'financials': 160,
                        'institutional_flow': 8, 'valuation': 8}


def is_timely(kind: str, period_end: str, decision_at: Any) -> bool:
    period = _dt(period_end)
    decision = _dt(decision_at)
    if not period or not decision:
        return False
    days = (decision.date() - period.date()).days
    return 0 <= days <= FRESHNESS_LIMIT_DAYS.get(kind, 0)


BREADTH_TYPES = ('monthly_revenue', 'financials', 'institutional_flow', 'valuation')


def period_for_section(section: dict, kind: str) -> str | None:
    """Explicit source report period; never substitute model asof/scan date."""
    if kind == 'monthly_revenue':
        return observable_period_end(kind, (section.get('latest') or {}).get('month'))
    if kind == 'financials':
        return observable_period_end(kind, section.get('latest_period'))
    if kind == 'institutional_flow':
        sessions = section.get('daily_rows') or []
        return observable_period_end(kind, sessions[-1].get('date')) if sessions else None
    if kind == 'valuation':
        return observable_period_end(kind, section.get('date'))
    return None


def annotate_breadth_live(snapshot: dict[str, dict], ledger_path: str | Path,
                          decision_at: Any = None) -> dict[str, dict]:
    """Archive whole-market evidence in one transaction and tag each field.

    Evidence first seen *today* cannot be used in a backtest of yesterday. The
    base daily price date stays separate from actual research decision time.
    """
    observed = _dt(decision_at or datetime.now(TZ))
    if observed is None:
        raise ValueError('decision_at invalid')
    ledger = PITLedger(ledger_path)
    rows = []
    for ticker, research in snapshot.items():
        if not isinstance(research, dict):
            continue
        for kind in BREADTH_TYPES:
            item = research.get(kind) or {}
            if not isinstance(item, dict) or not item.get('available'):
                continue
            period = period_for_section(item, kind)
            published_at = item.get('published_at')
            if period and period <= observed.date().isoformat():
                payload = {k: v for k, v in item.items() if k not in {'pit', 'first_seen_at', 'fetched_at'}}
                rows.append({'ticker': ticker, 'data_type': kind,
                             'period_end': period, 'published_at': published_at,
                             'source': item.get('source') or 'TWSE/TPEx', 'payload': payload})
    ledger.record_many(rows, observed_at=observed)
    for ticker, research in snapshot.items():
        if not isinstance(research, dict):
            continue
        research['pit_decision_at'] = observed.isoformat(timespec='seconds')
        research['pit_mode'] = 'live_observation'
        for kind in BREADTH_TYPES:
            item = research.get(kind) or {}
            if not isinstance(item, dict):
                continue
            period = period_for_section(item, kind)
            if not item.get('available'):
                item['pit'] = {'valid': False, 'status': 'MISSING', 'reason': '來源未提供有效資料'}
                continue
            if not period:
                item['pit'] = {'valid': False, 'status': 'UNVERIFIED_TIME',
                               'reason': '資料來源缺少可證明的資料所屬日期'}
                item['available'] = False
                continue
            version = ledger.latest_asof(ticker, kind, observed)
            if version is None or version['period_end'] != period:
                item['pit'] = {'valid': False, 'status': 'LOOKAHEAD' if period > observed.date().isoformat() else 'UNVERIFIED_TIME',
                               'reason': '資料期末晚於決策時間或未能寫入PIT版本庫'}
                item['available'] = False
                continue
            if not is_timely(kind, period, observed):
                item['pit'] = {'valid': False, 'status': 'STALE',
                               'reason': '超過該類型的建議資料時效，待下一筆新證據'}
                item['available'] = False
                continue
            item['pit'] = version['pit']
            item['pit'].update({'period_end': period, 'first_seen_at': version['first_seen_at'],
                                'published_at': version['published_at'],
                                'version_source': version['source']})
    return snapshot


def load_breadth_replay(tickers: list[str], ledger_path: str | Path,
                        decision_at: Any) -> dict[str, dict]:
    """Historical/replay mode is strictly offline: missing archive => missing factor.

    This API must not call a live 'latest' OpenAPI endpoint while replaying.
    """
    ledger = PITLedger(ledger_path)
    out = {}
    for ticker in tickers:
        one = {'stock_id': str(ticker).split('.')[0], 'breadth_comparable': True,
               'pit_mode': 'historical_replay', 'pit_decision_at': str(decision_at),
               'source': 'archived PIT observations'}
        for kind in BREADTH_TYPES:
            row = ledger.latest_asof(ticker, kind, decision_at)
            if row is None:
                one[kind] = {'available': False, 'pit': {'valid': False, 'status': 'NO_PIT_ARCHIVE',
                             'reason': '決策時間之前沒有實際保存的資料版本'}, 'source': 'PIT archive'}
            else:
                content = row['payload']
                if not is_timely(kind, row['period_end'], decision_at):
                    content['available'] = False
                    content['pit'] = {'valid': False, 'status': 'STALE',
                                      'reason': '當時已超出該類證據可用時效'}
                else:
                    content['pit'] = dict(row['pit'], period_end=row['period_end'],
                                          first_seen_at=row['first_seen_at'],
                                          published_at=row['published_at'], version_source=row['source'])
                one[kind] = content
        one['main_force_proxy'] = {'available': False}
        out[ticker] = one
    return out
