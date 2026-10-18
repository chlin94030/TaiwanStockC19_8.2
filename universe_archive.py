"""Historically observed investable universe versions; no retrospective survivor fill.

Only membership actually saved by a past decision timestamp is eligible for a
PIT test. A present TWSE/TPEx listing is NEVER copied back to older years.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import hashlib
import json
import sqlite3
from pit_data_manager import TZ, _dt


class UniverseArchive:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=20) as db:
            db.execute('''CREATE TABLE IF NOT EXISTS universe_snapshots (
              hash TEXT PRIMARY KEY, observed_at TEXT NOT NULL,
              source TEXT NOT NULL, payload_json TEXT NOT NULL
            )''')
            db.execute('''CREATE TABLE IF NOT EXISTS universe_observations (
              observed_at TEXT PRIMARY KEY, hash TEXT NOT NULL,
              FOREIGN KEY(hash) REFERENCES universe_snapshots(hash)
            )''')

    def record(self, members: list[dict], observed_at=None) -> dict:
        seen = _dt(observed_at or datetime.now(TZ))
        if seen is None:
            raise ValueError('invalid observation date')
        rows = sorted(({'ticker':str(x.get('ticker','')),
                        'name':str(x.get('name') or ''),
                        'industry':str(x.get('industry') or '')}
                      for x in members if x.get('ticker')), key=lambda d:d['ticker'])
        if not rows:
            raise ValueError('empty market universe is not a valid snapshot')
        raw = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()
        stamp = seen.isoformat(timespec='seconds')
        with sqlite3.connect(self.path, timeout=20) as db:
            db.execute('INSERT OR IGNORE INTO universe_snapshots VALUES (?,?,?,?)',
                       (digest, stamp, 'observed TWSE/TPEx current listings', raw))
            db.execute('INSERT OR IGNORE INTO universe_observations VALUES (?,?)',
                       (stamp, digest))
        return {'observed_at':stamp, 'n':len(rows), 'hash':digest[:16],
                'limitations':'此時實際觀測到的上市母體；不能還原首次觀測日前下市股票'}

    def replay(self, decision_at) -> dict:
        decision = _dt(decision_at)
        if decision is None:
            return {'valid':False, 'reason':'missing or invalid decision_at', 'members':[]}
        with sqlite3.connect(self.path, timeout=20) as db:
            row = db.execute('''SELECT o.observed_at,s.payload_json FROM universe_observations o
                JOIN universe_snapshots s ON o.hash=s.hash WHERE o.observed_at<=?
                ORDER BY o.observed_at DESC LIMIT 1''',
                (decision.isoformat(timespec='seconds'),)).fetchone()
        if row is None:
            return {'valid':False, 'reason':'無決策時間前的真實歷史上市母體，禁止使用今天的倖存者名單', 'members':[]}
        return {'valid':True, 'observed_at':row[0],
                'members':json.loads(row[1]),
                'warning':'須另納入歷史下市、停牌、換股、公司行為與最終清算價值，才可宣稱完成無倖存者偏差回測'}
