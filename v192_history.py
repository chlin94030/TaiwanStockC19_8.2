"""Portable historical database snapshots. No tokens, code, or live orders.

A run downloads a price interval per symbol, not once for every test cutoff.
The exported SQLite can be inspected offline with any SQLite client. Its CSV
companions are the safe replay interchange; no untrusted SQL script is loaded.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import os
import sqlite3
import threading
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
from v19_config import VERSION
from v19_data import atomic_json, atomic_csv


def _plain(df):
    f = df.copy()
    for col in f:
        if pd.api.types.is_datetime64_any_dtype(f[col]):
            f[col] = f[col].astype(str)
    return f


def export_database(root, prices, evidence, manifest, cfg):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    database = root/'history.sqlite'
    temp = root/f'history.{threading.get_ident()}.tmp.sqlite'
    temp.unlink(missing_ok=True)
    meta = {'version': VERSION, 'requested_start': cfg.start(), 'requested_end': cfg.end(),
            'symbols_including_benchmark': int(prices.ticker.nunique()),
            'price_rows': len(prices), 'evidence_rows': 0 if evidence is None else len(evidence),
            'historical_membership_verified': False, 'current_pool_survivorship_bias': True,
            'not_a_live_quote_database': True, 'settings': asdict(cfg)}
    try:
        with sqlite3.connect(temp) as conn:
            _plain(prices).to_sql('prices', conn, index=False, if_exists='replace')
            conn.execute('CREATE UNIQUE INDEX prices_key ON prices(ticker,date)')
            if manifest.get('pool'):
                pd.DataFrame(manifest['pool']).to_sql('selected_universe',conn,index=False,if_exists='replace')
                conn.execute('CREATE UNIQUE INDEX selected_ticker ON selected_universe(ticker)')
            quality=manifest.get('universe_quality') or {}
            if quality.get('sector_comparison'):
                pd.DataFrame(quality['sector_comparison']).to_sql('universe_sector_coverage',conn,index=False,if_exists='replace')
            if evidence is not None and not evidence.empty:
                _plain(evidence).to_sql('evidence', conn, index=False, if_exists='replace')
                conn.execute('CREATE INDEX evidence_key ON evidence(ticker,available_at)')
            else:
                conn.execute('CREATE TABLE evidence(ticker TEXT,date TEXT,available_at TEXT,source TEXT)')
            pd.DataFrame([{'key': k, 'value': json.dumps(v, ensure_ascii=False)} for k,v in meta.items()]).to_sql('metadata',conn,index=False)
            row = conn.execute('PRAGMA integrity_check').fetchone()
            if row[0] != 'ok':
                raise ValueError('SQLite integrity check failed')
        os.replace(temp, database)
    finally:
        temp.unlink(missing_ok=True)
    export = root/'database_export'; export.mkdir(exist_ok=True)
    atomic_csv(export/'input_prices.csv.gz', prices)
    if evidence is not None and not evidence.empty:
        atomic_csv(export/'input_evidence.csv.gz', evidence)
    else:
        (export/'input_evidence.csv.gz').unlink(missing_ok=True)
    atomic_json(export/'input_manifest.json', manifest)
    if (root/'selected_universe.csv').exists():
        (export/'selected_universe.csv').write_bytes((root/'selected_universe.csv').read_bytes())
    if (root/'universe_catalog.csv').exists():
        (export/'universe_catalog.csv').write_bytes((root/'universe_catalog.csv').read_bytes())
    if manifest.get('universe_quality'):
        atomic_json(export/'universe_quality.json',manifest['universe_quality'])
    meta['sqlite_sha256'] = hashlib.sha256(database.read_bytes()).hexdigest()
    atomic_json(export/'database_manifest.json', meta)
    (export/'README.txt').write_text(
        'Alpha Radar V'+VERSION+' historical snapshot\n'
        'SQLite tables: prices, evidence, metadata. Dates and source are preserved.\n'
        'Optional fields may be missing or disabled; no zeros are fabricated.\n'
        'Current sample universe is not historical exchange membership.\n'
        'Replay: python run_v19.py --replay-bundle PATH.zip --dates 100 --years 3 --as-of YYYY-MM-DD\n'
        'Use --optional-data only if evidence is included. Downloaded data use is subject to provider terms.\n', encoding='utf-8')
    out = root/'v192_history_database.zip'
    tmp = out.with_suffix('.tmp.zip')
    with ZipFile(tmp, 'w', compression=ZIP_DEFLATED) as z:
        z.write(database, 'history.sqlite')
        for f in sorted(export.iterdir()):
            if f.name in ('input_prices.csv.gz','input_evidence.csv.gz','input_manifest.json','database_manifest.json','README.txt',
                          'selected_universe.csv','universe_catalog.csv','universe_quality.json'):
                z.write(f, f.name)
    os.replace(tmp, out)
    return out
