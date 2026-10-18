"""Atomic JSON case checkpoints; data/evidence/config/code changes invalidate them.

No pickle or executable files are loaded. Checkpoints only reduce recomputation;
completed-case counts and missing outcomes are not silently changed on resume.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import pandas as pd
from v19_data import atomic_json, read_json
from v19_config import VERSION


class RunCancelled(RuntimeError):
    pass


def frame_digest(frame):
    if frame is None or frame.empty:
        return 'EMPTY'
    data = frame.copy().reindex(sorted(frame.columns), axis=1).astype(str)
    # Ignore DataFrame index but retain all price and optional-evidence contents.
    sort_by = [c for c in ('ticker', 'date', 'available_at', 'source') if c in data]
    data = data.sort_values(sort_by or list(data.columns), kind='stable').reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(data, index=False).values.tobytes()).hexdigest()


def code_digest():
    root = Path(__file__).resolve().parent
    # Includes every numerical/replay module used by this run, not UI or docs.
    names = ('v19_config.py', 'v19_features.py', 'v19_backtest.py', 'v191_selection.py',
             'v192_fast.py', 'v192_sampling.py', 'v192_checkpoint.py', 'v192_factors.py', 'v191_reporting.py',
             'v193_factors.py','v193_selection.py','v193_fast.py','v194_factors.py',
             'v194_selection.py','v194_evidence.py','v194_reporting.py','v195_selection.py','v196_factors.py','v196_selection.py','v1961_evidence.py')
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode()); h.update((root/name).read_bytes())
    return h.hexdigest()


def run_identity(prices, evidence, cfg):
    obj = {'version': VERSION, 'prices': frame_digest(prices),
           'evidence': frame_digest(evidence) if cfg.optional_data else 'DISABLED',
           'settings': asdict(cfg), 'code': code_digest()}
    digest = hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
    return digest, obj


class CaseStore:
    def __init__(self, root, identity, metadata):
        self.root = Path(root)/identity if root is not None else None
        self.identity = identity
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            atomic_json(self.root/'run_manifest.json', dict(metadata, identity=identity))

    def load(self, date, horizon):
        if self.root is None:
            return None
        obj = read_json(self.root/f'{pd.Timestamp(date).date()}_{int(horizon)}.json')
        if not isinstance(obj, dict):
            return None
        if (obj.get('identity') != self.identity or not obj.get('complete_checkpoint')
                or obj.get('date') != str(pd.Timestamp(date).date())
                or obj.get('horizon') != int(horizon)):
            return None
        payload = obj.get('payload')
        if not isinstance(payload, dict) or not {'records', 'trades', 'fit', 'trials', 'factors'}.issubset(payload):
            return None
        check = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if check != obj.get('payload_hash'):
            return None
        return payload

    def save(self, date, horizon, payload):
        if self.root is None:
            return
        path = self.root/f'{pd.Timestamp(date).date()}_{int(horizon)}.json'
        # Normalize numpy scalars, dates, and missing values through our JSON writer.
        staging = path.with_suffix('.staging.json')
        atomic_json(staging, payload)
        clean = read_json(staging)
        check = hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        atomic_json(path, {'identity': self.identity, 'complete_checkpoint': True,
                          'date': str(pd.Timestamp(date).date()), 'horizon': int(horizon),
                          'payload_hash': check, 'payload': clean})
        staging.unlink(missing_ok=True)
