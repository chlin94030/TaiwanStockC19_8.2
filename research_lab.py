"""Alpha Radar research laboratory: restartable, purged, dated factor search.

This script deliberately cannot approve a portfolio for live trading. It reads
recorded factor panels (including their stored cost-aware outcome fields) and
keeps development tuning separate from a final evaluation segment. An old
exposed segment is *never* represented as new independent evidence.

Usage:
  python research_lab.py --bundle V198_RESEARCH.zip --output lab --rounds 15
  python research_lab.py --bundle V198_RESEARCH.zip --output lab --rounds 0 \
      --patience 15 --max-minutes 20 --resume
  python research_lab.py --bundle NEW_BUNDLE.zip --output forward \
      --frozen-model lab/frozen_models.json --fresh-after 2026-10-15
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time
import zipfile

import numpy as np
import pandas as pd

FEATURES = (
    'p197_annual_high', 'p197_downside', 'p197_compression',
    'p197_liquidity', 'p197_trend_eff', 'p197_relative',
    'r196_dollar_turnover', 'r196_medium_return',
    'p197_market_gate',
)
# Market-gate is shared by all stocks on a date, so it is NOT a ranking factor.
RANK_FACTORS = tuple(x for x in FEATURES if x != 'p197_market_gate')
OPTIONAL = ('p197_foreign', 'p197_trust', 'p197_revenue',
            'p197_quality', 'p197_valuation')
EXPOSED_CUTOFF = '2026-10-15'


@dataclass(frozen=True)
class Trial:
    horizon: int
    weights: dict[str, float]
    source: str


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.tmp')
    part.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                               allow_nan=False, default=str), encoding='utf-8')
    os.replace(part, path)


def safe_number(value):
    x = float(value)
    return x if math.isfinite(x) else None


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def read_factor_panel(bundle, factors, horizons):
    needed = {'date', 'ticker', 'eligible'} | set(factors)
    for h in horizons:
        needed |= {f'net_{h}', f'mae_{h}', f'return_basis_{h}',
                   f'exit_date_{h}'}
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        name = 'factor_panel.csv.gz'
        if name not in names:
            raise ValueError('Bundle lacks factor_panel.csv.gz; export the V19.8 research diagnostic bundle.')
        info = archive.getinfo(name)
        if info.file_size > 8 * 1024 * 1024 * 1024:
            raise ValueError('Factor panel exceeds the 8 GiB safety ceiling')
        with archive.open(name) as stream:
            panel = pd.read_csv(stream, compression='gzip',
                                usecols=lambda x: x in needed,
                                low_memory=False)
    missing = sorted((needed - set(panel.columns)) - (set(factors) - set(RANK_FACTORS) - set(OPTIONAL)))
    if missing:
        raise ValueError('Missing required panel columns: ' + ', '.join(missing))
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce')
    panel = panel.dropna(subset=['date', 'ticker'])
    if panel.duplicated(['date', 'ticker']).any():
        raise ValueError('Duplicate ticker/date keys in factor panel')
    flag = panel['eligible']
    if not pd.api.types.is_bool_dtype(flag):
        panel['eligible'] = flag.astype(str).str.lower().isin(('true', '1'))
    for factor in factors:
        if factor not in panel:
            panel[factor] = np.nan
    for h in horizons:
        panel[f'exit_date_{h}'] = pd.to_datetime(panel[f'exit_date_{h}'], errors='coerce')
    return panel


class DateMatrix:
    """Factor ranks known as of T, and separately observed T+h outcomes."""
    def __init__(self, panel, factors, horizons, top_k=5, benchmark='0050.TW'):
        self.factors = tuple(factors)
        self.horizons = tuple(horizons)
        self.top_k = int(top_k)
        self.dates = pd.DatetimeIndex(sorted(panel.date.drop_duplicates().unique()))
        stocks = panel.loc[panel.ticker.ne(benchmark), 'ticker'].unique()
        self.tickers = tuple(sorted(stocks))
        n, s = len(self.dates), len(self.tickers)
        if n < 200 or s < self.top_k:
            raise ValueError('Insufficient calendar/history or stock universe')
        tindex = pd.Series(np.arange(n), index=self.dates)
        sin = {ticker: k for k, ticker in enumerate(self.tickers)}
        stocks = panel[panel.ticker.ne(benchmark)].copy()
        stocks['_d'] = stocks.date.map(tindex)
        stocks['_s'] = stocks.ticker.map(sin)
        self.eligible = np.zeros((n, s), dtype=bool)
        idx = (stocks['_d'].to_numpy(int), stocks['_s'].to_numpy(int))
        self.eligible[idx] = stocks.eligible.to_numpy(bool)
        self.scores = {}
        for f in factors:
            a = pd.to_numeric(stocks[f], errors='coerce').replace([np.inf, -np.inf], np.nan)
            a = a.where(stocks.eligible)
            # Do not normalize over the benchmark or incomplete/noneligible stocks.
            ranked = a.groupby(stocks.date, sort=False).rank(pct=True, method='average')
            matrix = np.full((n, s), np.nan, dtype=np.float32)
            matrix[idx] = ranked.to_numpy(np.float32)
            self.scores[f] = matrix
        bench = panel.loc[panel.ticker.eq(benchmark)].set_index('date').reindex(self.dates)
        self.returns = {}
        self.drawdowns = {}
        self.bases = {}
        self.exit_dates = {}
        self.benchmark = {}
        self.benchmark_basis = {}
        for h in horizons:
            returns = np.full((n, s), np.nan, dtype=np.float64)
            dd = np.full((n, s), np.nan, dtype=np.float64)
            basis = np.full((n, s), 'UNKNOWN', dtype='U24')
            ends = np.full((n, s), np.datetime64('NaT'), dtype='datetime64[ns]')
            returns[idx] = pd.to_numeric(stocks[f'net_{h}'], errors='coerce').to_numpy(float)
            dd[idx] = pd.to_numeric(stocks[f'mae_{h}'], errors='coerce').to_numpy(float)
            basis[idx] = stocks[f'return_basis_{h}'].fillna('UNKNOWN').astype(str).to_numpy()
            ends[idx] = stocks[f'exit_date_{h}'].to_numpy(dtype='datetime64[ns]')
            self.returns[h] = returns
            self.drawdowns[h] = dd
            self.bases[h] = basis
            self.exit_dates[h] = ends
            self.benchmark[h] = pd.to_numeric(bench[f'net_{h}'], errors='coerce').to_numpy(float)
            self.benchmark_basis[h] = bench[f'return_basis_{h}'].fillna('UNKNOWN').astype(str).to_numpy()
        self._cache = {}

    def evaluate(self, trial):
        h = trial.horizon
        if h not in self.horizons:
            raise ValueError('Unknown horizon')
        weights = {k: float(v) for k, v in trial.weights.items() if float(v) > 0}
        if not weights or not set(weights).issubset(self.scores):
            raise ValueError('Trial references missing or empty feature support')
        if min(weights.values()) <= 0 or abs(sum(weights.values()) - 1) > 1e-6:
            raise ValueError('Weights must be positive and sum to one')
        key = (h, tuple(sorted((k, round(v, 10)) for k, v in weights.items())))
        if key in self._cache:
            return self._cache[key]
        signal = np.zeros(self.eligible.shape, dtype=np.float64)
        complete = self.eligible.copy()
        for name, value in weights.items():
            feature = self.scores[name]
            complete &= np.isfinite(feature)
            signal += value * np.nan_to_num(feature, nan=0.0)
        signal[~complete] = -np.inf
        pick = np.argsort(-signal, axis=1, kind='stable')[:, :self.top_k]
        enough = complete.sum(axis=1) >= self.top_k
        selected = np.take_along_axis(self.returns[h], pick, axis=1)
        bases = np.take_along_axis(self.bases[h], pick, axis=1)
        reference_basis = self.benchmark_basis[h][:, None]
        compatible = ((bases == reference_basis) | (bases == 'CASH'))
        acceptable_basis = (reference_basis != 'UNKNOWN') & (reference_basis != 'MIXED_UNVERIFIED')
        valid = (enough & np.isfinite(selected).all(axis=1) &
                 compatible.all(axis=1) & acceptable_basis[:, 0] &
                 np.isfinite(self.benchmark[h]))
        out = np.full(len(self.dates), np.nan)
        out[valid] = selected.mean(axis=1)[valid]
        mae = np.take_along_axis(self.drawdowns[h], pick, axis=1)
        adverse = np.full(len(self.dates), np.nan)
        valid_mae = valid & np.isfinite(mae).all(axis=1)
        adverse[valid_mae] = mae.mean(axis=1)[valid_mae]
        ans = {'returns': out, 'mae': adverse,
               'benchmark': self.benchmark[h].copy(), 'valid': valid,
               'eligible_dates': enough, 'selected': pick}
        self._cache[key] = ans
        return ans


def nonoverlap_count(mask, h):
    last = -10_000
    n = 0
    for i in np.flatnonzero(mask):
        if int(i) - last >= int(h):
            n += 1
            last = int(i)
    return n


def describe_model(matrix, trial, selector):
    outcome = matrix.evaluate(trial)
    usable = selector & outcome['valid']
    r = outcome['returns'][usable]
    b = outcome['benchmark'][usable]
    x = r - b
    paired = int(len(x))
    coverage = float(paired / max(1, selector.sum()))
    return dict(n_dates=paired, requested_dates=int(selector.sum()), coverage=coverage,
                nonoverlap=nonoverlap_count(usable, trial.horizon),
                net_mean=safe_number(r.mean()) if paired else None,
                net_median=safe_number(np.median(r)) if paired else None,
                reference_mean=safe_number(b.mean()) if paired else None,
                excess_mean=safe_number(x.mean()) if paired else None,
                excess_std=safe_number(x.std()) if paired else None,
                beat_reference_fraction=safe_number((r > b).mean()) if paired else None,
                downside_mean=safe_number(outcome['mae'][usable][np.isfinite(outcome['mae'][usable])].mean())
                    if np.isfinite(outcome['mae'][usable]).any() else None)


def selection_score(stats, input_count):
    if stats['coverage'] < .90 or stats['n_dates'] < 35 or stats['nonoverlap'] < 4:
        return -float('inf')
    excess = stats['excess_mean']
    scatter = stats['excess_std'] or 0.0
    risk = stats['downside_mean'] if stats['downside_mean'] is not None else -0.15
    # Conservative research score, never a proof of expected alpha.
    return float(excess - .15 * scatter + .20 * min(risk, 0.0)
                 - .0004 * (input_count - 1))


def presets(h):
    h52='p197_annual_high'; down='p197_downside'
    comp='p197_compression'; liquid='p197_liquidity'
    initial = {f'one_{k}':{k: 1.0} for k in RANK_FACTORS}
    initial.update({
        'v198_high_defense': {h52:.65,down:.35},
        'v198_high_compress':{h52:.60,comp:.40},
        'v198_liquid_compress':{liquid:.65,comp:.35},
        'v198_liquid_defense':{liquid:.70,down:.30},
        'equal_three': {h52:1/3,down:1/3,liquid:1/3},
    })
    if h <= 7:
        initial['near_high_momentum'] = {h52:.5, 'r196_medium_return':.5}
    elif h <= 30:
        initial['compression_strength'] = {comp:.50, 'p197_relative':.50}
    else:
        initial['turnover_trend'] = {'r196_dollar_turnover':.55, 'p197_trend_eff':.45}
    return initial


def propose(rng, bank, max_inputs):
    size = rng.randint(1, max_inputs)
    keys = rng.sample(bank, size)
    # Dirichlet-like random convex weights without optional scientific packages.
    draws = [max(rng.gammavariate(1.8, 1.0), .01) for _ in keys]
    sm = sum(draws)
    return {k: v / sm for k, v in zip(keys, draws)}


def protocol_key(bundle_hash, factors, horizons, seed, max_inputs, batch=12, min_progress=.00015):
    doc = [bundle_hash, list(factors), list(horizons), seed, max_inputs,
           batch, min_progress, 'v1981-datepurge-research-v2']
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def dated_masks(matrix, h):
    total = len(matrix.dates)
    if total < 300:
        raise ValueError('Insufficient history for a separated development and audit segment')
    # Common *signal-day* membership, independent of future returns. Earlier
    # dates lack a completed 252-bar high signal; including them in the
    # denominator would falsely make every candidate look incomplete.
    anchor = np.isfinite(matrix.scores['p197_annual_high']).sum(axis=1) >= matrix.top_k
    eligible_indices = np.flatnonzero(anchor)
    if len(eligible_indices) < 120:
        raise ValueError('Not enough common past-observable dates for research')
    split = int(eligible_indices[int(len(eligible_indices) * .80)])
    # Purge h sessions (rather than accidentally reusing overlapping outcomes).
    dev_end = split - h
    dev = (np.arange(total) < dev_end) & anchor
    final = (np.arange(total) >= split) & anchor
    # These flags are benchmark-maturity checks, common to all candidates.
    dev &= np.isfinite(matrix.benchmark[h])
    final &= np.isfinite(matrix.benchmark[h])
    return dev, final, split


def optimise(args, matrix, bundle_hash):
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    fingerprints = protocol_key(bundle_hash, matrix.factors, args.horizons,
                                args.seed, args.max_inputs, args.batch, args.min_progress)
    protocols = {
        'design': 'NO_UNLIMITED_HISTORICAL_EVIDENCE',
        'algorithm': 'fixed sparse positive weights; date separated and purged',
        'data_sha256': bundle_hash,
        'features': matrix.factors,
        'horizons': args.horizons,
        'seed': args.seed,
        'support_limit': args.max_inputs,
        'batch': args.batch,
        'min_progress': args.min_progress,
        'optional_timing_verified': False if args.include_optional else None,
        'protocol_hash': fingerprints,
        'input_exposure': 'EXPOSED_DEVELOPMENT_NOT_INDEPENDENT',
        'approved_for_live': False,
    }
    atomic_json(outdir/'protocol.json', protocols)
    if (outdir/'sealed.json').exists():
        raise ValueError('This experiment is sealed. Use a new output folder and new evidence.')
    checkpoint = outdir/'checkpoint.json'
    if args.finalize and not checkpoint.exists():
        raise ValueError('Run development search before --finalize. No audit on unfrozen rules.')
    if args.reset and checkpoint.exists():
        checkpoint.unlink()
    if checkpoint.exists():
        if not args.resume:
            raise ValueError('Checkpoint exists; choose --resume or --reset')
        current = json.loads(checkpoint.read_text(encoding='utf-8'))
        if current['protocol_hash'] != fingerprints:
            raise ValueError('Dataset/feature/protocol mismatch: use a different --output folder')
    else:
        current = {'protocol_hash': fingerprints, 'horizons': {}}
    frozen = {'protocol': protocols, 'models': {}}
    started = time.monotonic()
    deadline = started + 60 * args.max_minutes
    for h in args.horizons:
        dev, audit, split = dated_masks(matrix, h)
        prefix = current['horizons'].get(str(h), {})
        prev_round = int(prefix.get('rounds', 0))
        winner = prefix.get('winner')
        best_score = float(prefix['score']) if prefix.get('score') is not None else -float('inf')
        no_better = int(prefix.get('no_better', 0))
        history = prefix.get('history', [])
        seen = set(prefix.get('seen', []))
        rng = random.Random(args.seed + h * 100_003)
        # Advance RNG deterministically on resume, no recycling sampled weights.
        for _ in range(prev_round * args.batch):
            propose(rng, matrix.factors, args.max_inputs)
        round_id = prev_round
        while (not args.finalize and no_better < args.patience and
               (args.rounds == 0 or round_id < args.rounds) and time.monotonic() < deadline):
            batch = []
            if round_id == 0:
                batch += [(k, v) for k, v in presets(h).items()
                          if set(v).issubset(matrix.factors)]
            for _ in range(args.batch):
                batch.append(('generated', propose(rng, matrix.factors, args.max_inputs)))
            winner_changed = False
            for label, weights in batch:
                ident = ';'.join(f'{k}={v:.8f}' for k, v in sorted(weights.items()))
                if ident in seen:
                    continue
                seen.add(ident)
                trial = Trial(h, weights, label)
                stats = describe_model(matrix, trial, dev)
                score = selection_score(stats, len(weights))
                if score > best_score + args.min_progress:
                    best_score = score
                    winner = {'source': label, 'weights': weights,
                              'development': stats, 'criterion': score}
                    winner_changed = True
            round_id += 1
            no_better = 0 if winner_changed else no_better + 1
            history.append({'round': round_id, 'best_score': safe_number(best_score),
                            'no_better': no_better, 'seen_candidates': len(seen)})
            current['horizons'][str(h)] = dict(rounds=round_id, score=safe_number(best_score),
                                              no_better=no_better, winner=winner,
                                              history=history, seen=sorted(seen))
            atomic_json(checkpoint, current)
            if no_better >= args.patience:
                break
        if winner is None:
            frozen['models'][str(h)] = {'status': 'NO_CANDIDATE_PASSED_MINIMUM_EVIDENCE',
                                       'independent_approval': False}
            continue
        model = {'weights': winner['weights'], 'source': winner['source'],
                 'development': winner['development'], 'rounds': round_id,
                 'candidate_count': len(seen), 'independent_approval': False}
        if not args.finalize:
            # This file contains NO audit-period return information whatsoever.
            model['status'] = 'EXPLORATION_ONLY_AUDIT_NOT_EVALUATED'
            model['audit'] = None
        else:
            trial = Trial(h, winner['weights'], winner['source'])
            atomic_json(outdir/f'frozen_{h}.json', dict(protocol_hash=fingerprints,
                        cutoff_date=str(matrix.dates[split].date()), horizon=h,
                        weights=winner['weights'], development=winner['development'],
                        status='FROZEN_EXPOSED_DEVELOPMENT'))
            held = describe_model(matrix, trial, audit)
            needed = {5:30,20:12,60:8}.get(h,max(8,240//h))
            development_bad = (winner['development']['excess_mean'] is None or
                winner['development']['excess_mean'] <= 0 or
                winner['development']['beat_reference_fraction'] <= .50)
            model['audit'] = held
            model['status'] = ('RESEARCH_NEGATIVE_DEVELOPMENT_EXCESS' if development_bad else
                    'AUDIT_UNDERPOWERED' if held['nonoverlap'] < needed else
                    'EXPOSED_AUDIT_NOT_NEW_HOLDOUT')
            model['independent_windows_required'] = needed
        frozen['models'][str(h)] = model
    if args.finalize:
        frozen['sealed'] = True
        # Sealing prevents further tuning on an audit already inspected.
        atomic_json(outdir/'frozen_models.json', frozen)
        atomic_json(outdir/'sealed.json', {'sealed': True,
                    'protocol_hash': fingerprints, 'input_sha256': bundle_hash,
                    'warning': 'This exposed data is NOT independent future evidence.'})
    else:
        frozen['sealed'] = False
        atomic_json(outdir/'exploratory_models.json', frozen)
    status_file = {'input_sha256': bundle_hash, 'results': frozen['models'],
                   'audit_exposed': bool(args.finalize),
                   'note': 'Not approved: repeated historical trials are not new independent data.'}
    atomic_json(outdir/'research_summary.json', status_file)
    return status_file


def forward(args, matrix, bundle_hash):
    if not args.fresh_after:
        raise ValueError('--fresh-after is required for a frozen forward-only evaluation')
    entry = json.loads(Path(args.frozen_model).read_text(encoding='utf-8'))
    if entry.get('sealed') is not True:
        raise ValueError('Only sealed frozen_models.json may enter forward evaluation')
    src_hash = entry['protocol']['data_sha256']
    if src_hash == bundle_hash:
        raise ValueError('Forward test requires a new data snapshot, not a recycled ZIP')
    if args.fresh_after < EXPOSED_CUTOFF:
        raise ValueError('Fresh date must be after the last exposed development bar')
    threshold = pd.Timestamp(args.fresh_after)
    out = {'forward_data_sha256': bundle_hash, 'frozen_sha256': digest_file(args.frozen_model),
           'fresh_after': args.fresh_after, 'results': {}, 'auto_trade_enabled': False}
    for text_h, model in entry['models'].items():
        h = int(text_h)
        if not model.get('weights'):
            continue
        mask = matrix.dates > threshold
        trial = Trial(h, model['weights'], model.get('source', 'frozen'))
        stat = describe_model(matrix, trial, mask)
        min_independent = {5:30, 20:12, 60:8}.get(h, max(8, 240//h))
        can_review = (stat['nonoverlap'] >= min_independent and stat['coverage'] >= .90
                      and stat['excess_mean'] is not None and stat['excess_mean'] > 0
                      and stat['beat_reference_fraction'] is not None
                      and stat['beat_reference_fraction'] > .55)
        out['results'][text_h] = {'metrics': stat, 'independent_windows_required': min_independent,
                                 'review_status': 'ELIGIBLE_FOR_MANUAL_REVIEW' if can_review
                                     else 'MORE_FORWARD_EVIDENCE_REQUIRED',
                                 'approved_for_live': False}
    atomic_json(Path(args.output)/'forward_evaluation.json', out)
    return out


def cli(argv=None):
    p = argparse.ArgumentParser(description='Restartable factor/weight research; never auto trades')
    p.add_argument('--bundle', required=True, help='V19.8 historical diagnostic or research ZIP')
    p.add_argument('--output', default='alpha_research_lab')
    p.add_argument('--horizons', type=int, nargs='+', choices=(5,20,60), default=[5,20,60])
    p.add_argument('--rounds', type=int, default=12,
                   help='0 = no fixed round count; patience and time limit still apply')
    p.add_argument('--batch', type=int, default=12)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--max-minutes', type=float, default=15.0)
    p.add_argument('--min-progress', type=float, default=.00015)
    p.add_argument('--max-inputs', type=int, choices=(1,2,3,4), default=3)
    p.add_argument('--seed', type=int, default=1981)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--finalize', action='store_true',
                   help='One-time audit; seals this experiment against later retuning')
    p.add_argument('--reset', action='store_true')
    p.add_argument('--include-optional', action='store_true')
    p.add_argument('--frozen-model', default='')
    p.add_argument('--fresh-after', default='')
    args = p.parse_args(argv)
    if args.rounds < 0 or args.batch < 1 or args.patience < 1 or args.max_minutes <= 0:
        p.error('Invalid search budget/patience/time limit')
    factors = RANK_FACTORS + (OPTIONAL if args.include_optional else ())
    bundle_hash = digest_file(args.bundle)
    print(f'Reading dated factor panel | snapshot {bundle_hash[:14]}', flush=True)
    panel = read_factor_panel(args.bundle, factors, args.horizons)
    dates = panel.date.nunique()
    tickers = panel.ticker.nunique()
    print(f'Input: {len(panel)} rows, {tickers} tickers, {dates} dates', flush=True)
    matrix = DateMatrix(panel, factors, args.horizons)
    result = forward(args, matrix, bundle_hash) if args.frozen_model else optimise(args, matrix, bundle_hash)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    cli()
