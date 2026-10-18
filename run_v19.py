"""CLI: python run_v19.py --years 3 --stocks 600 --dates 100 --horizons 5,20,60"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
from v19_config import Settings
from v19_data import download_history,read_json,atomic_json
from v19_service import load_inputs,merge_imported_evidence
from v19_backtest import analyze,save_analysis

def main(argv=None):
    p=argparse.ArgumentParser(description='Alpha Radar V19.8 stratified-market-universe walk-forward research')
    p.add_argument('--years',type=int,default=3);p.add_argument('--stocks',type=int,default=600)
    p.add_argument('--dates',type=int,default=100);p.add_argument('--horizons',default='5,20,60')
    p.add_argument('--trials',type=int,default=24);p.add_argument('--seed',type=int,default=1920)
    p.add_argument('--as-of',default='');p.add_argument('--data',default='data/v19')
    p.add_argument('--output',default='data/v19/cli_report');p.add_argument('--cache-only',action='store_true')
    p.add_argument('--optional-data',action='store_true');p.add_argument('--tickers',default='')
    p.add_argument('--reference-report',default='')
    p.add_argument('--replay-bundle',default='')
    p.add_argument('--raw-first',action='store_true')
    p.add_argument('--sampling',choices=['uniform','stratified'],default='uniform')
    p.add_argument('--universe-seed',type=int,default=1900)
    p.add_argument('--universe-mode',choices=['balanced','proportional','full_current','legacy_random'],default='balanced')
    p.add_argument('--min-download-coverage',type=float,default=.80)
    p.add_argument('--request-budget',type=int,default=900)
    p.add_argument('--download-only',action='store_true')
    p.add_argument('--refresh-history',action='store_true')
    p.add_argument('--fresh-run',action='store_true',help='Ignore validation checkpoints for this run')
    p.add_argument('--factor-design',choices=['legacy_v192','expanded_v193','sparse_v194','sparse_v195','tournament_v196','ten_angle_v197','balanced_v198'],default='balanced_v198')
    p.add_argument('--no-factor-audit',action='store_true')
    p.add_argument('--max-factors',type=int,default=6,help='Maximum underlying inputs in a sparse candidate')
    p.add_argument('--search-budget',type=int,default=160,help='Bounded number of support/method candidates per cutoff/horizon')
    p.add_argument('--financial-timing',choices=['disclosure_required','lagged_research'],default='disclosure_required')
    p.add_argument('--no-financial-history',action='store_true')
    a=p.parse_args(argv)
    replay=()
    if a.reference_report:
        from v191_audit import parse_markdown
        replay=tuple(parse_markdown(Path(a.reference_report).read_text(encoding='utf-8-sig'))['cutoffs'])
        if len(replay)!=a.dates:raise ValueError('dates must match the reference report')
    cfg=Settings(years=a.years,stocks=a.stocks,dates=a.dates,horizons=tuple(int(x) for x in a.horizons.split(',')),
                 trials=a.trials,seed=a.seed,as_of=a.as_of,optional_data=a.optional_data,
                 prefer_adjusted=not a.raw_first,replay_cutoffs=replay,
                 sampling_method=a.sampling,universe_seed=a.universe_seed,universe_mode=a.universe_mode,
                 min_download_coverage=a.min_download_coverage,request_budget=a.request_budget,
                 refresh_history=a.refresh_history,factor_audit=not a.no_factor_audit,factor_design=a.factor_design,
                 sparse_max_inputs=a.max_factors,sparse_budget=a.search_budget,
                 financial_timing=a.financial_timing,financial_history=not a.no_financial_history)
    cfg=replace(cfg,as_of=cfg.end())
    def progress(stage,i,total,msg):print(f'[{stage}] {i}/{total}: {msg}',flush=True)
    if a.replay_bundle:
        from v191_reporting import read_bundle
        prices,evidence,manifest=read_bundle(a.replay_bundle)
    elif a.cache_only:
        prices,evidence=load_inputs(a.data,cfg);manifest=read_json(Path(a.data)/'download_manifest.json')
    else:
        tickers=[s.strip() for s in a.tickers.split(',') if s.strip()] or None
        prices,evidence,manifest=download_history(a.data,cfg,progress,tickers=tickers)
        evidence=merge_imported_evidence(a.data,evidence,cfg)
        if not a.download_only and (manifest.get('download_completeness',1.0)<cfg.min_download_coverage or
                (cfg.factor_design=='balanced_v198' and cfg.universe_mode!='full_current' and
                 'pool_equities' in manifest and manifest.get('requested_target_coverage',0.0)<cfg.min_download_coverage)) :
            atomic_json(Path(a.output)/'run_status.json',{'status':'INCOMPLETE_UNIVERSE_RESUME_REQUIRED',
                'coverage':manifest.get('download_completeness'),'requested':manifest.get('pool_equities'),
                'downloaded':manifest.get('ready_equities'),'manifest':manifest})
            print('Only part of the selected equity universe downloaded; resume with the same settings before validation.')
            return 4
    if a.cache_only and cfg.factor_design=='balanced_v198' and cfg.universe_mode!='full_current' and len(set(prices.ticker)-{cfg.benchmark})<cfg.stocks*cfg.min_download_coverage:
        atomic_json(Path(a.output)/'run_status.json',{'status':'CACHE_POOL_INCOMPLETE','requested':cfg.stocks,
            'actual':len(set(prices.ticker)-{cfg.benchmark}),'required_fraction':cfg.min_download_coverage})
        print('Cache is smaller than requested broad universe; expand online first or set --stocks to its actual size.')
        return 4
    if prices.empty:
        atomic_json(Path(a.output)/'run_status.json',{'status':'NO_PRICE_DATA','source_manifest':manifest})
        print('No prices received. Inspect source manifest; caches and earlier reports have been preserved.')
        return 2
    if a.download_only:
        from v192_history import export_database
        archive=export_database(a.data,prices,evidence,manifest,cfg)
        print('DATABASE_SAVED | '+str(archive.resolve()))
        return 0
    report,panel,trades=analyze(prices,evidence,cfg,progress,manifest,
                              checkpoint_dir=None if a.fresh_run else Path(a.data)/'checkpoints')
    save_analysis(a.output,report,panel,trades,prices,evidence)
    print(report['status']+' | '+str(Path(a.output).resolve()))
    return 0 if report.get('cutoffs') else 3

if __name__=='__main__':raise SystemExit(main())
