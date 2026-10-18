from pathlib import Path
import sys,time,json,os
# Run from the unpacked app folder; no network or production data is used.
import numpy as np,pandas as pd
from v19_config import Settings
from v19_backtest import analyze,save_analysis
from v19_data import atomic_json
from v192_history import export_database

def main():
    import argparse
    parser=argparse.ArgumentParser(description='SYNTHETIC ONLY: 100-date full pipeline test; not investment performance')
    parser.add_argument('--output',default='data/v192_synthetic_test')
    args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(992)
    dates=pd.bdate_range(end='2026-10-15',periods=750)
    frames=[]
    for i in range(101):
        ticker='0050.TW' if i==0 else f'{2000+i}.TW'
        ret=rng.normal(.0004,.012,len(dates))+.001*np.sin(np.arange(len(dates))/37+i)
        close=(80+i)*np.exp(np.cumsum(ret));op=close*np.exp(rng.normal(0,.004,len(dates)))
        frames.append(pd.DataFrame({'date':dates,'ticker':ticker,'name':'SYNTHETIC_'+ticker,'sector':'TEST_ONLY',
            'open':op,'high':np.maximum(op,close)*1.008,'low':np.minimum(op,close)*.992,
            'close':close,'adj_close':close,'volume':rng.integers(1000000,3000000,len(dates)).astype(float),
            'source':'SYNTHETIC_TEST_ONLY'}))
    prices=pd.concat(frames,ignore_index=True)
    cfg=Settings(as_of='2026-10-15',stocks=100,dates=100,trials=24)
    t=time.perf_counter()
    def progress(stage,done,total,message):
        if done%10==0 or done==total:print(stage,done,total,message,'elapsed',round(time.perf_counter()-t,1),flush=True)
    r,p,trades=analyze(prices,cfg=cfg,progress=progress,manifest={'scope':'SYNTHETIC_TEST_ONLY'},checkpoint_dir=root/'checkpoints')
    elapsed=time.perf_counter()-t
    assert len(r['cutoffs'])==100 and len(set(r['cutoffs']))==100
    assert len(r['fits'])==300 and r['expected_adaptive_cases']==300
    assert r['data_kind']=='SYNTHETIC_TEST_ONLY'
    assert len(r['factor_cases'])==6000 and len(r['factor_summary'])==60
    assert len(r['results'])==1500 and len(trades)==3000
    save_analysis(root/'result',r,p,trades,prices)
    export_database(root/'database',prices,None,{'scope':'SYNTHETIC_TEST_ONLY'},cfg)
    summary={'status':'PASS','data_kind':r['data_kind'],'symbols_including_benchmark':101,'price_rows':len(prices),
             'cutoffs':100,'horizons':list(cfg.horizons),'validation_cases':300,'factor_cases':6000,
             'selected_trade_rows':len(trades),'results_rows':len(r['results']),'elapsed_seconds':round(elapsed,2),
             'new_weights_market_validated':False,'not_real_market_returns':True,'status_counts':r['fit_status_counts']}
    atomic_json(root/'test_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__':
    main()
