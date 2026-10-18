"""Public 1--3 year daily history, resumable cache, explicit provenance.

FinMind -> Yahoo chart -> bounded official TWSE monthly fallback. No credentials
are required by the code; provider entitlement/rate limits are never bypassed.
Existing V18 SQLite files are READ ONLY. Optional evidence cannot block prices.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta
import gzip
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
from v19_config import Settings, TZ

FINMIND = 'https://api.finmindtrade.com/api/v4/data'
YAHOO = 'https://query1.finance.yahoo.com/v8/finance/chart/'
TWSE = 'https://www.twse.com.tw/exchangeReport/STOCK_DAY'
COLS = ['date','ticker','open','high','low','close','volume','adj_close','source']
CACHE_LOCK = threading.RLock()
# A small, disclosed rescue universe only used when universe discovery fails.
# This is NOT represented as all TWSE/TPEx equities or as a historical universe.
FALLBACK = [
 ('1101.TW','Taiwan Cement'),('1216.TW','Uni-President'),('1301.TW','Formosa Plastics'),
 ('1303.TW','Nan Ya'),('1326.TW','Formosa Chemicals'),('1402.TW','Far Eastern New Century'),
 ('2002.TW','China Steel'),('2105.TW','Cheng Shin'),('2207.TW','Hotai Motor'),
 ('2301.TW','Lite-On'),('2303.TW','UMC'),('2308.TW','Delta'),('2317.TW','Hon Hai'),
 ('2324.TW','Compal'),('2330.TW','TSMC'),('2345.TW','Accton'),('2356.TW','Inventec'),
 ('2357.TW','ASUS'),('2376.TW','Gigabyte'),('2377.TW','MSI'),('2379.TW','Realtek'),
 ('2382.TW','Quanta'),('2383.TW','Elite Material'),('2408.TW','Nanya Technology'),
 ('2409.TW','AUO'),('2454.TW','MediaTek'),('2603.TW','Evergreen'),('2609.TW','Yang Ming'),
 ('2615.TW','Wan Hai'),('2801.TW','Chang Hwa Bank'),('2881.TW','Fubon Financial'),
 ('2882.TW','Cathay Financial'),('2886.TW','Mega Financial'),('2891.TW','CTBC Financial'),
 ('3008.TW','Largan'),('3017.TW','Asia Vital Components'),('3034.TW','Novatek'),
 ('3037.TW','Unimicron'),('3231.TW','Wistron'),('3443.TW','Global Unichip'),
 ('3711.TW','ASE'),('4938.TW','Pegatron'),('5871.TW','Chailease'),('6505.TW','Formosa Petrochemical'),
 ('9910.TW','Feng Tay'),('9914.TW','Merida'),('9933.TW','CTCI'),('9945.TW','Ruentex Development'),
 ('3105.TWO','Win Semiconductors'),('4128.TWO','Microbio'),('5347.TWO','Vanguard'),
 ('5483.TWO','Sino-American Silicon'),('6187.TWO','All Ring'),('8299.TWO','Phison'),
]

class SourceError(RuntimeError): pass
class BudgetExceeded(SourceError): pass


def atomic_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def clean(x):
        if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [clean(v) for v in x]
        if isinstance(x,(np.integer,)): return int(x)
        if isinstance(x,(np.floating,float)): return float(x) if np.isfinite(x) else None
        if isinstance(x,(np.bool_,)): return bool(x)
        if isinstance(x,(pd.Timestamp,datetime)): return x.isoformat()
        return x
    tmp=path.with_name(path.name+f'.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def read_json(path, default=None):
    try: return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError): return {} if default is None else default


def atomic_csv(path, frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{threading.get_ident()}.tmp')
    frame.to_csv(tmp,index=False,compression='gzip' if str(path).endswith('.gz') else None)
    os.replace(tmp,path)


def safe_symbol(symbol):
    symbol=str(symbol).upper().strip()
    if not re.fullmatch(r'(?:\d{4,6}\.(?:TW|TWO)|\^TWII)',symbol):
        raise ValueError(f'Invalid Taiwan ticker: {symbol!r}')
    return symbol


def clean_prices(frame, ticker=None, source=None):
    if not isinstance(frame,pd.DataFrame) or frame.empty: return pd.DataFrame(columns=COLS)
    f=frame.copy()
    f=f.rename(columns={c:str(c).lower() for c in f.columns})
    f=f.rename(columns={'max':'high','min':'low','trading_volume':'volume','adj close':'adj_close'})
    if ticker is not None: f['ticker']=safe_symbol(ticker)
    needed={'date','ticker','open','high','low','close','volume'}
    if not needed.issubset(f): raise SourceError('price columns missing: '+str(needed-set(f)))
    f['date']=pd.to_datetime(f['date'],errors='coerce',utc=True).dt.tz_convert(None).dt.normalize()
    for c in ('open','high','low','close','volume','adj_close'):
        if c in f:
            f[c]=pd.to_numeric(f[c].astype(str).str.replace(',','',regex=False),errors='coerce')
    if 'adj_close' not in f: f['adj_close']=np.nan
    if source is not None: f['source']=source
    if 'source' not in f: f['source']='imported_unverified'
    mask=f[['date','open','high','low','close','volume']].notna().all(axis=1)
    mask &= (f[['open','high','low','close']]>0).all(axis=1) & (f.volume>=0)
    mask &= np.isfinite(f[['open','high','low','close','volume']]).all(axis=1)
    tol=1e-6
    mask &= f.high+tol >= f[['open','close','low']].max(axis=1)
    mask &= f.low-tol <= f[['open','close','high']].min(axis=1)
    rejected=int((~mask).sum()); f=f.loc[mask,COLS]
    conflict=f[f.duplicated(['ticker','date'],keep=False)]
    if not conflict.empty:
        check=conflict.groupby(['ticker','date'])[['open','high','low','close','volume']].nunique()
        if (check>1).any().any(): raise SourceError('conflicting duplicate prices')
    f=f.drop_duplicates(['ticker','date']).sort_values(['ticker','date']).reset_index(drop=True)
    f.attrs['rejected_rows']=rejected
    return f


def parse_finmind(payload,ticker):
    if int(payload.get('status',200)) != 200: raise SourceError(str(payload.get('msg','FinMind error')))
    rows=payload.get('data') or []
    code=safe_symbol(ticker).split('.')[0]
    if any(str(r.get('stock_id',code))!=code for r in rows): raise SourceError('FinMind symbol mismatch')
    return clean_prices(pd.DataFrame(rows),ticker,'FinMind_TaiwanStockPrice')


def parse_yahoo(payload,ticker):
    result=(payload.get('chart') or {}).get('result') or []
    if not result: raise SourceError('Yahoo empty chart')
    obj=result[0]
    if str((obj.get('meta') or {}).get('symbol','')).upper()!=safe_symbol(ticker):
        raise SourceError('Yahoo symbol mismatch')
    ts=obj.get('timestamp') or []
    quotes=(obj.get('indicators') or {}).get('quote') or []
    if not ts or not quotes: raise SourceError('Yahoo no daily bars')
    f=pd.DataFrame(quotes[0]);f['date']=pd.to_datetime(ts,unit='s',utc=True).tz_convert(TZ).strftime('%Y-%m-%d')
    adj=(obj.get('indicators') or {}).get('adjclose') or []
    if adj: f['adj_close']=adj[0].get('adjclose')
    return clean_prices(f,ticker,'Yahoo_daily')


def parse_twse(payload,ticker,month):
    if payload.get('stat') != 'OK': raise SourceError('TWSE no data: '+str(payload.get('stat')))
    if str(payload.get('date',''))[:6]!=str(month)[:6]: raise SourceError('TWSE returned different month')
    if ticker.split('.')[0] not in str(payload.get('title','')): raise SourceError('TWSE symbol mismatch')
    rows=[]
    for r in payload.get('data',[]):
        y,m,d=map(int,str(r[0]).split('/'));y=y+1911 if y<1911 else y
        rows.append(dict(date=f'{y:04d}-{m:02d}-{d:02d}',open=r[3],high=r[4],low=r[5],close=r[6],volume=r[1]))
    return clean_prices(pd.DataFrame(rows),ticker,'TWSE_monthly_raw')


class Client:
    def __init__(self,root,settings=Settings(),transport=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.cfg=settings;self.get=transport or requests.get
        self.token=os.getenv('FINMIND_TOKEN','').strip()
        self.lock=threading.Lock();self.calls=0;self.blocked={};self.events=[]
        self.budget_file=self.root/'request_history.json'
        self.history=read_json(self.budget_file,[])
        self.cooldown_file=self.root/'provider_cooldowns.json'
        self.cooldowns=read_json(self.cooldown_file)
        if not isinstance(self.history,list):self.history=[]
    def request(self,url,params=None,provider='other'):
        # Finite retries; a rate limit or entitlement error never spins/retries.
        key=provider+':'+str((params or {}).get('dataset',''))
        last=None
        for attempt in range(self.cfg.network_retries+1):
            with self.lock:
                now=time.time();self.history=[x for x in self.history if now-float(x.get('at',0))<3600]
                limit=560 if self.token else 280
                if self.calls>=self.cfg.request_budget:raise BudgetExceeded('job request budget reached; cached work retained')
                if provider in self.blocked:raise SourceError(provider+': '+self.blocked[provider])
                if key in self.blocked:raise SourceError(key+': '+self.blocked[key])
                until=float(self.cooldowns.get(provider,0))
                if until>now:raise BudgetExceeded(provider+': COOLDOWN_ACTIVE; resume after '+str(int(until-now))+' seconds')
                if provider=='FinMind' and sum(x.get('provider')=='FinMind' for x in self.history)>=limit:
                    raise BudgetExceeded('FinMind rolling hour budget reached; resume after quota resets')
                self.calls+=1;self.history.append({'at':now,'provider':provider})
                atomic_json(self.budget_file,self.history)
            headers={'User-Agent':'Mozilla/5.0 (AlphaRadar personal research)','Accept':'application/json'}
            if provider=='FinMind' and self.token:headers['Authorization']='Bearer '+self.token
            try:
                response=self.get(url,params=params,headers=headers,timeout=(4,self.cfg.timeout_seconds))
                code=int(response.status_code)
                if code>=500:
                    last=SourceError(provider+': HTTP_'+str(code))
                    if attempt<self.cfg.network_retries:
                        time.sleep(min(2**attempt,4));continue
                    raise last
                if code in (401,402,403,429):
                    with self.lock:
                        self.blocked[provider if code in (401,429) else key]='HTTP_'+str(code)
                        if code==429:
                            try:wait=float(getattr(response,'headers',{}).get('Retry-After',60))
                            except (ValueError,TypeError):wait=60.
                            self.cooldowns[provider]=time.time()+max(1,min(wait,3600))
                            atomic_json(self.cooldown_file,self.cooldowns)
                    raise SourceError(provider+' permission or rate limit: '+str(code))
                response.raise_for_status();body=response.json()
                status=int(body.get('status',200)) if isinstance(body,dict) else 200
                if provider=='FinMind' and status in (401,402,403,429):
                    with self.lock:
                        self.blocked[provider if status in (401,429) else key]='API_'+str(status)
                        if status==429:
                            self.cooldowns[provider]=time.time()+60
                            atomic_json(self.cooldown_file,self.cooldowns)
                    raise SourceError(provider+' permission or rate limit: API_'+str(status))
                if not isinstance(body,dict):raise SourceError(provider+': NON_OBJECT_RESPONSE')
                return body
            except requests.RequestException as e:
                last=SourceError(provider+': '+type(e).__name__)
                if 'NameResolution' in str(e) or 'Failed to resolve' in str(e):
                    with self.lock:self.blocked[provider]='DNS_UNAVAILABLE'
                    raise last from e
                if attempt<self.cfg.network_retries:
                    time.sleep(min(2**attempt,4));continue
                raise last from e
            except (ValueError,TypeError) as e:
                raise SourceError(provider+': INVALID_JSON_OR_STATUS') from e
        raise last or SourceError(provider+': REQUEST_FAILED')
    def finmind(self,dataset,code=None,start=None,end=None):
        p={'dataset':dataset}
        if code:p['data_id']=code
        if start:p['start_date']=start
        if end:p['end_date']=end
        return self.request(FINMIND,p,'FinMind')
    def universe(self):
        from v198_universe import balanced_sample, normalize_catalog, summarize_catalog
        cache=self.root/'universe.json'
        cached=read_json(cache)
        try:
            if (time.time()-float(cached.get('at',0))<86400 and cached.get('rows')):
                rows=normalize_catalog(cached['rows']);scope=cached.get('scope','CACHED_CURRENT_SNAPSHOT')
            else:
                body=self.finmind('TaiwanStockInfo')
                df=pd.DataFrame(body.get('data') or [])
                required={'stock_id','stock_name','type','date'}
                if not required.issubset(df):raise SourceError('universe schema missing')
                df['stock_id']=df.stock_id.astype(str)
                df['type']=df.type.astype(str).str.lower().str.strip()
                df=df.sort_values('date').drop_duplicates('stock_id',keep='last')
                df=df[df.type.isin(['twse','tpex']) & df.stock_id.str.fullmatch('[1-9][0-9]{3}')]
                rows=normalize_catalog([{'ticker':r.stock_id+('.TW' if r.type=='twse' else '.TWO'),
                       'name':r.stock_name, 'sector':str(getattr(r,'industry_category','UNKNOWN'))}
                       for r in df.itertuples()])
                if not rows:raise SourceError('empty equity universe')
                scope='LATEST_FINMIND_STOCK_INFO_NOT_HISTORICAL_MEMBERSHIP'
                atomic_json(cache,{'at':time.time(),'rows':rows,'scope':scope})
        except (SourceError,ValueError,KeyError) as e:
            self.events.append({'stage':'universe','error':str(e)})
            rows=normalize_catalog(cached.get('rows') or [])
            scope=cached.get('scope') or 'STALE_CATALOG_CURRENT_SNAPSHOT'
            if not rows:
                # A user-supplied, sourced catalog supports research when the
                # provider's current-stock discovery endpoint is unavailable.
                # Membership remains UNVERIFIED, never official point-in-time.
                supplied=self.root/'historical_universe.csv'
                if supplied.exists():
                    df=pd.read_csv(supplied,dtype=str).fillna('')
                    if not {'ticker','source'}.issubset(df):
                        raise SourceError('historical_universe.csv needs ticker and source')
                    sourced=df.loc[df.source.astype(str).str.strip().ne('')]
                    rows=normalize_catalog([{'ticker':r.get('ticker',''),
                        'name':r.get('name',''),'sector':r.get('sector','UNKNOWN'),
                        'listed_date':r.get('listed_date',''),
                        'delisted_date':r.get('delisted_date',''),
                        'member_source':r.get('source','')} for r in sourced.to_dict('records')])
                    if rows:scope='USER_SUPPLIED_SOURCED_CATALOG_UNVERIFIED_MEMBERSHIP'
                if not rows:
                    if self.cfg.universe_mode!='legacy_random':
                        raise SourceError('BROAD_UNIVERSE_CATALOG_UNAVAILABLE: '+str(e)+
                            '; connect FinMind or provide sourced historical_universe.csv / cached universe.json; no silent small-stock rescue') from e
                    rows=normalize_catalog([{'ticker':t,'name':n,'sector':'UNKNOWN'} for t,n in FALLBACK])
                    scope='EXPLICIT_LEGACY_54_STOCK_FALLBACK_NOT_REPRESENTATIVE'
        # User-supplied historical symbol coverage can supplement current survivors.
        # Such additions are clearly tagged, NOT called exchange-verified membership.
        history_file=self.root/'historical_universe.csv'
        if history_file.exists():
            data=pd.read_csv(history_file,dtype=str).fillna('')
            if not {'ticker','source'}.issubset(data):raise SourceError('historical_universe.csv needs ticker and source')
            extras=[]
            for r in data.to_dict('records'):
                if not str(r.get('source','')).strip():continue
                extras.append({'ticker':r.get('ticker',''),'name':r.get('name',''),
                    'sector':r.get('sector','UNKNOWN'), 'listed_date':r.get('listed_date',''),
                    'delisted_date':r.get('delisted_date',''), 'member_source':r['source']})
            prior={r['ticker']:r for r in rows}
            for r in normalize_catalog(extras):
                if r['ticker'] not in prior:prior[r['ticker']]=r
                else:
                    for k in ('listed_date','delisted_date','member_source'):
                        if r.get(k):prior[r['ticker']][k]=r[k]
            rows=sorted(prior.values(),key=lambda x:x['ticker'])
            scope+='; USER_HISTORICAL_SYMBOL_SUPPLEMENT_UNVERIFIED'
        selected=balanced_sample(rows,self.cfg.stocks,self.cfg.universe_seed,self.cfg.universe_mode)
        self.catalog_rows=rows
        self.universe_quality=summarize_catalog(rows,selected)
        self.universe_quality.update(mode=self.cfg.universe_mode,scope=scope,
            selection_is_historical_listing_membership=False,
            provenance='FinMind current stock-info snapshot and optional user-supplied symbols')
        atomic_json(self.root/'universe_quality.json',self.universe_quality)
        atomic_csv(self.root/'selected_universe.csv',pd.DataFrame(selected))
        atomic_csv(self.root/'universe_catalog.csv',pd.DataFrame(rows))
        return selected,scope
    def prices(self,ticker,start,end,official_fallback=False,force=False):
        ticker=safe_symbol(ticker); cache=self.root/'prices'/f'{ticker}.csv.gz'
        meta_path=cache.with_suffix('.json');meta=read_json(meta_path)
        try:old=clean_prices(pd.read_csv(cache)) if cache.exists() else pd.DataFrame(columns=COLS)
        except (ValueError,OSError,SourceError):old=pd.DataFrame(columns=COLS)
        if (not force and meta.get('requested_start','9999')<=start and meta.get('requested_end','')>=end
                and len(old)>0
                and (not self.cfg.prefer_adjusted or meta.get('adjustment_policy')=='v191_prefer_adjusted')):
            return old[(old.date>=start)&(old.date<=end)].copy(),dict(meta,cached=True,state='CACHED_RANGE')
        # Request the full <=3y interval: one call per stock, never one call per day.
        errors=[];new=pd.DataFrame()
        def yahoo():
            p1=int(pd.Timestamp(start,tz=TZ).timestamp())
            p2=int((pd.Timestamp(end,tz=TZ)+pd.Timedelta(days=1)).timestamp())
            return parse_yahoo(self.request(YAHOO+ticker,{'period1':p1,'period2':p2,
                    'interval':'1d','events':'div,splits'},'Yahoo'),ticker)
        # One internally consistent series. Never stitch a provider's adjusted
        # prices into another provider's raw OHLC. Access failures are disclosed.
        if self.cfg.prefer_adjusted:
            try:new=yahoo()
            except (SourceError,ValueError,KeyError) as e:errors.append(str(e))
        if new.empty:
            try:new=parse_finmind(self.finmind('TaiwanStockPrice',ticker.split('.')[0],start,end),ticker)
            except (SourceError,ValueError,KeyError) as e:errors.append(str(e))
        if new.empty and not self.cfg.prefer_adjusted:
            try:new=yahoo()
            except (SourceError,ValueError,KeyError) as e:errors.append(str(e))
        # Official monthly source is bounded and not hammered for an entire market.
        if new.empty and official_fallback and ticker.endswith('.TW'):
            months=pd.date_range(pd.Timestamp(start).replace(day=1),pd.Timestamp(end).replace(day=1),freq='MS')
            parts=[]
            for month in months:
                try:
                    stamp=month.strftime('%Y%m%d')
                    month_cache=self.root/'monthly_raw'/ticker/(stamp[:6]+'.json')
                    payload=read_json(month_cache) if not force else {}
                    if not payload:
                        payload=self.request(TWSE,{'date':stamp,'stockNo':ticker.split('.')[0],'response':'json'},'TWSE')
                    part=parse_twse(payload,ticker,stamp)
                    atomic_json(month_cache,payload)
                    parts.append(part)
                    time.sleep(.35)
                except (SourceError,ValueError,KeyError) as e:
                    errors.append(str(e))
                    if 'DNS_' in str(e) or isinstance(e,BudgetExceeded):break
            if parts:new=pd.concat(parts,ignore_index=True)
        if new.empty:
            # Do not say a failed refresh is current. Preserve last usable prices.
            result=old[(old.date>=start)&(old.date<=end)].copy()
            return result,{'ticker':ticker,'state':'STALE_CACHE' if len(result) else 'UNAVAILABLE','errors':errors,
                           'rows':len(result),'requested_start':start,'requested_end':end}
        new=new[(new.date>=start)&(new.date<=end)].copy()
        if new.empty:raise SourceError(ticker+': provider returned outside requested range')
        # Prefer a consistent provider series; do not mix Yahoo-adjusted with raw
        # points from another provider inside one forward-return calculation.
        combined=new.sort_values('date').drop_duplicates('date',keep='last')
        atomic_csv(cache,combined)
        meta={'ticker':ticker,'state':'FETCHED','at':time.time(),'requested_start':start,'requested_end':end,
              'first':str(combined.date.min().date()),'last':str(combined.date.max().date()),'rows':len(combined),
              'source':str(combined.source.iloc[-1]),'adjustment_policy':'v191_prefer_adjusted' if self.cfg.prefer_adjusted else 'raw_first',
              'adjusted_fraction':float(combined.adj_close.notna().mean()),
              'errors':errors,'sha256':hashlib.sha256(cache.read_bytes()).hexdigest()}
        atomic_json(meta_path,meta)
        return combined,meta
    def evidence(self,ticker,start,end):
        """Daily institutional flows and published revenue; optional, separately cached.

        Monthly vendor dates are period markers, not publication dates.
        Unknown disclosure times are cached but not scored in strict mode.
        Financial history download is independent of the selected model engine.
        """
        cache=self.root/'evidence'/f'{ticker}.csv.gz';code=ticker.split('.')[0]
        out=[]
        for dataset in ('TaiwanStockInstitutionalInvestorsBuySell','TaiwanStockMonthRevenue','TaiwanStockMarginPurchaseShortSale'):
            if dataset.startswith('TaiwanStockMonth'):
                begin=(pd.Timestamp(start)-pd.DateOffset(months=15)).date().isoformat()
            else:begin=start
            try:
                cache_key=hashlib.sha256(f'{dataset}|{code}|{begin}|{end}'.encode()).hexdigest()[:24]
                dataset_cache=self.root/'evidence_raw'/f'{cache_key}.json'
                saved=read_json(dataset_cache)
                usable=(not self.cfg.refresh_history and saved.get('complete') and
                        (saved.get('payload',{}).get('data') or time.time()-float(saved.get('at',0))<3600))
                if usable:
                    obj=saved['payload']
                else:
                    obj=self.finmind(dataset,code,begin,end)
                    if int(obj.get('status',200))!=200:raise SourceError(dataset+': API_FAILED')
                    atomic_json(dataset_cache,{'complete':True,'dataset':dataset,'ticker':ticker,
                        'start':begin,'end':end,'at':time.time(),'payload':obj})
                df=pd.DataFrame(obj.get('data') or [])
                if df.empty:continue
                if dataset=='TaiwanStockInstitutionalInvestorsBuySell':
                    keys={'Foreign_Investor':'foreign_net','Investment_Trust':'trust_net','Dealer_self':'dealer_net'}
                    for day,g in df.groupby('date'):
                        r={'ticker':ticker,'date':day,'available_at':str(pd.Timestamp(day)+pd.Timedelta(days=1)),
                           'source':dataset,'timing':'next_day_after_vendor_date'}
                        for name,key in keys.items():
                            q=g[g['name']==name]
                            if len(q):r[key]=float(pd.to_numeric(q['buy'],errors='coerce').sum()-pd.to_numeric(q['sell'],errors='coerce').sum())
                        out.append(r)
                elif dataset=='TaiwanStockMonthRevenue':
                    from v1961_evidence import monthly_revenue_records
                    monthly = monthly_revenue_records(ticker, df, self.cfg)
                    if not monthly.empty:
                        out.extend(monthly.to_dict('records'))
                        omitted = int(monthly.available_at.isna().sum())
                        if omitted:
                            self.events.append({'stage':dataset,'ticker':ticker,
                                'reason':'REVENUE_DISCLOSURE_TIME_MISSING_CACHED_NOT_SCORED',
                                'omitted_periods':omitted})
                else:
                    col='MarginPurchaseTodayBalance'
                    if col not in df:continue
                    for r in df.to_dict('records'):
                        out.append({'ticker':ticker,'date':r['date'],'available_at':str(pd.Timestamp(r['date'])+pd.Timedelta(days=1)),
                                    'margin_balance':r[col],'source':dataset,'timing':'next_day_after_vendor_date'})
            except (SourceError,ValueError,KeyError,TypeError) as e:
                self.events.append({'stage':dataset,'ticker':ticker,'error':str(e)})
        result=pd.DataFrame(out)
        if self.cfg.optional_data and self.cfg.financial_history:
            from v194_evidence import additional_evidence
            finance=additional_evidence(self,ticker,start,end)
            if not finance.empty:result=pd.concat([result,finance],ignore_index=True)
        if not result.empty:atomic_csv(cache,result)
        return result


def import_legacy_sqlite(path,start,end):
    path=Path(path)
    if not path.exists():return pd.DataFrame(columns=COLS)
    with sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True) as conn:
        df=pd.read_sql_query('SELECT ticker,date,open,high,low,close,volume FROM daily_prices WHERE date>=? AND date<=?',conn,params=(start,end))
    return clean_prices(df,source='V18_cache_source_unverified')


def download_history(root,cfg=Settings(),progress=None,transport=None,tickers=None):
    root=Path(root);client=Client(root,cfg,transport)
    if tickers is None:universe,scope=client.universe()
    else:
        universe=[{'ticker':safe_symbol(t),'name':safe_symbol(t),'sector':'user_pool'} for t in dict.fromkeys(tickers)]
        scope='USER_FIXED_UNIVERSE_NOT_HISTORICAL_MEMBERSHIP'
    todo=list(universe)
    if cfg.benchmark not in {r['ticker'] for r in todo}:todo.append({'ticker':cfg.benchmark,'name':'Benchmark 0050','sector':'benchmark'})
    frames=[];metas=[]
    # Small parallelism and a shared rolling request budget, not an API flood.
    with ThreadPoolExecutor(max_workers=cfg.download_workers) as pool:
        futures={pool.submit(client.prices,r['ticker'],cfg.start(),cfg.end(),i<2 or r['ticker']==cfg.benchmark,cfg.refresh_history):r for i,r in enumerate(todo)}
        for i,future in enumerate(as_completed(futures),1):
            item=futures[future]
            try:
                frame,meta=future.result()
                if not frame.empty:
                    frame['name']=item['name'];frame['sector']=item['sector']
                    # Carry sourced historical listing metadata into both the
                    # saved price snapshot and the factor eligibility filter.
                    for field in ('listed_date','delisted_date','member_source'):
                        if item.get(field):frame[field]=item[field]
                    frames.append(frame)
                metas.append(meta)
            except Exception as e:metas.append({'ticker':item['ticker'],'state':'UNAVAILABLE','errors':[type(e).__name__+': '+str(e)]})
            if progress:progress('download',i,len(todo),item['ticker'])
    prices=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    evidence=[]
    if cfg.optional_data:
        actual_equities=set(prices.ticker) if not prices.empty else set()
        evidence_universe=[r for r in universe if r['ticker'] in actual_equities]
        for i,r in enumerate(evidence_universe,1):
            frame=client.evidence(r['ticker'],cfg.start(),cfg.end())
            if not frame.empty:evidence.append(frame)
            if progress:progress('optional',i,len(evidence_universe),r['ticker'])
    ev=pd.concat(evidence,ignore_index=True) if evidence else pd.DataFrame()
    extras=root/'optional_evidence.csv'
    if cfg.optional_data and extras.exists():
        imported=pd.read_csv(extras)
        if not {'ticker','available_at','source'}.issubset(imported):raise SourceError('optional CSV needs ticker, available_at, source')
        ev=pd.concat([ev,imported],ignore_index=True)
    manifest={'scope':scope,'pool':universe,'requested_start':cfg.start(),'requested_end':cfg.end(),
              'requests':client.calls,'sources':metas,'optional_errors':client.events,
              'historical_membership_verified':False,'data_rows':len(prices),
              'requested_equities':cfg.stocks,'pool_equities':len(universe),'universe_mode':cfg.universe_mode,
              'catalog_equities':len(getattr(client,'catalog_rows',universe)),
              'universe_seed':cfg.universe_seed,'cutoff_seed':cfg.seed,
              'optional_enabled':cfg.optional_data,'blocked_sources':client.blocked,
              'ready_symbols':int(prices.ticker.nunique()) if not prices.empty else 0,
              'ready_equities':int(prices.loc[prices.ticker!=cfg.benchmark,'ticker'].nunique()) if not prices.empty else 0,
              'download_completeness':float(prices.loc[prices.ticker!=cfg.benchmark,'ticker'].nunique()/len(universe)) if len(universe) and not prices.empty else 0.0,
              'requested_target_coverage':float(prices.loc[prices.ticker!=cfg.benchmark,'ticker'].nunique()/cfg.stocks) if cfg.stocks and not prices.empty else 0.0,
              'catalog_target_coverage':float(len(universe)/cfg.stocks) if cfg.stocks else 0.0,
              'universe_quality':getattr(client,'universe_quality',None),
              'unavailable_symbols':sum(m.get('state')=='UNAVAILABLE' for m in metas),'created_at':datetime.now(TZ).isoformat()}
    from v198_universe import summarize_catalog
    quality=summarize_catalog(getattr(client,'catalog_rows',universe),universe,
        set(prices.ticker) if not prices.empty else set())
    from v198_universe import history_depth_report
    quality.update(history_depth_report(prices,universe,cfg.warmup))
    quality.update(mode=cfg.universe_mode,scope=scope)
    manifest['universe_quality']=quality
    atomic_json(root/'universe_quality.json',quality)
    atomic_json(root/'download_manifest.json',manifest)
    if not prices.empty:atomic_csv(root/'history_prices.csv.gz',prices)
    if not ev.empty:atomic_csv(root/'historical_evidence.csv.gz',ev)
    if not prices.empty:
        from v192_history import export_database
        if progress:progress('database',0,1,'Saving portable historical database')
        export_database(root,prices,ev,manifest,cfg)
        if progress:progress('database',1,1,'Historical database saved')
    return prices,ev,manifest
