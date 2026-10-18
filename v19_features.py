"""Causal factor engine shared by live daily ranking and historical replay.

Every rolling window ends at the signal date. W pivots require two right bars.
Unpublished optional evidence is never backfilled. All factors are 0--100
research scores, NOT probabilities. Tradeability is a separate decision.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from v19_config import FACTORS, TECH, OPTIONAL, Settings, prior, TZ
from v19_data import clean_prices


def clipped(x):return x.clip(0.,100.) if isinstance(x,pd.Series) else np.clip(x,0.,100.)

def rsi(close,period=14):
    change=close.diff();gain=change.clip(lower=0).ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    loss=(-change.clip(upper=0)).ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    result=100-100/(1+gain/loss.replace(0,np.nan))
    result=result.mask((loss==0)&(gain>0),100).mask((loss==0)&(gain==0),50)
    return result


def w_patterns(low,high,close):
    """Two confirmed distinct troughs, intervening neckline; no future extrema."""
    lo=np.asarray(low,dtype=float);hi=np.asarray(high,dtype=float);c=np.asarray(close,dtype=float)
    n=len(c);score=np.full(n,np.nan);neck=np.full(n,np.nan);stages=['INSUFFICIENT']*n
    pivots=[]
    for t in range(n):
        j=t-2
        if j>=2 and np.isfinite(lo[j-2:t+1]).all() and lo[j]<=min(lo[j-2:j]) and lo[j]<min(lo[j+1:t+1]):
            pivots.append(j)
        if t<39:continue
        score[t]=30.;stages[t]='NONE'
        current=[i for i in pivots if t-100<=i<=t-2]
        second=[i for i in current if t-i<=30]
        match=None
        for b in reversed(second):
            for a in reversed(current):
                if not 7<=b-a<=62:continue
                base=min(lo[a],lo[b])
                if base<=0 or abs(lo[b]/lo[a]-1)>.065:continue
                nr=float(np.max(hi[a+1:b]));height=nr/base-1
                if not .045<=height<=.5:continue
                if np.min(lo[b+1:t+1])<base*.97:continue
                after=np.arange(b+2,t+1)
                crosses=[int(k) for k in after if c[k]>nr*1.006 and c[k-1]<=nr*1.006]
                if crosses and t-crosses[-1]<=14:
                    k=crosses[-1]
                    if c[t]<nr*.975:state,val='FAILED',10.
                    elif t-k>=2 and np.min(lo[k+1:t+1])<=nr*1.025 and c[t]>=nr:state,val='RETEST',90.
                    else:state,val='BREAKOUT',82.
                elif base*1.025<=c[t]<nr*1.006:
                    state,val='FORMING',55.+20.*np.clip((c[t]/base-1)/height,0,1)
                else:continue
                match=(val,nr,state);break
            if match:break
        if match:score[t],neck[t],stages[t]=match
    return score,neck,stages


def technical_frame(prices):
    f=prices.sort_values('date').copy().reset_index(drop=True)
    # Preserve physical share volume separately from the normalized factor.
    f['share_volume']=f['volume'].astype(float)
    # Chain causal returns. Missing adjusted quotes must not reset the price
    # coordinate to 1 and fabricate a jump at a source-coverage boundary.
    good=f.adj_close.notna() & (f.adj_close>0)
    paired=good & good.shift(1,fill_value=False)
    growth=(f.close/f.close.shift(1)).where(~paired,f.adj_close/f.adj_close.shift(1))
    growth.iloc[0]=1.
    c=float(f.close.iloc[0])*growth.cumprod()
    adjustment=c/f.close
    o=f.open*adjustment;hi=f.high*adjustment;lo=f.low*adjustment;v=f.share_volume
    f['adjustment_transition']=good.ne(good.shift(1,fill_value=bool(good.iloc[0])))
    f['adjusted_available']=good
    f['calc_close']=c;f['calc_open']=o;f['calc_high']=hi;f['calc_low']=lo
    f['ma20']=c.rolling(20,min_periods=20).mean();ma60=c.rolling(60,min_periods=60).mean()
    ma5=c.rolling(5,min_periods=5).mean();f['ma60']=ma60
    true_range=pd.concat([hi-lo,(hi-c.shift()).abs(),(lo-c.shift()).abs()],axis=1).max(axis=1)
    f['atr']=true_range.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    f['atr_pct']=100*f.atr/c
    f['rsi']=rsi(c);rsv=100*(c-lo.rolling(9).min())/(hi.rolling(9).max()-lo.rolling(9).min()).replace(0,np.nan)
    f['k']=rsv.ewm(alpha=1/3,adjust=False,min_periods=3).mean();f['d']=f.k.ewm(alpha=1/3,adjust=False,min_periods=3).mean()
    f['kd_cross']=((f.k>f.d)&(f.k.shift()<=f.d.shift())).astype(float)
    dif=c.ewm(span=12,adjust=False,min_periods=12).mean()-c.ewm(span=26,adjust=False,min_periods=26).mean()
    hist=dif-dif.ewm(span=9,adjust=False,min_periods=9).mean()
    f['macd']=dif;f['macd_hist']=hist;f['macd_cross']=((hist>0)&(hist.shift()<=0)).astype(float)
    f['bias20']=100*(c/f.ma20-1)
    f['ret5']=c.pct_change(5,fill_method=None);f['ret20']=c.pct_change(20,fill_method=None)
    f['ret60']=c.pct_change(60,fill_method=None);f['ret1']=c.pct_change(fill_method=None)
    f['volume_ratio']=v/v.shift().rolling(20,min_periods=20).mean().replace(0,np.nan)
    f['turnover20']=(f.close*v).rolling(20,min_periods=20).mean()
    prior_high=hi.shift().rolling(20,min_periods=20).max();f['prior_high20']=prior_high
    position=(c-lo)/(hi-lo).replace(0,np.nan);position=position.fillna(.5)
    f['close_position']=position
    f['trend']=clipped(50+220*(c/f.ma20-1)+250*(f.ma20/ma60-1)+220*f.ma20.pct_change(5,fill_method=None))
    distance=c/prior_high-1
    f['breakout']=clipped(48+500*distance+12*np.clip(f.volume_ratio-1,-1,2)+18*(position-.5))
    low_rsi=f.rsi.rolling(7).min();rsi_recovery=clipped((f.rsi-low_rsi)*4)*((low_rsi<38)&(f.rsi>low_rsi))
    kd_low=(f.k.shift().rolling(5).min()<35)&(f.k>f.d)
    bias_repair=(f.bias20<1)&(f.bias20>-20)&(f.bias20>f.bias20.shift())
    falling=(c<lo.shift().rolling(10).min())|(f.bias20<-20)
    f['reversal']=clipped(28+0.3*rsi_recovery+18*kd_low+16*bias_repair+12*(hist>hist.shift())-25*falling)
    dry=(v.rolling(5).mean()/v.shift(5).rolling(20).mean().replace(0,np.nan)<.7)
    stable=(c>=c.shift(5)*.98)&(c>c.shift())
    f['volume']=clipped(42+18*(dry&stable)+16*((f.volume_ratio>1.2)&(position>.6)&(c>c.shift()))
                         +16*(position-.5)-25*((f.volume_ratio>2)&(position<.3)))
    f['pattern'],f['neckline'],f['pattern_stage']=w_patterns(lo,hi,c)
    f['bars']=np.arange(len(f))+1
    f['support10']=lo.shift().rolling(10).min()
    # Prices and support displayed in actual current-price units.
    for col in ('ma20','ma60','atr','support10','neckline','prior_high20'):
        f[col]=f[col]/adjustment
    discontinuity=(f.close.pct_change(fill_method=None).abs()>.25)&(~paired)
    f['raw_discontinuity']=discontinuity
    f['recent_discontinuity']=discontinuity.rolling(60,min_periods=1).max().astype(bool)
    return f


def attach_optional(frame,evidence,cfg=None):
    f=frame.sort_values('date').copy()
    for k in OPTIONAL:f[k]=np.nan
    for k in ('revenue_yoy','eps_growth','eps_ttm','gross_margin','foreign_net','trust_net','dealer_net','margin_balance','holder_change','pe_ratio','cash_conversion_ratio','debt_ratio'):
        f[k]=np.nan
    if evidence is None or evidence.empty:return f
    from v1961_evidence import sanitize_revenue_evidence
    if 'ticker' not in evidence:
        return f
    local_evidence=evidence[evidence.ticker.astype(str).eq(str(f.ticker.iloc[0]))]
    ev=sanitize_revenue_evidence(local_evidence, cfg).copy()
    if cfg is not None and cfg.financial_timing=='disclosure_required' and 'timing' in ev:
        ev=ev[~ev.timing.fillna('').astype(str).str.startswith('ASSUMED_')].copy()
    if 'source' in ev:
        ev=ev[ev.source.notna() & ev.source.astype(str).str.strip().ne('')]

    if not {'ticker','available_at','source'}.issubset(ev):return f
    ticker=str(f.ticker.iloc[0]);ev=ev[ev.ticker.astype(str)==ticker]
    if ev.empty:return f
    def local_time(value):
        try:
            stamp=pd.Timestamp(value)
            if pd.isna(stamp):return pd.NaT
            stamp=stamp.tz_localize(TZ) if stamp.tzinfo is None else stamp.tz_convert(TZ)
            return stamp.tz_localize(None)
        except (ValueError,TypeError):return pd.NaT
    ev['available_at']=ev.available_at.map(local_time)
    # End-of-day features may use evidence released up to 23:59 that date.
    f['_decision']=f.date+pd.Timedelta(hours=23,minutes=59)
    for col in ('revenue_yoy','eps_growth','eps_ttm','gross_margin','foreign_net','trust_net','dealer_net','margin_balance','holder_change','pe_ratio','cash_conversion_ratio','debt_ratio'):
        if col not in ev:continue
        good=ev.dropna(subset=['available_at',col]).sort_values('available_at').drop_duplicates('available_at',keep='last')
        if good.empty:continue
        good[col]=pd.to_numeric(good[col],errors='coerce')
        days=100 if col in ('revenue_yoy',) else 400 if col=='cash_conversion_ratio' else 190 if col in ('eps_growth','eps_ttm','gross_margin','debt_ratio') else 16 if col=='holder_change' else 10 if col=='pe_ratio' else 4
        joined=pd.merge_asof(f[['_decision']].sort_values('_decision'),good[['available_at',col]],
                             left_on='_decision',right_on='available_at',direction='backward',tolerance=pd.Timedelta(days=days))
        values=joined[col].to_numpy()
        if col.endswith('_net'):
            # One reported session must not be copied into three consecutive
            # sessions. Repeated publication timestamps are missing, NOT zero.
            repeated=joined.available_at.duplicated()&joined.available_at.notna()
            values[repeated.to_numpy()]=np.nan
        f[col]=values
    yy=f.revenue_yoy; accel=yy-yy.shift(21)
    f['revenue']=clipped(50+yy*.65+accel.fillna(0)*.3).where(yy.notna())
    q=pd.concat([clipped(50+f.eps_growth*.5),clipped(f.gross_margin*1.6)],axis=1)
    f['quality']=q.mean(axis=1).where(q.notna().any(axis=1))
    denom=f.share_volume.shift().rolling(5,min_periods=5).sum().replace(0,np.nan)
    for col,key in [('foreign_net','foreign'),('trust_net','trust'),('dealer_net','dealer')]:
        s=f[col];flow=s.rolling(5,min_periods=5).sum()/denom
        continuity=(s>0).astype(float).where(s.notna()).rolling(3,min_periods=3).sum()
        f[key]=clipped(45+450*flow+5*continuity).where(flow.notna()&continuity.notna())
    joint=((f.foreign_net>0)&(f.trust_net>0)).astype(float).where(f[['foreign_net','trust_net']].notna().all(axis=1))
    f['joint']=clipped(30+20*joint.rolling(3,min_periods=3).sum())
    delta=f.margin_balance.pct_change(5,fill_method=None)
    f['margin']=clipped(50-200*delta+12*((f.ret5>=-.02)&(f.ret1>0))).where(delta.notna())
    f['holders']=clipped(50+f.holder_change*12).where(f.holder_change.notna())
    return f.drop(columns=['_decision'])


def build_panel(prices,evidence=None,cfg=Settings()):
    if prices is None or prices.empty:raise ValueError('No price data')
    # Keep descriptive metadata, but never score the current company name or
    # sector labels as if they were original historical evidence.
    meta=prices.drop_duplicates('ticker').set_index('ticker')
    raw=clean_prices(prices)
    raw=raw[(raw.date>=cfg.start())&(raw.date<=cfg.end())]
    parts=[]
    for ticker,g in raw.groupby('ticker',sort=True):
        if len(g)<20:continue
        out=technical_frame(g)
        for col in ('name','sector'):
            out[col]=str(meta.loc[ticker,col]) if col in meta else ticker
        for col in ('listed_date','delisted_date'):
            if col in meta:
                out[col]=meta.loc[ticker,col]
        if 'member_source' in meta:out['member_source']=str(meta.loc[ticker,'member_source'])
        out=attach_optional(out,evidence if cfg.optional_data else None,cfg)
        parts.append(out)
    if not parts:raise ValueError('Insufficient valid price history')
    panel=pd.concat(parts,ignore_index=True).sort_values(['date','ticker']).reset_index(drop=True)
    # Missing benchmark rows must not erase every stock's relative strength.
    # The cross-sectional fallback is a disclosed feature proxy, not an ETF quote.
    dense=panel.groupby('date').ticker.nunique()
    cal=pd.DatetimeIndex(dense[dense>=max(2,int(panel.ticker.nunique()*.5))].index)
    b=panel[panel.ticker==cfg.benchmark][['date','calc_close','ma20','ma60','close','recent_discontinuity']].drop_duplicates('date').set_index('date')
    market=panel[panel.ticker!=cfg.benchmark].groupby('date').ret20.median()
    fallback=panel.date.map(market)
    if b.empty:
        panel['benchmark_kind']='CROSS_SECTION_MEDIAN_NOT_INDEX'
        bm=fallback
        panel['regime']='UNKNOWN'
    else:
        bc=b.calc_close.reindex(cal.union(b.index).sort_values())
        bret=bc/bc.shift(20)-1  # Never forward-fill a suspended/missing ETF bar.
        bret=bret.mask(b.recent_discontinuity.reindex(bret.index).astype('boolean').fillna(True).astype(bool))
        actual=panel.date.map(bret)
        bm=actual.fillna(fallback)
        panel['benchmark_kind']=np.where(actual.notna(),cfg.benchmark,'CROSS_SECTION_MEDIAN_NOT_INDEX')
        regime=pd.Series(np.where((b.close>=b.ma20)&(b.ma20>=b.ma60),'BULL',np.where(b.close>=b.ma60,'NEUTRAL','BEAR')),index=b.index)
        regime=regime.mask(b.recent_discontinuity,'UNKNOWN')
        panel['regime']=panel.date.map(regime).fillna('UNKNOWN')
    panel['relative_strength']=clipped(50+200*(panel.ret20-bm)+100*panel.ret5)
    panel['eligible']=(panel.bars>=cfg.warmup)&(panel.close>=5)&(panel.turnover20>=cfg.min_turnover)&(panel.share_volume>0)
    panel['eligible'] &= (~panel.recent_discontinuity)&panel[list(TECH)].notna().all(axis=1)&(panel.ticker!=cfg.benchmark)
    panel['eligible'] &= panel.bias20<25
    for col,comparator in (('listed_date','after'),('delisted_date','before')):
        if col in panel:
            bound=pd.to_datetime(panel[col].replace('',pd.NaT),errors='coerce')
            known=bound.notna()
            panel['eligible'] &= (~known) | ((panel.date>=bound) if comparator=='after' else (panel.date<=bound))
    from v193_factors import extend_panel
    panel=extend_panel(panel,cfg)
    from v194_factors import add_atomic
    panel=add_atomic(panel,cfg)
    from v196_factors import add_scorecards
    panel=add_scorecards(panel,cfg)
    from v197_ten_pillars import add_pillars
    panel=add_pillars(panel,cfg)
    return panel


def score_rows(frame,weights,min_coverage=.75):
    keys=[k for k in weights if k in FACTORS and weights[k]>0]
    if not keys:return pd.Series(np.nan,index=frame.index),pd.Series(0.,index=frame.index)
    w=np.array([weights[k] for k in keys]);w=w/w.sum()
    a=frame.reindex(columns=keys).to_numpy(dtype=float)
    mask=np.isfinite(a);coverage=mask@w
    score=np.divide(np.nan_to_num(a)@w,coverage,out=np.full(len(frame),np.nan),where=coverage>0)
    score[coverage<min_coverage]=np.nan
    # Overheat is a risk penalty, not another duplicated momentum factor.
    if 'bias20' in frame and not any(str(k).startswith(('r196_','p197_')) for k in keys):score-=np.clip(frame.bias20.to_numpy()-12,0,15)*.7
    return pd.Series(score,index=frame.index),pd.Series(coverage,index=frame.index)


def rank_date(panel,date,weights,cfg=Settings(),top_k=None):
    rows=panel[(panel.date==pd.Timestamp(date))&panel.eligible].copy()
    if rows.empty:return rows
    rows['score'],rows['factor_coverage']=score_rows(rows,weights,min_coverage=1.-1e-9 if cfg.factor_design in ('sparse_v194','sparse_v195','tournament_v196','ten_angle_v197','balanced_v198') else .75)
    rows=rows.dropna(subset=['score']).sort_values(['score','ticker'],ascending=[False,True])
    return rows.head(top_k or cfg.top_k)


def current_candidates(panel,cfg=Settings(),weights=None,horizon=5,setup=None,limit=20):
    date=panel.date.max();weights=weights or prior(horizon)
    rows=rank_date(panel,date,weights,cfg,top_k=max(1,panel.ticker.nunique()))
    if rows.empty:return rows
    rows['setup']=np.where((rows.reversal>65)&(rows.bias20<5),'REVERSAL',np.where(rows.breakout>65,'BREAKOUT','WATCH'))
    good_business=(rows.revenue_yoy>0)&(rows.eps_ttm>0)&(rows.gross_margin>0)
    rows.loc[good_business&(rows.bias20<0)&(rows.reversal>55),'setup']='GROWTH_PULLBACK'
    rows['stop_reference']=np.maximum(rows.support10,rows.close-1.8*rows.atr)
    rows['stop_reference']=np.minimum(rows.stop_reference,rows.close*.99)
    rows['stop_distance_pct']=100*(rows.close/rows.close-rows.stop_reference/rows.close)
    rows['reference_low']=rows.close-.5*rows.atr;rows['reference_high']=rows.close+.25*rows.atr
    rows['chase_ceiling']=rows.close+np.minimum(rows.atr,rows.close*.03)
    rows['is_buy_signal']=False
    if setup: rows=rows[rows.setup==setup]
    return rows.head(limit)
