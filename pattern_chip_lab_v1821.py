"""V18.21 shadow research: completed candles, chart patterns, monthly growth, chips.

No claim that these are fitted or calibrated probabilities. Nothing in this file
can produce an execution BUY. Keep full-universe inputs and PIT distinct from
richer display-only evidence. All dates are explicit to prevent lookahead.
"""
from __future__ import annotations

from datetime import date
from typing import Mapping
import math
import numpy as np
import pandas as pd

VERSION = 'v18.21-pattern-chip-experimental'
REVERSAL_WEIGHTS = {
    'w_bottom': .14, 'kd_turn': .11, 'macd_turn': .11, 'rsi_rebound': .09,
    'dryup': .10, 'bias_reversion': .08, 'volume_break': .07,
    'revenue_growth': .10, 'chips': .20,
}
BREAKOUT_WEIGHTS = {
    'w_bottom': .12, 'kd_turn': .06, 'macd_turn': .14, 'rsi_rebound': .04,
    'volume_break': .12, 'breakout': .14, 'relative_accel': .08,
    'revenue_growth': .10, 'chips': .20,
}
CHIP_WEIGHTS = {
    'trust_streak': .20, 'foreign_streak': .14, 'foreign_trust_sync': .16,
    'today_net': .12, 'dealer_self': .08, 'margin_clean': .11,
    'large_holder': .11, 'borrow_down': .04, 'branch_proxy': .04,
}
for _weights in (REVERSAL_WEIGHTS, BREAKOUT_WEIGHTS, CHIP_WEIGHTS):
    assert abs(sum(_weights.values()) - 1.0) < 1e-10


def _num(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None


def _clip(x):
    return round(float(np.clip(x, 0, 100)),1)


def _part(score, why, date_label='', source=''):
    return {'score':_clip(score) if score is not None else None,
            'why':str(why), 'date':str(date_label or ''), 'source':str(source or '')}


def _score(factors:Mapping, weights:Mapping, *, min_coverage=0.0):
    coverage=0.0; total=0.0
    for name,wt in weights.items():
        val=_num((factors.get(name) or {}).get('score'))
        if val is not None:
            coverage += wt
            total += wt*val
        else:
            total += wt*50.0
    # Missing items never renormalize all remaining items up to 100.
    total -= 12.0*(1.0-coverage)
    return {'score':_clip(total) if coverage >= min_coverage else None,
            'evidence_coverage':round(coverage,3), 'provisional':True}


def _rsi_wilder(close:pd.Series, periods:int=14) -> pd.Series:
    delta=close.diff()
    up=delta.clip(lower=0).ewm(alpha=1/periods,adjust=False,min_periods=periods).mean()
    down=(-delta.clip(upper=0)).ewm(alpha=1/periods,adjust=False,min_periods=periods).mean()
    rs=up/down.replace(0,np.nan)
    out=100-100/(1+rs)
    out=out.where(down>0, 100.0).where((up+down)>0,50.0)
    return out


def _stoch_kd(high:pd.Series, low:pd.Series, close:pd.Series):
    ll=low.rolling(9,min_periods=9).min()
    hh=high.rolling(9,min_periods=9).max()
    rsv=((close-ll)/(hh-ll).replace(0,np.nan)*100).clip(0,100).fillna(50)
    k=rsv.ewm(alpha=1/3,adjust=False).mean()
    d=k.ewm(alpha=1/3,adjust=False).mean()
    return k,d


def w_bottom_pattern(df:pd.DataFrame) -> dict:
    """Confirmed pivots need two bars to the right *by this cutoff*.

    Search the latest 100 sessions. The neckline is between the two lows,
    never based on bars after the second low. No future candles are accessed.
    """
    if not isinstance(df,pd.DataFrame) or len(df)<40 or not {'Low','High','Close','Volume'}.issubset(df.columns):
        return {'stage':'INSUFFICIENT','score':None,'reason':'W底至少需要40根已完成日K'}
    frame=df.tail(110)
    lo=pd.to_numeric(frame['Low'],errors='coerce').to_numpy(dtype=float)
    hi=pd.to_numeric(frame['High'],errors='coerce').to_numpy(dtype=float)
    close=pd.to_numeric(frame['Close'],errors='coerce').to_numpy(dtype=float)
    volume=pd.to_numeric(frame['Volume'],errors='coerce').to_numpy(dtype=float)
    n=len(lo)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all() and np.isfinite(close).all() and np.isfinite(volume).all()):
        return {'stage':'INSUFFICIENT','score':None,'reason':'W底價格或量能缺漏'}
    # Confirmation at t requires the two bars *after* the potential low.
    pivots=[i for i in range(2,n-2) if lo[i] <= np.min(lo[i-2:i]) and lo[i] < np.min(lo[i+1:i+3])]
    matches=[]
    for j in pivots:
        for i in pivots:
            if i>=j or not (7 <= j-i <= 62) or n-1-j>34:continue
            valley1=float(lo[i]); valley2=float(lo[j])
            trough_ratio=abs(valley2/valley1-1)
            if trough_ratio>0.065:continue
            neckline=float(np.max(hi[i+1:j]))
            height=neckline/min(valley1,valley2)-1
            if height<0.055 or height>0.55:continue
            # A deeper third trough after the second invalidates the pattern.
            if np.min(lo[j+1:]) < min(valley1,valley2)*0.975:continue
            breakout_idx=next((k for k in range(j+2,n) if close[k]>neckline*1.006 and close[k-1]<=neckline*1.006),None)
            if breakout_idx is not None and breakout_idx >= n-15:
                low_after=lo[breakout_idx+1:]
                if len(low_after) and np.min(low_after)<neckline*.968:
                    stage='FAILED'; score=12.0
                elif n-1-breakout_idx >= 2 and len(low_after) and np.min(low_after) <= neckline*1.024 and close[-1]>=neckline:
                    stage='RETEST';score=90.0
                else:
                    stage='BREAKOUT';score=86.0
                recent_vol=volume[breakout_idx]
                prior_vol=volume[max(0,breakout_idx-20):breakout_idx]
                vr=recent_vol/np.mean(prior_vol) if len(prior_vol) and np.mean(prior_vol)>0 else None
                if stage=='BREAKOUT' and vr is not None and vr<1.10:score-=18.0
            elif close[-1]>=valley2*1.02 and close[-1]<neckline*1.006:
                stage='FORMING';score=58+18*min(1.0,max(0,(close[-1]/valley2-1)/height))
                vr=None
            else:
                continue
            score-=min(14.0,trough_ratio*100)
            matches.append((j,i,{'stage':stage,'score':_clip(score),
                'neckline':round(neckline,3),
                'support':round(min(valley1,valley2),3),
                'trough_gap_pct':round(trough_ratio*100,2),
                'gap_sessions':j-i,'breakout_volume_ratio':round(vr,2) if vr is not None else None,
                'reason':{'FORMING':'雙底形成中，頸線尚未突破','BREAKOUT':'已突破頸線，需防假突破',
                          'RETEST':'突破後回測頸線暫守住','FAILED':'跌破頸線，W底失效'}.get(stage,'W底資料不充分')}))
    if matches:
        matches.sort(key=lambda q:(q[0],q[1]),reverse=True)
        return matches[0][2]
    return {'stage':'NONE','score':28.0,'reason':'未找到已確認的雙底結構'}


def technical_factors(df:pd.DataFrame, *, benchmark_ret5=None, benchmark_ret20=None):
    required={'Open','High','Low','Close','Volume'}
    if df is None or len(df)<145 or not required.issubset(df.columns):
        return {'available':False,'reason':'至少145根完整日K'}
    px=df.tail(150).loc[:,['Open','High','Low','Close','Volume']].apply(pd.to_numeric,errors='coerce')
    if px.isna().any().any() or (px[['Open','High','Low','Close']]<=0).any().any() or (px['Volume']<0).any():
        return {'available':False,'reason':'價量資料異常'}
    if (px['High']<px[['Open','Low','Close']].max(axis=1)).any() or (px['Low']>px[['Open','Close']].min(axis=1)).any():
        return {'available':False,'reason':'OHLC價位不一致'}
    c=px['Close'];o=px['Open'];h=px['High'];lo=px['Low'];v=px['Volume'];dt=str(px.index[-1])[:10]
    ma20=float(c.iloc[-20:].mean());ma60=float(c.iloc[-60:].mean())
    k,d=_stoch_kd(h,lo,c)
    rsi=_rsi_wilder(c)
    ema12=c.ewm(span=12,adjust=False).mean();ema26=c.ewm(span=26,adjust=False).mean()
    dif=ema12-ema26; signal=dif.ewm(span=9,adjust=False).mean();hist=dif-signal
    cross_kd=(k.shift(1)<=d.shift(1)) & (k>d)
    recent_kd=bool(cross_kd.tail(3).any())
    low_kd=float(k.tail(5).min())
    kd_score=84 if recent_kd and low_kd < 35 else 66 if recent_kd else 54 if k.iloc[-1]>d.iloc[-1] and k.iloc[-1]>k.iloc[-2] else 24
    macd_up=bool(hist.iloc[-1]>0 and hist.iloc[-2]<=0)
    macd_3=bool(((hist>0)&(hist.shift(1)<=0)).tail(3).any())
    macd_score=90 if macd_up and dif.iloc[-1]>0 else 82 if macd_3 else 72 if dif.iloc[-1]>0 and hist.iloc[-1]>0 and hist.iloc[-1]>hist.iloc[-2] else 53 if hist.iloc[-1]>hist.iloc[-2] else 26
    # RSI oversold alone is a warning; a recovering oscillator is the bullish condition.
    rsi_now=float(rsi.iloc[-1]);rsi_prev=float(rsi.iloc[-2]);rsi_low=float(rsi.tail(7).min())
    rsi_score=86 if rsi_low<32 and rsi_now>35 and rsi_now>rsi_prev else 72 if rsi_low<38 and rsi_now>rsi_prev+2 else 58 if 45<=rsi_now<=65 and rsi_now>rsi_prev else 25 if rsi_now<30 else 43
    bias=(float(c.iloc[-1])/ma20-1)*100
    bprev=(float(c.iloc[-2])/float(c.iloc[-21:-1].mean())-1)*100
    bias_score=82 if -12<=bias<=-2 and bias>bprev+0.45 else 63 if -6<=bias<=3 and bias>bprev else 25 if bias < -12 and bias<=bprev else 45
    avg20=float(v.iloc[-21:-1].mean()); vr=float(v.iloc[-1]/avg20) if avg20>0 else None
    vol_short=float(v.iloc[-5:].mean()); vol_long=float(v.iloc[-25:-5].mean())
    # Dry-up without stabilization is NOT positive evidence (a falling knife).
    dry=vol_long>0 and vol_short/vol_long<.75
    stable=float(c.iloc[-1])>=float(c.iloc[-6])*.98 and float(c.iloc[-1])>float(c.iloc[-2])
    dry_score=82 if dry and stable and hist.iloc[-1]>hist.iloc[-2] else 54 if dry and stable else 17 if dry else 40
    volume_break=(_clip(50+30*math.log2(max(.1,vr))) if vr is not None else None)
    if vr is not None and vr>4.5:volume_break=_clip(volume_break-22)
    high20=float(h.iloc[-21:-1].max()); dist=(float(c.iloc[-1])/high20-1)*100
    breakout=87 if dist>0.3 and (vr or 0)>=1.2 else 70 if dist>-.8 and (vr or 0)>=.95 else 45 if dist>-4 else 25
    b5=_num(benchmark_ret5);b20=_num(benchmark_ret20)
    relative=None if b5 is None or b20 is None else _clip(50+500*(float(c.iloc[-1]/c.iloc[-6]-1)-b5)+120*(float(c.iloc[-1]/c.iloc[-21]-1)-b20))
    w=w_bottom_pattern(px)
    f={'w_bottom':_part(w.get('score'),w.get('reason'),dt,'完成日K'),
       'kd_turn':_part(kd_score,'KD低檔黃金交叉' if recent_kd and low_kd<35 else 'KD位置與轉折',dt,'完成日K'),
       'macd_turn':_part(macd_score,'MACD柱狀體／DIF轉強',dt,'完成日K'),
       'rsi_rebound':_part(rsi_score,'RSI超跌回升（超跌本身非買進）',dt,'完成日K'),
       'dryup':_part(dry_score,'量縮須配合價格止跌',dt,'完成日K'),
       'bias_reversion':_part(bias_score,'MA20負乖離修復',dt,'完成日K'),
       'volume_break':_part(volume_break,'價格轉強配合成交量',dt,'完成日K'),
       'breakout':_part(breakout,'距前20日高點與量比',dt,'完成日K'),
       'relative_accel':_part(relative,'相對同日大盤5日／20日強度' if relative is not None else '缺少同日大盤證據',dt,'完成日K')}
    return {'available':True,'price_date':dt,'factors':f,'w_bottom':w,
            'metrics':{'kd_k':round(float(k.iloc[-1]),2),'kd_d':round(float(d.iloc[-1]),2),
                       'macd_dif':round(float(dif.iloc[-1]),4),'macd_signal':round(float(signal.iloc[-1]),4),
                       'macd_hist':round(float(hist.iloc[-1]),4),'rsi14':round(rsi_now,2),
                       'ma20_bias_pct':round(bias,2),'volume_ratio':round(vr,2) if vr is not None else None}}


def _date(raw):
    try:return date.fromisoformat(str(raw)[:10]).isoformat()
    except (TypeError,ValueError):return None


def _verified_series(rows, *, price_date, trading_dates, keys, min_sessions=1):
    """Requires precisely the last N trading sessions; missing days invalidate streak."""
    dates=[_date(x) for x in trading_dates]
    dates=[x for x in dates if x and x<=str(price_date)]
    if len(dates)<min_sessions:return []
    unique={}
    for row in rows:
        if not isinstance(row,dict):continue
        d=_date(row.get('date'))
        if d and d<=str(price_date) and d not in unique:unique[d]=row
    if not all(k in unique for k in dates[-min_sessions:]):return []
    result=[unique[k] for k in dates[-min_sessions:]]
    if not all(all(_num(r.get(field)) is not None for field in keys) for r in result):return []
    return result


def chip_factors(research:dict|None, *, price_date:str, trading_dates, avg_volume_shares:float, stable_price:bool=False):
    """PIT fail-closed for daily streak and weekly holdings.

    A downloaded *today* research record can be displayed prospectively, but
    cannot be considered a historical backtest observation without ledger proof.
    """
    research=research if isinstance(research,dict) else {}
    inst=research.get('institutional_flow') or {}
    # An explicit provider PIT invalidity overrides all numeric data.
    if (inst.get('pit') and not inst['pit'].get('valid')) or not inst.get('available'):
        inst={}
    rows=inst.get('daily_rows') or []
    actual=_verified_series(rows,price_date=price_date,trading_dates=trading_dates,
                            keys=('foreign_net_lots','trust_net_lots'),min_sessions=3)
    last=_verified_series(rows,price_date=price_date,trading_dates=trading_dates,
                          keys=('foreign_net_lots','trust_net_lots'),min_sessions=1)
    avg_lots=max(1.,float(avg_volume_shares)/1000.)
    def streak(field):
        if not actual:return None
        vals=[float(r[field]) for r in actual]
        count=sum(v>0 for v in vals)
        # A 1/3 or 2/3 positive history is NEVER a 3-day consecutive buy.
        base={0:19,1:38,2:57,3:80}[count]
        return _clip(base+12*np.tanh(sum(vals)/max(1.,avg_lots*3)*3))
    trust_count=sum(float(r['trust_net_lots'])>0 for r in actual)
    foreign_count=sum(float(r['foreign_net_lots'])>0 for r in actual)
    trust=streak('trust_net_lots');foreign=streak('foreign_net_lots')
    sync=None; pairs=0
    if actual:
        pairs=sum(float(r['foreign_net_lots'])>0 and float(r['trust_net_lots'])>0 for r in actual)
        sync=_clip(20+25*pairs)
    day=None
    if last:
        day=_clip(50+30*np.tanh((float(last[-1]['foreign_net_lots'])+float(last[-1]['trust_net_lots']))/avg_lots*4))
    # Dealer hedging / total is NEVER a proxy for dealer self-directed buying.
    dealer_rows=_verified_series(rows,price_date=price_date,trading_dates=trading_dates,
                                 keys=('dealer_self_net_lots',),min_sessions=1)
    dealer_3=_verified_series(rows,price_date=price_date,trading_dates=trading_dates,
                              keys=('dealer_self_net_lots',),min_sessions=3)
    if dealer_3:
        count=sum(float(r['dealer_self_net_lots'])>0 for r in dealer_3)
        dealer=_clip(27+17*count+10*np.tanh(sum(float(r['dealer_self_net_lots']) for r in dealer_3)/avg_lots))
    else:
        dealer=_clip(50+12*np.tanh(float(dealer_rows[-1]['dealer_self_net_lots'])/avg_lots*4)) if dealer_rows else None
    margin=research.get('margin_trading') or research.get('margin') or {}
    if margin.get('pit') and not margin['pit'].get('valid'):margin={}
    margin_date=_date(margin.get('date') or margin.get('latest_date'))
    valid_margin_date=bool(margin_date and margin_date in {_date(t) for t in list(trading_dates)[-7:]} and margin_date<=price_date)
    delta=_num(margin.get('margin_lots_delta_5d')) if valid_margin_date else None
    margin_score=(_clip(80 if delta<0 and stable_price else 44 if delta<0 else 27 if delta>0 and not stable_price else 57) if delta is not None else None)
    borrow=_num(margin.get('short_lots_delta_3d')) if delta is not None else None
    borrow_score=_clip(78 if borrow<0 else 35 if borrow>0 else 50) if borrow is not None else None
    holders=research.get('holding_distribution') or research.get('shareholding') or {}
    hist=sorted((r for r in (holders.get('history') or []) if isinstance(r,dict) and _date(r.get('date')) and _date(r.get('date'))<=price_date),key=lambda r:r.get('date'))
    holder_score=None
    if len(hist)>=2:
        dlast=_date(hist[-1].get('date'))
        if (date.fromisoformat(price_date)-date.fromisoformat(dlast)).days<=14:
            a,b=_num(hist[-1].get('large_holder_pct')),_num(hist[-2].get('large_holder_pct'))
            if a is not None and b is not None and 0<=a<=100 and 0<=b<=100:
                holder_score=_clip(50+(a-b)*20)
    branch=research.get('main_force_proxy') or {}
    branch_score=None
    if branch.get('available') and (not branch.get('pit') or branch['pit'].get('valid')):
        branch_rows=_verified_series(branch.get('daily_rows') or [],price_date=price_date,trading_dates=trading_dates,
                                    keys=('net_lots',),min_sessions=3)
        if branch_rows:
            branch_score=_clip(50+25*np.tanh(sum(float(r['net_lots']) for r in branch_rows)/avg_lots*2))
    factors={
        'trust_streak':_part(trust,f'投信近3日買超 {trust_count}/3日；須3/3才算連買' if actual else '缺連續三交易日投信資料',price_date,'法人逐日'),
        'foreign_streak':_part(foreign,f'外資近3日買超 {foreign_count}/3日；須3/3才算連買' if actual else '缺連續三交易日外資資料',price_date,'法人逐日'),
        'foreign_trust_sync':_part(sync,f'外資投信同步買超 {pairs}/3日' if actual else '缺連續三交易日同步證據',price_date,'法人逐日'),
        'today_net':_part(day,'外資＋投信當日淨買超（不包含避險自營商）',price_date,'法人逐日'),
        'dealer_self':_part(dealer,'自營商自行買賣，不含避險',price_date,'法人逐日'),
        'margin_clean':_part(margin_score,'融資5日變化與股價是否止穩',price_date,'融資餘額'),
        'large_holder':_part(holder_score,'千張以上大戶持股連週變化',price_date,'集保週資料'),
        'borrow_down':_part(borrow_score,'融券餘額三日變化（非借券數量）',price_date,'融券餘額'),
        'branch_proxy':_part(branch_score,'券商分點代理，非官方主力定義',price_date,'分點逐日'),
    }
    sc=_score(factors,CHIP_WEIGHTS,min_coverage=.10)
    return {**sc,'factors':factors,'continuous_flow_sessions':len(actual),'today_flow_sessions':len(last),
            'trust_positive_days_3':trust_count if actual else None,
            'foreign_positive_days_3':foreign_count if actual else None,
            'joint_positive_days_3':pairs if actual else None,
            'chip_evidence_note':'交易日不完整不計連買；自營避險不當自行買賣；大戶為週資料；未證明歷史公告日者僅供前瞻觀察',
            'historical_pit_verified':bool(inst.get('pit') and inst['pit'].get('valid'))}


def revenue_factor(research, *, decision_date:str, decision_at:str|None=None):
    rev=(research or {}).get('monthly_revenue') or {}
    if rev.get('pit') and not rev['pit'].get('valid'):
        return _part(None,'月營收資料時間不符',decision_date)
    latest=rev.get('latest') or {}
    rows=sorted([r for r in (rev.get('rows') or []) if isinstance(r,dict)],key=lambda x:str(x.get('month') or ''))
    eligible=[]
    for row in rows:
        month=str(row.get('month') or '')
        # YYYY-MM is a fiscal period, not an invented release timestamp.
        # Current-month numbers are not a completed and released month.
        if len(month)==7 and month[4]=='-' and month[:4].isdigit() and month[5:].isdigit() and month < decision_date[:7]:
            eligible.append(row)
    yoy=_num(eligible[-1].get('yoy_pct')) if eligible else None
    prev=_num(eligible[-2].get('yoy_pct')) if len(eligible)>1 else None
    # A latest-only source must provide the actual completed reference month.
    if yoy is None and not rows:
        ref=str(latest.get('month') or '')
        if len(ref)==7 and ref < decision_date[:7] and rev.get('pit',{}).get('valid') is True:
            yoy=_num(latest.get('yoy_pct'))
    if yoy is None:return _part(None,'缺少可核對營收YoY',decision_date)
    # Treat historical replay with no contemporaneous PIT as unverifiable.
    if decision_at and rev.get('pit') and rev['pit'].get('valid') is False:return _part(None,'營收公告晚於決策日',decision_date)
    accel=(yoy-prev) if prev is not None else None
    score=50+1.2*yoy+(min(18,max(-18,accel*1.3)) if accel is not None else 0)
    return _part(score,f"營收YoY {yoy:+.1f}%"+(f'；月增速變化 {accel:+.1f}百分點' if accel is not None else '；無前月不得宣稱加速'),decision_date,rev.get('source','營收'))


def build_candidate(stock:dict, technical:dict, *, research:dict|None=None, trading_dates=(), benchmark=None) ->dict:
    """New independent technical/chip shadow lane. Production ranking unchanged."""
    date_label=str(stock.get('price_date') or '')
    if not technical.get('available') or date_label != technical.get('price_date'):
        return {'version':VERSION,'available':False,'reason':'型態價格與日線模型日期不一致','is_buy_signal':False}
    stock_research=research if research is not None else stock.get('ranking_research') or {}
    metrics=technical.get('metrics') or {}
    chip=chip_factors(stock_research,price_date=date_label,trading_dates=trading_dates,
                      avg_volume_shares=float(stock.get('avg_volume_20d') or 0),
                      stable_price=_num(metrics.get('ma20_bias_pct')) is not None and float(metrics['ma20_bias_pct'])>=-6)
    revenue=revenue_factor(stock_research,decision_date=date_label)
    factors=dict(technical['factors']);factors['revenue_growth']=revenue;factors['chips']=_part(chip.get('score'),
                         '籌碼子因子，不能用單日買超冒充連續買超',date_label,'多資料來源')
    rev=_score(factors,REVERSAL_WEIGHTS)
    breakout=_score(factors,BREAKOUT_WEIGHTS)
    alpha=_num(((stock.get('horizons') or {}).get('short') or {}).get('ranking_score'))
    rev_rank=_clip(.35*alpha+.65*rev['score']) if alpha is not None and rev.get('score') is not None else None
    break_rank=_clip(.35*alpha+.65*breakout['score']) if alpha is not None and breakout.get('score') is not None else None
    rank=max(x for x in (rev_rank,break_rank) if x is not None) if any(x is not None for x in (rev_rank,break_rank)) else None
    mode='反轉蓄勢' if rev_rank is not None and (break_rank is None or rev_rank>=break_rank) else '突破加速'
    blockers=[]
    if (rev['evidence_coverage'] if mode=='反轉蓄勢' else breakout['evidence_coverage'])<.65:blockers.append('完整證據不足')
    if _num(stock.get('avg_turnover_20d')) is None or float(stock['avg_turnover_20d'])<30_000_000:blockers.append('成交額不足')
    if _num(metrics.get('ma20_bias_pct')) is not None and metrics['ma20_bias_pct']>12:blockers.append('股價過熱')
    if ((stock.get('selection_quality') or {}).get('short') or {}).get('eligible') is False:blockers.append('原模型品質門檻')
    return {'version':VERSION,'available':True,'is_buy_signal':False,'is_calibrated_probability':False,
            'date':date_label,'mode':mode,'research_rank':rank,'reversal_rank':rev_rank,'breakout_rank':break_rank,
            'reversal_score':rev,'breakout_score':breakout,'w_bottom':technical.get('w_bottom') or {},
            'technical_metrics':metrics,'factors':factors,'chip':chip,
            'eligible_research':len(blockers)==0 and rank is not None and rank>=60,'blockers':blockers,
            'alpha':alpha,'interpretation':'實驗性研究排序，非漲停機率或正式買點'}


def research_board(snapshot:dict|None, *, n=10, only_eligible=False)->list[dict]:
    if not isinstance(snapshot,dict):return []
    dt=str(snapshot.get('price_date') or '');out=[]
    for s in snapshot.get('stocks') or []:
        lane=s.get('pattern_chip_shadow') or {}
        if lane.get('version')!=VERSION or not lane.get('available') or str(lane.get('date'))!=dt:continue
        if only_eligible and not lane.get('eligible_research'):continue
        if _num(lane.get('research_rank')) is None:continue
        w=lane.get('w_bottom') or {}
        out.append({'ticker':s.get('ticker'),'name':s.get('name'),'alpha':lane.get('alpha'),
                    'score':lane.get('research_rank'),'mode':lane.get('mode'),
                    'w_stage':w.get('stage'),'neckline':w.get('neckline'),
                    'reversal':lane.get('reversal_rank'),'breakout':lane.get('breakout_rank'),
                    'chip_coverage':(lane.get('chip') or {}).get('evidence_coverage'),
                    'eligible':lane.get('eligible_research'),
                    'blockers':lane.get('blockers') or [],'date':dt,
                    'is_buy_signal':False})
    out.sort(key=lambda x:(-x['score'],str(x['ticker'])))
    return out[:max(0,min(int(n),30))]
