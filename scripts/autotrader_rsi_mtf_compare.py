"""Research-only multi-timeframe filter on unchanged BASE RSI+EMA entries.

Uses the last COMPLETED calendar week's RSI and causally confirmed daily swing lows.
No live orders, no revised entries, no future weekly candle leakage.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.autotrader.historical import load_parquet,walkforward
from src.autotrader.rsi_retest import detect_rsi_retests,rsi_sma
from src.autotrader.learning import Bar,D

def context(bars,weekly_floor=D("60"),pivot_span=2):
    import pandas as pd
    # Group by ISO calendar week, not a rolling 5-day lookahead.
    dates=[pd.Timestamp(b.timestamp).date() for b in bars]
    keys=[d.isocalendar()[:2] for d in dates]
    groups=defaultdict(list)
    for i,key in enumerate(keys):groups[key].append(i)
    ordered=sorted(groups)
    weekly=[]
    last_completed={}
    for key in ordered:
        indices=groups[key]
        end=indices[-1]
        weekly.append(Bar(bars[end].timestamp,bars[indices[0]].open,
                          max(bars[j].high for j in indices),
                          min(bars[j].low for j in indices),bars[end].close,0))
        last_completed[key]=len(weekly)-2
    wrsi,_=rsi_sma(weekly)
    # A pivot at j is only known at j+pivot_span, never at j.
    confirmed=[]
    pivots=[]
    for i in range(len(bars)):
        j=i-pivot_span
        if j>=pivot_span:
            low=bars[j].low
            if all(low<bars[k].low for k in range(j-pivot_span,j)) and all(low<=bars[k].low for k in range(j+1,j+pivot_span+1)):
                pivots.append((j,low))
        confirmed.append(tuple(pivots[-2:]))
    result={}
    for i,b in enumerate(bars):
        wi=last_completed[keys[i]]
        weekly_rsi=wrsi[wi] if wi>=0 else None
        pair=confirmed[i]
        higher_low=len(pair)==2 and pair[-1][1]>pair[-2][1]
        intact=higher_low and b.close>pair[-1][1]
        result[b.timestamp]=dict(weekly_rsi=str(weekly_rsi) if weekly_rsi is not None else None,
            weekly_ok=weekly_rsi is not None and weekly_rsi>=weekly_floor,
            higher_low=higher_low,structure_intact=intact,
            last_pivot_at=bars[pair[-1][0]].timestamp if pair else None)
    return result

def net_cost(records,fee):
    for r in records:
        if r["status"]!="COMPLETE":continue
        entry=D(r["entry"]);exit_price=D(r["exit_price"]);risk=entry-D(r["stop"])
        r["gross_r"]=r["r_multiple"]
        r["r_multiple"]=str((exit_price-entry-(entry+exit_price)*fee/D(10000))/risk)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--directory",default="data/market")
    p.add_argument("--split-at",default="2026-04-01")
    p.add_argument("--weekly-rsi-min",default="60")
    p.add_argument("--pivot-span",type=int,default=2)
    p.add_argument("--fee-bps",default="10")
    p.add_argument("--output",default="data/autotrader/rsi_mtf_comparison.json")
    a=p.parse_args()
    if a.pivot_span<1 or D(a.fee_bps)<0 or not D(0)<=D(a.weekly_rsi_min)<=D(100):p.error("Invalid parameters")
    paths=sorted(Path(a.directory).glob("*/1D/ohlc.parquet"))
    if not paths:p.error("No daily parquet found")
    report=dict(split_at=a.split_at,weekly_rsi_min=a.weekly_rsi_min,pivot_span=a.pivot_span,fee_bps=a.fee_bps,
        note="Research only. Same BASE trades filtered at signal close by last completed weekly RSI and two confirmed daily pivot lows. Weekly partial candles excluded. Parameters are exploratory, not optimized.",
        symbols={})
    evidence=[]
    for path in paths:
        symbol=path.parent.parent.name.upper()
        try:
            bars=load_parquet(path)
            ctx=context(bars,D(a.weekly_rsi_min),a.pivot_span)
            base=detect_rsi_retests(symbol,bars)
            net_cost(base,D(a.fee_bps))
            selected=[]
            for r in base:
                c=ctx[r["signal_at"]]
                ok=c["weekly_ok"] and c["structure_intact"]
                evidence.append(dict(variant="BASE",**r,**c,mtf_pass=ok))
                if ok:selected.append(r)
            report["symbols"][symbol]={"BASE":walkforward(base,a.split_at),
                "MTF":walkforward(selected,a.split_at),"base_signals":len(base),"mtf_signals":len(selected)}
        except (ValueError,RuntimeError,TypeError,KeyError) as exc:
            report["symbols"][symbol]={"error":str(exc)}
    for variant in ("BASE","MTF"):
        for period in ("train","unseen"):
            n=w=0;total=D(0)
            for item in report["symbols"].values():
                if variant not in item:continue
                s=item[variant][period];n+=s["complete"];w+=s["wins"]
                if s["complete"]:total+=D(s["mean_r"])*s["complete"]
            report.setdefault("aggregate",{}).setdefault(variant,{})[period]={
                "completed":n,"wins":w,"win_rate_pct":str(D(100)*w/n) if n else None,
                "mean_net_r":str(total/n) if n else None}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(".tmp");tmp.write_text(json.dumps(report,indent=2),encoding="utf-8");tmp.replace(out)
    ev=out.with_name(out.stem+"_evidence.jsonl");tmp=ev.with_suffix(".tmp")
    with tmp.open("w",encoding="utf-8") as f:
        for r in evidence:f.write(json.dumps(r,sort_keys=True)+"\n")
    tmp.replace(ev)
    for variant in ("BASE","MTF"):
        print(variant,report["aggregate"][variant])
    print("Report:",out,"Evidence:",ev)
if __name__=="__main__":main()
