"""Multi-symbol daily OHLC research, with no execution authority.

Runs independently on each symbol. Signal filters use only bars known at signal close.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from decimal import Decimal
from .historical import load_parquet, detect_setups, walkforward
from .learning import D, summarize

def filter_records(bars,records,ema_period=21,volume_period=20,volume_multiple="1.5"):
    by_time={b.timestamp:i for i,b in enumerate(bars)}
    ema=None
    emas={}
    alpha=D(2)/D(ema_period+1)
    for b in bars:
        ema=b.close if ema is None else alpha*b.close+(D(1)-alpha)*ema
        emas[b.timestamp]=ema
    enriched=[]
    for source in records:
        r=dict(source)
        i=by_time[r["signal_at"]]
        previous=bars[i-volume_period:i]
        r["ema_trend_pass"]=i>=ema_period and bars[i].close>emas[r["signal_at"]]
        r["volume_breakout_pass"]=bool(len(previous)==volume_period and
            sum(b.volume for b in previous)>0 and
            D(bars[i].volume)>=D(volume_multiple)*D(sum(b.volume for b in previous))/D(volume_period))
        enriched.append(r)
    return enriched

def group_report(records,split_at,fee_bps="0"):
    fee=D(fee_bps)
    groups={"BASE":lambda r:True,"EMA_TREND":lambda r:r["ema_trend_pass"],
            "VOLUME":lambda r:r["volume_breakout_pass"],
            "EMA_TREND_VOLUME":lambda r:r["ema_trend_pass"] and r["volume_breakout_pass"]}
    result={}
    for name,accept in groups.items():
        chosen=[dict(r) for r in records if accept(r)]
        for r in chosen:
            if r.get("status")=="COMPLETE":
                entry=D(r["entry"]);risk=entry-D(r["stop"])
                exit_price=D(r["exit_price"])
                # Approximate round-trip turnover cost. Does not model slippage/taxes precisely.
                r["r_multiple"]=str((exit_price-entry-(entry+exit_price)*fee/D(10000))/risk)
        result[name]=walkforward(chosen,split_at)
        result[name]["signals"]=len(chosen)
    return result

def run(paths,split_at,lookback=20,horizon=5,fee_bps="10",volume_multiple="1.5"):
    results={};all_records=[]
    for path in paths:
        symbol=path.parent.name.upper() if path.parent.name.upper() not in ("LEARNING","MARKET") else path.stem.split("_")[0].upper()
        try:
            bars=load_parquet(path)
            records=filter_records(bars,detect_setups(symbol,bars,lookback,horizon),volume_multiple=volume_multiple)
            results[symbol]={"file":str(path),"bars":len(bars),
                             "variants":group_report(records,split_at,fee_bps)}
            all_records.extend(records)
        except (ValueError,RuntimeError,KeyError,TypeError) as exc:
            results[symbol]={"file":str(path),"error":str(exc)}
    return {"split_at":split_at,"fee_bps":fee_bps,"symbols":results},all_records

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--directory",default="data/learning")
    p.add_argument("--glob",default="*_1D_training_v2.parquet")
    p.add_argument("--split-at",required=True)
    p.add_argument("--lookback",type=int,default=20)
    p.add_argument("--horizon",type=int,default=5)
    p.add_argument("--fee-bps",default="10")
    p.add_argument("--volume-multiple",default="1.5")
    p.add_argument("--output",default="data/autotrader/batch_learning_report.json")
    args=p.parse_args()
    if D(args.fee_bps)<0 or D(args.volume_multiple)<=0:p.error("Invalid costs or volume threshold")
    paths=sorted(Path(args.directory).rglob(args.glob))
    if not paths:p.error("No matching parquet files found")
    report,records=run(paths,args.split_at,args.lookback,args.horizon,args.fee_bps,args.volume_multiple)
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(".tmp");tmp.write_text(json.dumps(report,indent=2),encoding="utf-8");tmp.replace(dest)
    evidence=dest.with_name(dest.stem+"_evidence.jsonl")
    tmp=evidence.with_suffix(".tmp")
    with tmp.open("w",encoding="utf-8") as f:
        for r in records:f.write(json.dumps(r,sort_keys=True)+"\n")
    tmp.replace(evidence)
    for symbol,data in report["symbols"].items():
        if "error" in data:
            print(f"{symbol}: ERROR {data['error']}")
        else:
            print(f"{symbol}: {data['bars']} bars")
            for name,v in data["variants"].items():
                t=v["unseen"]
                print(f"  {name}: unseen {t['complete']} trades, win {t['win_rate']}%, avg {t['mean_r']}R")
    print(f"Report: {dest}\nEvidence: {evidence}")
if __name__=="__main__":main()
