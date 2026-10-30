"""Audit RSI retest signals and compare fixed exit horizons without changing entry rules.

Research-only. Horizons are alternative scenarios, not out-of-sample optimization.
"""
import argparse
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.autotrader.historical import load_parquet, walkforward
from src.autotrader.rsi_retest import detect_rsi_retests
from src.autotrader.learning import D, evaluate_setup

def evaluate_horizons(bars,symbol,horizons,fee_bps):
    # Run detection once at the longest horizon; entries and confirmations are unchanged.
    base=detect_rsi_retests(symbol,bars,horizon=max(horizons))
    index={b.timestamp:i for i,b in enumerate(bars)}
    variants={}
    for horizon in horizons:
        records=[]
        for original in base:
            r=dict(original)
            if "entry" in r and "target" in r and r["status"]!="SKIPPED":
                i=index[r["signal_at"]]
                result=evaluate_setup(symbol,r["setup"],i,bars,D(r["entry"]),D(r["stop"]),D(r["target"]),horizon)
                r.update(result)
                r["status"]="COMPLETE" if result["complete"] else "PENDING"
                if r["status"]=="COMPLETE":
                    entry=D(r["entry"]);exit_price=D(r["exit_price"]);risk=entry-D(r["stop"])
                    r["gross_r"]=r["r_multiple"]
                    r["r_multiple"]=str((exit_price-entry-(entry+exit_price)*D(fee_bps)/D(10000))/risk)
            records.append(r)
        variants[str(horizon)]=records
    return variants

def audit_records(records):
    return {"outcomes":dict(Counter(r.get("outcome","UNKNOWN") for r in records if r["status"]=="COMPLETE")),
            "skipped_reasons":dict(Counter(r.get("reason","UNKNOWN") for r in records if r["status"]=="SKIPPED")),
            "missing_crossover":sum("cross_at" not in r for r in records if r["status"]=="COMPLETE"),
            "missing_retest":sum("retest_at" not in r for r in records if r["status"]=="COMPLETE"),
            "invalid_event_order":sum(not (r.get("cross_at","") < r.get("retest_at","") < r.get("signal_at","") < r.get("entry_at","")) for r in records if r["status"]=="COMPLETE")}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--directory",default="data/market")
    p.add_argument("--split-at",default="2026-04-01")
    p.add_argument("--horizons",default="5,10,15")
    p.add_argument("--fee-bps",default="10")
    p.add_argument("--output",default="data/autotrader/rsi_horizon_audit.json")
    args=p.parse_args()
    try:
        horizons=sorted(set(int(v.strip()) for v in args.horizons.split(",")))
        if not horizons or min(horizons)<1 or D(args.fee_bps)<0:raise ValueError()
    except ValueError:p.error("Positive integer horizons and nonnegative fee-bps required")
    paths=sorted(Path(args.directory).glob("*/1D/ohlc.parquet"))
    if not paths:p.error("No canonical daily market data found")
    report={"split_at":args.split_at,"horizons":horizons,"fee_bps":args.fee_bps,
            "note":"Horizon scenarios share entries; results are not independent trials. No parameter selection on unseen results.",
            "symbols":{}}
    for path in paths:
        symbol=path.parent.parent.name.upper()
        try:
            bars=load_parquet(path)
            variants=evaluate_horizons(bars,symbol,horizons,args.fee_bps)
            report["symbols"][symbol]={h:{"performance":walkforward(records,args.split_at),
                                          "audit":audit_records(records)}
                                       for h,records in variants.items()}
        except (ValueError,RuntimeError,TypeError,KeyError) as exc:
            report["symbols"][symbol]={"error":str(exc)}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(".tmp");tmp.write_text(json.dumps(report,indent=2),encoding="utf-8");tmp.replace(out)
    for h in horizons:
        count=wins=0;total=D(0);outcomes=Counter()
        for item in report["symbols"].values():
            if str(h) not in item:continue
            stats=item[str(h)]["performance"]["unseen"]
            # Mean is weighted by the number of trades, not by symbols.
            n=stats["complete"];count+=n;wins+=stats["wins"]
            if n:total+=D(stats["mean_r"])*n
            # Outcome breakdown includes training and unseen; not included in headline.
        print(f"{h} bars: unseen {count} trades, win {wins}/{count}, avg {total/count if count else 'N/A'}R")
    print(f"Report: {out}")
if __name__=="__main__":main()
