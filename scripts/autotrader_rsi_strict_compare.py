"""Compare frozen RSI+EMA baseline with opt-in strict RSI retest, research only."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.autotrader.historical import load_parquet,walkforward
from src.autotrader.rsi_retest import detect_rsi_retests
from src.autotrader.learning import D

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
    p.add_argument("--fee-bps",default="10")
    p.add_argument("--output",default="data/autotrader/rsi_strict_comparison.json")
    args=p.parse_args()
    fee=D(args.fee_bps)
    if fee<0:p.error("fee-bps must be nonnegative")
    paths=sorted(Path(args.directory).glob("*/1D/ohlc.parquet"))
    if not paths:p.error("No daily parquet found")
    report={"split_at":args.split_at,"fee_bps":args.fee_bps,
        "definitions":{"BASE":"Existing RSI retest + four EMA confirmation",
                       "STRICT":"Same rules plus prior RSI extension >5% above RSI SMA and a falling RSI retest candle"},
        "note":"Research-only comparison, no execution. Strict can shift retest/entry dates; trades are not necessarily a subset of BASE.",
        "symbols":{}}
    evidence=[]
    for path in paths:
        symbol=path.parent.parent.name.upper()
        try:
            bars=load_parquet(path)
            result={}
            for label,strict in (("BASE",False),("STRICT",True)):
                records=detect_rsi_retests(symbol,bars,strict_retest=strict)
                net_cost(records,fee)
                result[label]=walkforward(records,args.split_at)
                evidence.extend({"variant":label,**r} for r in records)
            report["symbols"][symbol]=result
        except (ValueError,RuntimeError,TypeError,KeyError) as exc:
            report["symbols"][symbol]={"error":str(exc)}
    for label in ("BASE","STRICT"):
        for period in ("train","unseen"):
            count=wins=0;total=D(0)
            for item in report["symbols"].values():
                if label not in item:continue
                s=item[label][period];n=s["complete"];count+=n;wins+=s["wins"]
                if n:total+=D(s["mean_r"])*n
            report.setdefault("aggregate",{}).setdefault(label,{})[period]={
                "completed":count,"wins":wins,"win_rate_pct":str(D(wins)*100/count) if count else None,
                "mean_net_r":str(total/count) if count else None}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(".tmp");tmp.write_text(json.dumps(report,indent=2),encoding="utf-8");tmp.replace(out)
    ev=out.with_name(out.stem+"_evidence.jsonl");tmp=ev.with_suffix(".tmp")
    with tmp.open("w",encoding="utf-8") as f:
        for r in evidence:f.write(json.dumps(r,sort_keys=True)+"\n")
    tmp.replace(ev)
    for label in ("BASE","STRICT"):
        for period in ("train","unseen"):
            print(f"{label} {period}: {report['aggregate'][label][period]}")
    print(f"Report: {out}\nEvidence: {ev}")
if __name__=="__main__":main()
