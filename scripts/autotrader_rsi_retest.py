"""Backtest the single confirmed RSI retest strategy across canonical 1D market data."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.autotrader.historical import load_parquet,walkforward
from src.autotrader.rsi_retest import detect_rsi_retests
from src.autotrader.learning import D

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--directory",default="data/market")
    p.add_argument("--split-at",required=True)
    p.add_argument("--fee-bps",default="10")
    p.add_argument("--output",default="data/autotrader/rsi_retest_report.json")
    args=p.parse_args()
    if D(args.fee_bps)<0:p.error("fee-bps must be nonnegative")
    paths=sorted(Path(args.directory).glob("*/1D/ohlc.parquet"))
    if not paths:p.error("No 1D market parquet files found")
    report={"strategy":"RSI_MA_RETEST_1D","split_at":args.split_at,
            "fee_bps":args.fee_bps,"symbols":{}}
    evidence=[]
    for path in paths:
        symbol=path.parent.parent.name.upper()
        try:
            bars=load_parquet(path)
            records=detect_rsi_retests(symbol,bars)
            for r in records:
                if r["status"]=="COMPLETE":
                    entry=D(r["entry"]);exit_price=D(r["exit_price"])
                    risk=entry-D(r["stop"])
                    r["gross_r"]=r["r_multiple"]
                    r["r_multiple"]=str((exit_price-entry-(entry+exit_price)*D(args.fee_bps)/D(10000))/risk)
            report["symbols"][symbol]={"bars":len(bars),**walkforward(records,args.split_at)}
            evidence.extend(records)
        except (ValueError,RuntimeError,TypeError,KeyError) as exc:
            report["symbols"][symbol]={"error":str(exc)}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(".tmp");tmp.write_text(json.dumps(report,indent=2),encoding="utf-8");tmp.replace(out)
    evidence_path=out.with_name(out.stem+"_evidence.jsonl")
    tmp=evidence_path.with_suffix(".tmp")
    with tmp.open("w",encoding="utf-8") as f:
        for r in evidence:f.write(json.dumps(r,sort_keys=True)+"\n")
    tmp.replace(evidence_path)
    for symbol,result in report["symbols"].items():
        if "error" in result:print(f"{symbol}: ERROR {result['error']}")
        else:
            u=result["unseen"]
            print(f"{symbol}: unseen {u['complete']} trades, win {u['win_rate']}%, avg {u['mean_r']}R; skipped {result['skipped']}, pending {result['pending']}")
    print(f"Report: {out}\nEvidence: {evidence_path}")
if __name__=="__main__":main()
