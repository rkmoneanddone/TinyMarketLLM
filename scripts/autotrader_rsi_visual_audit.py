"""Research-only visual signal audit: exports RSI/EMA context without changing rules."""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.autotrader.historical import load_parquet
from src.autotrader.rsi_retest import detect_rsi_retests,rsi_sma,ema_alignment
from src.autotrader.learning import D

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",default="data/market")
    p.add_argument("--split-at",default="2026-04-01")
    p.add_argument("--output",default="data/autotrader/rsi_signal_visual_audit.csv")
    args=p.parse_args()
    paths=sorted(Path(args.directory).glob("*/1D/ohlc.parquet"))
    if not paths:p.error("No 1D OHLC parquet found")
    rows=[]
    for path in paths:
        symbol=path.parent.parent.name.upper()
        bars=load_parquet(path)
        index={b.timestamp:i for i,b in enumerate(bars)}
        rsi,ma=rsi_sma(bars)
        ema=ema_alignment(bars)
        for rec in detect_rsi_retests(symbol,bars):
            if "cross_at" not in rec or "retest_at" not in rec:continue
            cross=index[rec["cross_at"]];retest=index[rec["retest_at"]];signal=index[rec["signal_at"]]
            # This diagnostic is deliberately NOT a trading condition.
            initial_peak=max(rsi[cross:retest]) if retest>cross else rsi[cross]
            pullback=rsi[retest]<rsi[retest-1]
            moved_away=initial_peak>ma[cross]*D("1.05")
            rows.append(dict(symbol=symbol,period="UNSEEN" if rec["signal_at"]>=args.split_at else "TRAIN",
                cross_at=rec["cross_at"],retest_at=rec["retest_at"],signal_at=rec["signal_at"],
                status=rec["status"],outcome=rec.get("outcome",""),r_multiple=rec.get("r_multiple",""),
                cross_rsi=str(rsi[cross]),retest_rsi=str(rsi[retest]),retest_rsi_sma=str(ma[retest]),
                confirmation_rsi=str(rsi[signal]),confirmation_rsi_sma=str(ma[signal]),
                retest_below_previous_rsi=pullback,initial_rsi_moved_5pct_above_sma=moved_away,
                ema_aligned_on_confirmation=ema[signal],
                confirmation_close_above_previous_high=bars[signal].close>bars[signal-1].high,
                signal_close=str(bars[signal].close),prior_high=str(bars[signal-1].high),
                entry=rec.get("entry",""),stop=rec.get("stop",""),target=rec.get("target","")))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]) if rows else ["symbol","period","signal_at","status"]
    tmp=out.with_suffix(".tmp")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(out)
    for period in ("TRAIN","UNSEEN"):
        subset=[r for r in rows if r["period"]==period]
        print(f"{period}: {len(subset)} auditable signals; {sum(r['retest_below_previous_rsi'] for r in subset)} with falling RSI on retest; {sum(r['initial_rsi_moved_5pct_above_sma'] for r in subset)} with prior RSI extension")
    print(f"Signal audit: {out}")
    print("Diagnostics are descriptive only; strategy and paper trading are unchanged.")
if __name__=="__main__":main()
