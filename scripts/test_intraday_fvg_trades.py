from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tiny_market_llm.scanner import TinyMarketScanner
from src.tiny_market_llm.scanner.market_sections import fvg_trade_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest FVG-hold + rising-RSI on local intraday data.")
    parser.add_argument("--symbols", nargs="+", default=["GRASIM", "TCS", "RELIANCE", "TBZ"])
    parser.add_argument("--timeframe", default="1H", choices=["1H"])
    parser.add_argument("--from-date", default="2026-01-01")
    args = parser.parse_args()

    start = pd.Timestamp(args.from_date, tz="UTC")
    all_records: list[pd.DataFrame] = []
    summaries: list[dict] = []
    scanner = TinyMarketScanner()

    for requested_symbol in args.symbols:
        symbol = requested_symbol.strip().upper()
        path = ROOT / "data" / "market" / symbol / args.timeframe / "ohlc.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run download_intraday.py first.")
        source = pd.read_parquet(path)
        prepared = scanner.prepare(source)
        records = fvg_trade_records(
            prepared,
            symbol,
            start,
            timeframe=args.timeframe,
            maximum_wait_candles=12,
            evaluation_candles=20,
        )
        if not records.empty:
            all_records.append(records)
        resolved = records[records["outcome"].isin(["TARGET", "STOP", "TIME_EXIT"])] if not records.empty else records
        profitable = int((resolved["pnl_pct"] > 0).sum()) if not resolved.empty else 0
        summaries.append({
            "symbol": symbol,
            "candles": len(source),
            "signals": len(records),
            "resolved": len(resolved),
            "targets": int((records["outcome"] == "TARGET").sum()) if not records.empty else 0,
            "stops": int((records["outcome"] == "STOP").sum()) if not records.empty else 0,
            "time_exits": int((records["outcome"] == "TIME_EXIT").sum()) if not records.empty else 0,
            "open": int((records["outcome"] == "OPEN").sum()) if not records.empty else 0,
            "ambiguous": int((records["outcome"] == "AMBIGUOUS").sum()) if not records.empty else 0,
            "profitable_resolved": profitable,
            "success_rate_pct": round(profitable / len(resolved) * 100, 2) if len(resolved) else None,
        })

    combined = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
    output_dir = ROOT / "data" / "research"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fvg_hold_rsi_{args.timeframe}_{args.from_date}_four_stocks"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    combined.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "setup": "bullish FVG -> hold/retest -> rising RSI",
        "timeframe": args.timeframe,
        "start": args.from_date,
        "evaluation_candles": 20,
        "minimum_reward_risk": 1.0,
        "summaries": summaries,
        "records": combined.to_dict("records"),
    }, indent=2, default=str), encoding="utf-8")

    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\n[TRADES] {csv_path}")
    print(f"[RECORD] {json_path}")


if __name__ == "__main__":
    main()
