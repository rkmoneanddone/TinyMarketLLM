from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_market_scanner import load_configuration, load_frames, scanner_from, write_report
from src.tiny_market_llm.scanner.market_sections import resample_ohlc, weekly_fvg_hold_sequences


def main() -> None:
    config = load_configuration()
    daily = load_frames(config)["GRASIM"]
    scanner = scanner_from(config)
    weekly = resample_ohlc(daily, "W-FRI")
    records = weekly_fvg_hold_sequences(scanner.prepare(weekly), "GRASIM")
    research = ROOT / "data" / "research"
    research.mkdir(parents=True, exist_ok=True)
    csv_path = research / "GRASIM_weekly_fvg_sequences.csv"
    json_path = research / "GRASIM_weekly_fvg_sequences.json"
    records.to_csv(csv_path, index=False)
    successes = int(records["reached_prior_high"].sum())
    summary = {
        "symbol": "GRASIM", "timeframe": "1W", "weekly_candles": len(weekly),
        "independent_sequences": len(records), "reached_prior_high": successes,
        "success_rate": successes / len(records) if len(records) else None,
        "definition": "bullish FVG -> hold within 12 weeks -> RSI rising -> prior 20-week high within 8 weeks",
    }
    json_path.write_text(json.dumps({"summary": summary, "records": records.to_dict("records")}, indent=2, default=str), encoding="utf-8")
    report = write_report(records, summary, config, "GRASIM_weekly_fvg_sequences")
    print(records.to_string(index=False))
    print(f"\n[SUMMARY] {summary}")
    print(f"[RECORDS] {csv_path} | {json_path}")
    print(f"[REPORT] {report}")


if __name__ == "__main__":
    main()
