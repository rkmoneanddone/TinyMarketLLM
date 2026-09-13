from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_market_scanner import load_configuration, load_frames, scanner_from, write_report
from src.tiny_market_llm.scanner.market_sections import daily_fvg_trade_records


def main() -> None:
    config = load_configuration()
    scanner = scanner_from(config)
    prepared = scanner.prepare(load_frames(config)["GRASIM"])
    records = daily_fvg_trade_records(prepared, "GRASIM", pd.Timestamp("2026-01-01", tz="UTC"))
    research = ROOT / "data" / "research"
    research.mkdir(parents=True, exist_ok=True)
    csv_path = research / "GRASIM_daily_fvg_trades_from_2026.csv"
    json_path = research / "GRASIM_daily_fvg_trades_from_2026.json"
    records.to_csv(csv_path, index=False)
    summary = {
        "symbol": "GRASIM", "timeframe": "1D", "start": "2026-01-01",
        "signals": len(records), "targets_hit": int((records["outcome"] == "TARGET").sum()),
        "stops_hit": int((records["outcome"] == "STOP").sum()),
        "open": int((records["outcome"] == "OPEN").sum()), "minimum_reward_risk": 1.0,
    }
    json_path.write_text(json.dumps({"summary": summary, "records": records.to_dict("records")}, indent=2, default=str), encoding="utf-8")
    report = write_report(records, summary, config, "GRASIM_daily_fvg_trades_from_2026")
    print(records.to_string(index=False))
    print(f"\n[SUMMARY] {summary}")
    print(f"[REPORT] {report}")


if __name__ == "__main__":
    main()
