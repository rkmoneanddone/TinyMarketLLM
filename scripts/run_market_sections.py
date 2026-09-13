from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_market_scanner import load_configuration, load_frames, scanner_from, write_report
from src.tiny_market_llm.scanner.market_sections import high_breakout_history, independent_breakouts


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline TinyMarketLLM market sections")
    parser.add_argument("--section", choices=("high-breakouts",), default="high-breakouts")
    parser.add_argument("--symbol", choices=("GRASIM", "RELIANCE", "TCS"))
    args = parser.parse_args()

    config = load_configuration()
    frames = load_frames(config)
    if args.symbol:
        frames = {args.symbol: frames[args.symbol]}
    scanner = scanner_from(config)

    histories = []
    latest = []
    for symbol, frame in frames.items():
        prepared = scanner.prepare(frame)
        history = high_breakout_history(prepared, symbol)
        histories.append(history)
        latest.append(history.iloc[-1])

    all_history = pd.concat(histories, ignore_index=True)
    raw_events = all_history[all_history["volume_confirmed"]].copy()
    events = independent_breakouts(raw_events, scanner.config.trade_evaluation_horizon)
    resolved = events[events["trade_outcome"].isin(["TARGET", "STOP"])]
    targets = int((resolved["trade_outcome"] == "TARGET").sum())
    low, high = scanner._wilson_interval(targets, len(resolved))
    summary = {
        "section": "10_year_high_breakouts",
        "data_policy": "local_only_maximum_10_years",
        "symbols": sorted(frames),
        "start": str(min(frame["timestamp"].min() for frame in frames.values())),
        "end": str(max(frame["timestamp"].max() for frame in frames.values())),
        "confirmed_independent_events": int(len(events)),
        "resolved_events": int(len(resolved)),
        "targets": targets,
        "stops": int((resolved["trade_outcome"] == "STOP").sum()),
        "target_rate": targets / len(resolved) if len(resolved) else None,
        "target_rate_ci_95_low": low,
        "target_rate_ci_95_high": high,
        "note": "10Y/stored-history high, not guaranteed lifetime ATH.",
    }
    rows = pd.DataFrame(latest).sort_values(
        ["volume_confirmed", "distance_from_high_pct"], ascending=[False, False]
    )
    report = write_report(rows, summary, config, "section_10_year_high_breakouts")
    print(rows.to_string(index=False))
    print(f"\n[REPORT] {report}")


if __name__ == "__main__":
    main()
