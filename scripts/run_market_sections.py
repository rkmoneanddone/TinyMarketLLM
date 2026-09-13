from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_market_scanner import load_configuration, load_frames, scanner_from, write_report
from src.tiny_market_llm.scanner.market_sections import (
    high_breakout_history,
    independent_breakouts,
    next_day_setup_events,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline TinyMarketLLM market sections")
    parser.add_argument("--section", choices=("high-breakouts", "next-day"), default="high-breakouts")
    parser.add_argument("--symbol", choices=("GRASIM", "RELIANCE", "TCS"))
    args = parser.parse_args()

    config = load_configuration()
    frames = load_frames(config)
    if args.symbol:
        frames = {args.symbol: frames[args.symbol]}
    scanner = scanner_from(config)

    if args.section == "next-day":
        run_next_day_section(config, frames, scanner)
        return

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


def run_next_day_section(config: dict, frames: dict[str, pd.DataFrame], scanner) -> None:
    events = []
    latest = []
    for symbol, frame in frames.items():
        prepared = scanner.prepare(frame)
        events.append(next_day_setup_events(
            prepared, symbol, scanner.config.eligible_setups,
            scanner.config.trade_evaluation_horizon,
        ))
        current = prepared.iloc[-1]
        latest.append({
            "timestamp": current["timestamp"], "symbol": symbol, "close": current["close"],
            "setup": current["setup"], "timeframe": "NEXT_DAY",
        })
    all_events = pd.concat(events, ignore_index=True)
    cutoff = pd.Timestamp("2024-01-01", tz="UTC")
    metric_rows = []
    approved = []
    for setup_name, group in all_events.groupby("setup_name"):
        record = {"setup_name": setup_name}
        credible = True
        for label, sample in (
            ("discovery", group[group["timestamp"] < cutoff]),
            ("unseen", group[group["timestamp"] >= cutoff]),
        ):
            wins = int(sample["success_after_cost_buffer"].sum())
            low, high = scanner._wilson_interval(wins, len(sample))
            record.update({
                f"{label}_events": int(len(sample)),
                f"{label}_success_rate": wins / len(sample) if len(sample) else None,
                f"{label}_ci_95_low": low,
                f"{label}_average_move_pct": float(sample["future_move_1_pct"].mean()) if len(sample) else None,
            })
            credible &= len(sample) >= 20 and low is not None and low > 0.5
        record["approved_for_next_day"] = credible
        if credible:
            approved.append(setup_name)
        metric_rows.append(record)

    latest_rows = pd.DataFrame(latest)
    latest_rows["decision"] = latest_rows["setup"].apply(
        lambda value: "BUY" if set(value.split("|")) & set(approved) else "WAIT"
    )
    latest_rows["reason"] = latest_rows.apply(
        lambda row: "Validated next-day setup" if row["decision"] == "BUY"
        else "No setup has credible next-day edge in discovery and unseen periods",
        axis=1,
    )
    summary = {
        "section": "next_day_trade",
        "data_policy": "local_only_maximum_10_years",
        "cost_buffer_pct": 0.1,
        "validation_cutoff": str(cutoff),
        "approved_setups": approved,
        "setup_metrics": metric_rows,
        "note": "WAIT is mandatory when no setup passes both periods.",
    }
    report = write_report(latest_rows, summary, config, "section_next_day_trade")
    print(latest_rows.to_string(index=False))
    print(f"\n[REPORT] {report}")


if __name__ == "__main__":
    main()
