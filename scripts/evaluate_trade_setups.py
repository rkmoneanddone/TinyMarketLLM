from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_market_scanner import load_configuration, load_frames, scanner_from, write_report


def independent_events(events: pd.DataFrame, cooldown: int) -> pd.DataFrame:
    kept = []
    for (_, _), group in events.sort_values("bar_index").groupby(["symbol", "setup_name"]):
        last_bar = -10**9
        for row in group.itertuples(index=False):
            if row.bar_index - last_bar >= cooldown:
                kept.append(row._asdict())
                last_bar = row.bar_index
    return pd.DataFrame(kept, columns=events.columns)


def summarise(group: pd.DataFrame, target_multiple: float, stop_multiple: float) -> dict:
    resolved = group[group["outcome"].isin(["TARGET", "STOP"])]
    targets = int((resolved["outcome"] == "TARGET").sum())
    stops = int((resolved["outcome"] == "STOP").sum())
    rate = targets / len(resolved) if len(resolved) else None
    scanner = group.attrs["scanner"]
    low, high = scanner._wilson_interval(targets, len(resolved))
    breakeven = stop_multiple / (target_multiple + stop_multiple)
    expectancy = rate * target_multiple - (1 - rate) * stop_multiple if rate is not None else None
    return {
        "occurrences": int(len(group)),
        "symbols": int(group["symbol"].nunique()),
        "resolved": int(len(resolved)),
        "targets": targets,
        "stops": stops,
        "ambiguous": int((group["outcome"] == "AMBIGUOUS").sum()),
        "neither": int((group["outcome"] == "NEITHER").sum()),
        "target_rate": rate,
        "target_rate_ci_95_low": low,
        "target_rate_ci_95_high": high,
        "breakeven_target_rate": breakeven,
        "expectancy_atr_before_costs": expectancy,
        "credible_edge": bool(len(resolved) >= 20 and group["symbol"].nunique() >= 2 and low is not None and low > breakeven),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast model-free evaluation of selective trade setups")
    parser.add_argument("--start", help="First setup date; default is earliest available")
    parser.add_argument("--end", help="Final setup date; default is latest available")
    parser.add_argument("--cooldown", type=int, help="Minimum candles between repeated setup events")
    parser.add_argument("--symbol", choices=("GRASIM", "RELIANCE", "TCS"))
    args = parser.parse_args()

    config = load_configuration()
    frames = load_frames(config)
    if args.symbol:
        frames = {args.symbol: frames[args.symbol]}
    scanner = scanner_from(config, "core")
    cooldown = args.cooldown or scanner.config.trade_evaluation_horizon

    records = []
    for symbol, frame in frames.items():
        prepared = scanner.prepare(frame).reset_index(drop=True)
        prepared["bar_index"] = prepared.index
        prepared["symbol"] = symbol
        if args.start:
            prepared = prepared[prepared["timestamp"] >= scanner._coerce_cutoff(args.start, prepared["timestamp"])]
        if args.end:
            prepared = prepared[prepared["timestamp"] <= scanner._coerce_cutoff(args.end, prepared["timestamp"])]
        prepared = prepared[(prepared["setup"] != "NONE") & prepared["setup_direction"].isin(["UP", "DOWN"])]
        for row in prepared.itertuples(index=False):
            outcome = row.buy_trade_outcome if row.setup_direction == "UP" else row.sell_trade_outcome
            for setup_name in row.setup.split("|"):
                records.append({
                    "timestamp": row.timestamp, "symbol": symbol, "bar_index": row.bar_index,
                    "setup_name": setup_name, "direction": row.setup_direction,
                    "close": row.close, "outcome": outcome,
                })
    if not records:
        raise ValueError("No setup events were found in the requested window.")

    raw = pd.DataFrame(records)
    events = independent_events(raw, cooldown)
    events.attrs["scanner"] = scanner
    aggregate = []
    for setup_name, group in events.groupby("setup_name"):
        group.attrs["scanner"] = scanner
        aggregate.append({"scope": "setup", "setup_name": setup_name, "symbol": "ALL", **summarise(
            group, scanner.config.target_atr_multiple, scanner.config.stop_atr_multiple
        )})
        for symbol, symbol_group in group.groupby("symbol"):
            symbol_group.attrs["scanner"] = scanner
            aggregate.append({"scope": "setup_symbol", "setup_name": setup_name, "symbol": symbol, **summarise(
                symbol_group, scanner.config.target_atr_multiple, scanner.config.stop_atr_multiple
            )})
    rows = pd.DataFrame(aggregate).sort_values(
        ["credible_edge", "expectancy_atr_before_costs", "resolved"], ascending=[False, False, False]
    )
    summary = {
        "mode": "independent_setup_evaluation",
        "start": str(events["timestamp"].min()), "end": str(events["timestamp"].max()),
        "symbols": sorted(events["symbol"].unique()), "cooldown_candles": cooldown,
        "raw_setup_records": int(len(raw)), "independent_setup_records": int(len(events)),
        "target_atr_multiple": scanner.config.target_atr_multiple,
        "stop_atr_multiple": scanner.config.stop_atr_multiple,
        "credible_setups": rows[(rows["scope"] == "setup") & rows["credible_edge"]]["setup_name"].tolist(),
    }
    name = f"setup_evaluation_{args.start or 'earliest'}_to_{args.end or 'latest'}"
    report = write_report(rows, summary, config, name)
    print(rows.to_string(index=False))
    print(f"\n[REPORT] {report}")


if __name__ == "__main__":
    main()
