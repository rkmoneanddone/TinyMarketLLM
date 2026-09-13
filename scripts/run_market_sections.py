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
    swing_setup_events,
    ema_alignment_history,
    resample_ohlc,
    rsi_reversal_history,
    breakout_pullback_history,
    morning_star_history,
)


SWING_HORIZONS = {
    "1_WEEK": 5, "2_WEEKS": 10, "3_WEEKS": 15,
    "1_MONTH": 21, "2_MONTHS": 42, "3_MONTHS": 63,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline TinyMarketLLM market sections")
    parser.add_argument(
        "--section", choices=(
            "high-breakouts", "next-day", "swing", "ema-alignment", "rsi-reversal", "breakout-pullback",
            "morning-star",
        ),
        default="high-breakouts",
    )
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
    if args.section == "swing":
        run_swing_section(config, frames, scanner)
        return
    if args.section == "ema-alignment":
        run_ema_alignment_section(config, frames, scanner)
        return
    if args.section == "rsi-reversal":
        run_rsi_reversal_section(config, frames, scanner)
        return
    if args.section == "breakout-pullback":
        run_breakout_pullback_section(config, frames, scanner)
        return
    if args.section == "morning-star":
        run_morning_star_section(config, frames, scanner)
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


def run_swing_section(config: dict, frames: dict[str, pd.DataFrame], scanner) -> None:
    events = []
    latest_setups = {}
    latest_timestamp = None
    for symbol, frame in frames.items():
        prepared = scanner.prepare(frame)
        events.append(swing_setup_events(
            prepared, symbol, scanner.config.eligible_setups, SWING_HORIZONS,
        ))
        latest_setups[symbol] = str(prepared.iloc[-1]["setup"])
        latest_timestamp = prepared.iloc[-1]["timestamp"]
    all_events = pd.concat(events, ignore_index=True)
    cutoff = pd.Timestamp("2024-01-01", tz="UTC")
    metrics = []
    approved: set[tuple[str, str]] = set()
    for (setup_name, horizon), group in all_events.groupby(["setup_name", "horizon"]):
        record = {"setup_name": setup_name, "horizon": horizon}
        credible = True
        for label, sample in (
            ("discovery", group[group["timestamp"] < cutoff]),
            ("unseen", group[group["timestamp"] >= cutoff]),
        ):
            wins = int(sample["success_after_cost_buffer"].sum())
            low, _ = scanner._wilson_interval(wins, len(sample))
            mean_move = float(sample["future_move_pct"].mean()) if len(sample) else None
            record.update({
                f"{label}_events": int(len(sample)),
                f"{label}_success_rate": wins / len(sample) if len(sample) else None,
                f"{label}_ci_95_low": low,
                f"{label}_average_move_pct": mean_move,
            })
            credible &= len(sample) >= 20 and low is not None and low > 0.5 and mean_move is not None and mean_move > 0
        record["approved"] = credible
        if credible:
            approved.add((setup_name, horizon))
        metrics.append(record)

    rows = []
    for symbol, setup_value in latest_setups.items():
        present = set(setup_value.split("|"))
        for horizon in SWING_HORIZONS:
            matched = sorted(name for name in present if (name, horizon) in approved)
            rows.append({
                "timestamp": latest_timestamp, "symbol": symbol, "timeframe": horizon,
                "setup": setup_value, "decision": "BUY" if matched else "WAIT",
                "validated_setup": "|".join(matched) if matched else "NONE",
                "reason": "Validated swing setup" if matched else "No setup passes discovery and unseen validation",
            })
    output = pd.DataFrame(rows)
    summary = {
        "section": "swing_trade",
        "data_policy": "local_only_maximum_10_years",
        "horizons": SWING_HORIZONS,
        "cost_buffer_pct": 0.2,
        "validation_cutoff": str(cutoff),
        "approved_setup_horizons": [f"{setup}:{horizon}" for setup, horizon in sorted(approved)],
        "metrics": metrics,
    }
    report = write_report(output, summary, config, "section_swing_trade")
    print(output.to_string(index=False))
    print(f"\n[REPORT] {report}")


def run_ema_alignment_section(config: dict, frames: dict[str, pd.DataFrame], scanner) -> None:
    histories = []
    latest = []
    for symbol, frame in frames.items():
        history = ema_alignment_history(scanner.prepare(frame), symbol)
        histories.append(history)
        latest.append(history.iloc[-1])
    all_history = pd.concat(histories, ignore_index=True)
    cutoff = pd.Timestamp("2024-01-01", tz="UTC")
    metrics = []
    approved = []
    for setup_name, group in all_history[all_history["setup"] != "NONE"].groupby("setup"):
        record = {"setup": setup_name}
        credible = True
        for label, sample in (
            ("discovery", group[group["timestamp"] < cutoff]),
            ("unseen", group[group["timestamp"] >= cutoff]),
        ):
            resolved = sample[sample["trade_outcome"].isin(["TARGET", "STOP"])]
            targets = int((resolved["trade_outcome"] == "TARGET").sum())
            low, _ = scanner._wilson_interval(targets, len(resolved))
            record.update({
                f"{label}_events": int(len(sample)), f"{label}_resolved": int(len(resolved)),
                f"{label}_target_rate": targets / len(resolved) if len(resolved) else None,
                f"{label}_ci_95_low": low,
            })
            credible &= len(resolved) >= 20 and low is not None and low > 3 / 7
        record["approved"] = credible
        if credible:
            approved.append(setup_name)
        metrics.append(record)
    rows = pd.DataFrame(latest)[[
        "timestamp", "symbol", "close", "ema_9", "ema_21", "ema_50", "ema_200",
        "above_all_emas", "below_all_emas", "bull_stack", "bear_stack", "setup",
    ]]
    rows["decision"] = rows["setup"].apply(lambda setup: "BUY" if setup in approved and setup.endswith("LONG") else "SELL" if setup in approved else "WAIT")
    rows["reason"] = rows["decision"].map({"BUY": "Validated EMA alignment", "SELL": "Validated EMA alignment"}).fillna(
        "No new validated EMA 9/21/50/200 entry"
    )
    summary = {
        "section": "ema_9_21_50_200_alignment", "data_policy": "local_only_maximum_10_years",
        "validation_cutoff": str(cutoff), "approved_setups": approved, "metrics": metrics,
        "note": "Above/below-all status is informational; only a validated transition can trade.",
    }
    report = write_report(rows, summary, config, "section_ema_alignment")
    print(rows.to_string(index=False))
    print(f"\n[REPORT] {report}")


def run_rsi_reversal_section(config: dict, frames: dict[str, pd.DataFrame], scanner) -> None:
    histories = []
    latest = []
    for symbol, daily in frames.items():
        timeframe_frames = {
            "DAILY": daily,
            "WEEKLY": resample_ohlc(daily, "W-FRI"),
            "MONTHLY": resample_ohlc(daily, "ME"),
        }
        for timeframe, frame in timeframe_frames.items():
            history = rsi_reversal_history(scanner.prepare(frame), symbol, timeframe)
            histories.append(history)
            latest.append(history.iloc[-1])
    all_history = pd.concat(histories, ignore_index=True)
    cutoff = pd.Timestamp("2024-01-01", tz="UTC")
    metrics = []
    approved = []
    for (timeframe, setup_name), group in all_history[all_history["setup"] != "NONE"].groupby(["timeframe", "setup"]):
        record = {"timeframe": timeframe, "setup": setup_name}
        credible = True
        for label, sample in (
            ("discovery", group[group["timestamp"] < cutoff]),
            ("unseen", group[group["timestamp"] >= cutoff]),
        ):
            resolved = sample[sample["trade_outcome"].isin(["TARGET", "STOP"])]
            targets = int((resolved["trade_outcome"] == "TARGET").sum())
            low, _ = scanner._wilson_interval(targets, len(resolved))
            record.update({
                f"{label}_resolved": int(len(resolved)),
                f"{label}_target_rate": targets / len(resolved) if len(resolved) else None,
                f"{label}_ci_95_low": low,
            })
            credible &= len(resolved) >= 20 and low is not None and low > 3 / 7
        record["approved"] = credible
        if credible:
            approved.append((timeframe, setup_name))
        metrics.append(record)
    rows = pd.DataFrame(latest)[["timestamp", "symbol", "timeframe", "close", "rsi_14", "below_25", "above_80", "setup"]]
    rows["decision"] = rows.apply(
        lambda row: "BUY" if (row["timeframe"], row["setup"]) in approved and row["setup"].endswith("LONG")
        else "SELL" if (row["timeframe"], row["setup"]) in approved else "WAIT", axis=1,
    )
    rows["reason"] = rows["decision"].apply(
        lambda value: "Validated RSI reversal" if value != "WAIT" else "No new validated RSI reversal"
    )
    summary = {
        "section": "multi_timeframe_rsi_reversal", "data_policy": "local_only_maximum_10_years",
        "validation_cutoff": str(cutoff),
        "approved": [f"{timeframe}:{setup}" for timeframe, setup in approved], "metrics": metrics,
        "note": "Extreme RSI is status only; entry requires a confirmed threshold reclaim/rejection.",
    }
    report = write_report(rows, summary, config, "section_rsi_reversal")
    print(rows.to_string(index=False))
    print(f"\n[REPORT] {report}")


def run_breakout_pullback_section(config: dict, frames: dict[str, pd.DataFrame], scanner) -> None:
    events = []
    latest_timestamp = None
    latest_close = {}
    for symbol, frame in frames.items():
        prepared = scanner.prepare(frame)
        events.append(breakout_pullback_history(prepared, symbol))
        latest_timestamp = prepared.iloc[-1]["timestamp"]
        latest_close[symbol] = float(prepared.iloc[-1]["close"])
    all_events = pd.concat(events, ignore_index=True)
    cutoff = pd.Timestamp("2024-01-01", tz="UTC")
    metrics = []
    approved = []
    for setup_name, group in all_events.groupby("setup"):
        record = {"setup": setup_name}
        credible = True
        for label, sample in (
            ("discovery", group[group["timestamp"] < cutoff]),
            ("unseen", group[group["timestamp"] >= cutoff]),
        ):
            resolved = sample[sample["trade_outcome"].isin(["TARGET", "STOP"])]
            targets = int((resolved["trade_outcome"] == "TARGET").sum())
            low, _ = scanner._wilson_interval(targets, len(resolved))
            record.update({
                f"{label}_resolved": int(len(resolved)),
                f"{label}_target_rate": targets / len(resolved) if len(resolved) else None,
                f"{label}_ci_95_low": low,
            })
            credible &= len(resolved) >= 20 and low is not None and low > 3 / 7
        record["approved"] = credible
        if credible:
            approved.append(setup_name)
        metrics.append(record)
    recent_cutoff = latest_timestamp - pd.Timedelta(days=15)
    recent = all_events[all_events["timestamp"] >= recent_cutoff]
    rows = []
    for symbol, close in latest_close.items():
        matches = recent[(recent["symbol"] == symbol) & recent["setup"].isin(approved)]
        rows.append({
            "timestamp": latest_timestamp, "symbol": symbol, "close": close,
            "setup": "|".join(sorted(matches["setup"].unique())) if len(matches) else "NONE",
            "decision": "BUY" if len(matches) else "WAIT",
            "reason": "Validated recent breakout pullback" if len(matches) else "No recent validated OB/FVG pullback",
        })
    output = pd.DataFrame(rows)
    summary = {
        "section": "breakout_order_block_fvg_pullback",
        "data_policy": "local_only_maximum_10_years", "validation_cutoff": str(cutoff),
        "approved_setups": approved, "metrics": metrics,
        "definitions": {"fvg": "low[t] > high[t-2]", "order_block": "last bearish candle within 5 bars before breakout", "zone_lifetime": 10},
    }
    report = write_report(output, summary, config, "section_breakout_pullback")
    print(output.to_string(index=False))
    print(f"\n[REPORT] {report}")


def run_morning_star_section(config: dict, frames: dict[str, pd.DataFrame], scanner) -> None:
    histories = []
    latest = []
    for symbol, frame in frames.items():
        history = morning_star_history(scanner.prepare(frame), symbol)
        histories.append(history)
        latest.append(history.iloc[-1])
    all_history = pd.concat(histories, ignore_index=True)
    events = all_history[all_history["setup"] == "MORNING_STAR_AT_SUPPORT"]
    cutoff = pd.Timestamp("2024-01-01", tz="UTC")
    metric = {"setup": "MORNING_STAR_AT_SUPPORT"}
    approved = True
    for label, sample in (
        ("discovery", events[events["timestamp"] < cutoff]),
        ("unseen", events[events["timestamp"] >= cutoff]),
    ):
        resolved = sample[sample["trade_outcome"].isin(["TARGET", "STOP"])]
        targets = int((resolved["trade_outcome"] == "TARGET").sum())
        low, _ = scanner._wilson_interval(targets, len(resolved))
        metric.update({
            f"{label}_resolved": int(len(resolved)),
            f"{label}_target_rate": targets / len(resolved) if len(resolved) else None,
            f"{label}_ci_95_low": low,
        })
        approved &= len(resolved) >= 20 and low is not None and low > 3 / 7
    metric["approved"] = approved
    rows = pd.DataFrame(latest)[["timestamp", "symbol", "close", "prior_support", "pattern_low", "setup"]]
    rows["decision"] = rows["setup"].apply(lambda setup: "BUY" if approved and setup == "MORNING_STAR_AT_SUPPORT" else "WAIT")
    rows["reason"] = rows["decision"].apply(
        lambda value: "Validated Morning Star at support" if value == "BUY" else "No new validated Morning Star at support"
    )
    summary = {
        "section": "morning_star_at_support", "data_policy": "local_only_maximum_10_years",
        "validation_cutoff": str(cutoff), "metric": metric,
        "note": "Three-candle shape, ATR size, and prior support are all required.",
    }
    report = write_report(rows, summary, config, "section_morning_star")
    print(rows.to_string(index=False))
    print(f"\n[REPORT] {report}")


if __name__ == "__main__":
    main()
