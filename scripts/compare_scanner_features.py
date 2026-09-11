from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_market_scanner import load_configuration, load_frames, scanner_from, write_report


def metric_rows(feature_set: str, summary: dict) -> list[dict]:
    rows = [{
        "feature_set": feature_set,
        "horizon": summary["decision_horizon_candles"],
        "scope": "primary",
        "rows": summary["unseen_rows"],
        "accuracy": summary["accuracy"],
        "accuracy_ci_95_low": summary["accuracy_ci_95_low"],
        "balanced_accuracy": summary["balanced_accuracy"],
        "majority_baseline": summary["majority_baseline_accuracy"],
        "edge_vs_baseline": summary["accuracy"] - summary["majority_baseline_accuracy"],
        "directional_rows": summary["directional_rows"],
        "resolved_trades": summary["resolved_trades"],
        "model_status": summary["model_status"],
    }]
    for horizon, metrics in summary["horizon_metrics"].items():
        rows.append({
            "feature_set": feature_set,
            "horizon": int(horizon),
            "scope": "horizon",
            "rows": metrics["rows"],
            "accuracy": metrics["accuracy"],
            "accuracy_ci_95_low": metrics["accuracy_ci_95_low"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "majority_baseline": metrics["majority_baseline_accuracy"],
            "edge_vs_baseline": metrics["accuracy"] - metrics["majority_baseline_accuracy"],
            "directional_rows": None,
            "resolved_trades": None,
            "model_status": None,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare core and chart-derived scanner features")
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--training-years", type=int, default=3)
    parser.add_argument("--symbol", choices=("GRASIM", "RELIANCE", "TCS"))
    args = parser.parse_args()

    config = load_configuration()
    frames = load_frames(config)
    if args.symbol:
        frames = {args.symbol: frames[args.symbol]}

    summaries = {}
    rows = []
    for feature_set in ("core", "chart"):
        print(f"[RUN] {feature_set.upper()} features")
        scanner = scanner_from(config, feature_set)
        _, summary = scanner.daily_walk_forward_test(
            frames,
            test_start=args.test_start,
            test_end=args.test_end,
            training_years=args.training_years,
        )
        summaries[feature_set] = summary
        rows.extend(metric_rows(feature_set, summary))

    comparison = pd.DataFrame(rows)
    core = summaries["core"]
    chart = summaries["chart"]
    review = {
        "mode": "feature_set_comparison",
        "test_start": args.test_start,
        "test_end": args.test_end,
        "training_years": args.training_years,
        "chart_improves_accuracy": chart["accuracy"] > core["accuracy"],
        "chart_improves_balanced_accuracy": chart["balanced_accuracy"] > core["balanced_accuracy"],
        "automatic_promotion": False,
        "reason": "Candidate features remain research-only until repeated walk-forward evidence clears credibility gates.",
        "core": core,
        "chart": chart,
    }
    name = f"feature_comparison_{args.training_years}y_{args.test_start}_to_{args.test_end}"
    report = write_report(comparison, review, config, name)
    print(comparison.to_string(index=False))
    print(f"\n[REPORT] {report}")


if __name__ == "__main__":
    main()
