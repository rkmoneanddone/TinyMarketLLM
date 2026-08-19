from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from src.tiny_market_llm.data.market_repository import (
    MarketRepository,
)

from src.tiny_market_llm.features.feature_engine import (
    FeatureEngine,
)

from src.tiny_market_llm.backtest.backtest_engine import (
    BacktestEngine,
)

from scripts.baseline_predictor import (
    BaselinePredictor,
)


def main():

    print("=" * 44)
    print(" TinyMarketLLM - Backtest Engine Test")
    print("=" * 44)
    print()

    repository = MarketRepository()

    print("[1] Loading GRASIM 1D...")
    data = repository.load(
        "GRASIM",
        "1D",
    )

    print(
        f"[OK] Candles: {len(data)}"
    )

    print()
    print("[2] Calculating features...")

    feature_engine = FeatureEngine()

    features = feature_engine.calculate(
        data
    )

    print(
        f"[OK] Feature rows: {len(features)}"
    )

    print()
    print("[3] Creating baseline predictor...")

    predictor = BaselinePredictor()

    print("[OK] Predictor ready.")

    print()
    print("[4] Running walk-forward backtest...")

    engine = BacktestEngine(
        evaluation_horizon=10,
        sustain_bars=3,
        move_tolerance_pct=0.5,
    )

    results = engine.run(
        features,
        predictor,
    )

    print(
        f"[OK] Evaluation records: "
        f"{len(results)}"
    )

    if results.empty:
        raise RuntimeError(
            "Backtest returned no results."
        )

    print()
    print("[5] Checking result columns...")

    required = [
        "decision_timestamp",
        "predicted_direction",
        "predicted_move_pct",
        "predicted_duration",
        "prediction_confidence",
        "actual_direction",
        "actual_move_pct",
        "actual_duration",
        "max_favorable_move_pct",
        "max_adverse_move_pct",
        "sustained",
        "passed",
    ]

    missing = [
        column
        for column in required
        if column not in results.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing result columns: {missing}"
        )

    print("[OK] All result columns present.")

    print()
    print("[6] Result summary...")

    total = len(results)

    passed = int(
        results["passed"].sum()
    )

    failed = total - passed

    success_rate = (
        passed / total * 100
    )

    print(
        f"Total tests   : {total}"
    )

    print(
        f"Passed        : {passed}"
    )

    print(
        f"Failed        : {failed}"
    )

    print(
        f"Success rate  : {success_rate:.2f}%"
    )

    print()
    print("========== LAST 15 RESULTS ==========")

    columns = [
        "decision_timestamp",
        "predicted_direction",
        "predicted_move_pct",
        "actual_direction",
        "actual_move_pct",
        "max_favorable_move_pct",
        "max_adverse_move_pct",
        "sustained",
        "passed",
    ]

    print(
        results[columns]
        .tail(15)
        .to_string(index=False)
    )

    print()
    print("========== RESULT ==========")
    print("PASS")


if __name__ == "__main__":
    main()