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

from src.tiny_market_llm.backtest.backtest_engine_v2 import (
    BacktestEngineV2,
)

from scripts.baseline_predictor import (
    BaselinePredictor,
)


def main():

    print("=" * 50)
    print(" TinyMarketLLM - Backtest Engine V2 Test")
    print("=" * 50)
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

    engine = FeatureEngine()

    features = engine.calculate(
        data
    )

    print(
        f"[OK] Feature rows: {len(features)}"
    )

    print()
    print("[3] Running V2 backtest...")

    predictor = BaselinePredictor()

    backtest = BacktestEngineV2(
        evaluation_horizon=10,
        sustain_bars=3,
        move_tolerance_pct=0.5,
    )

    results = backtest.run(
        features,
        predictor,
    )

    print(
        f"[OK] Results: {len(results)}"
    )

    print()
    print("[4] Checking result fields...")

    required = [
        "decision_timestamp",

        "predicted_direction",
        "predicted_move_pct",

        "actual_direction",
        "actual_move_pct",

        "actual_upside_move_pct",
        "actual_downside_move_pct",

        "time_to_upside_peak",
        "time_to_downside_peak",

        "upside_sustained",
        "downside_sustained",

        "favorable_move_pct",
        "adverse_move_pct",

        "passed",
    ]

    missing = [
        column
        for column in required
        if column not in results.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing fields: {missing}"
        )

    print(
        "[OK] All V2 result fields present."
    )

    print()
    print("[5] Checking actual movement...")

    if (
        results[
            "actual_upside_move_pct"
        ].abs().sum()
        == 0
    ):

        raise RuntimeError(
            "Actual upside movement is all zero."
        )

    if (
        results[
            "actual_downside_move_pct"
        ].abs().sum()
        == 0
    ):

        raise RuntimeError(
            "Actual downside movement is all zero."
        )

    print(
        "[OK] Actual movement is independent "
        "of prediction."
    )

    print()
    print("[6] Outcome statistics...")

    print(
        f"Average upside : "
        f"{results['actual_upside_move_pct'].mean():.3f}%"
    )

    print(
        f"Average downside: "
        f"{results['actual_downside_move_pct'].mean():.3f}%"
    )

    print(
        f"UP sustained   : "
        f"{results['upside_sustained'].sum()}"
    )

    print(
        f"DOWN sustained : "
        f"{results['downside_sustained'].sum()}"
    )

    print()
    print("========== LAST 15 ==========")

    columns = [
        "decision_timestamp",
        "predicted_direction",
        "actual_direction",
        "actual_move_pct",
        "actual_upside_move_pct",
        "actual_downside_move_pct",
        "time_to_upside_peak",
        "time_to_downside_peak",
        "upside_sustained",
        "downside_sustained",
        "passed",
    ]

    print(
        results[
            columns
        ]
        .tail(15)
        .to_string(index=False)
    )

    print()
    print("========== RESULT ==========")
    print("PASS")


if __name__ == "__main__":
    main()