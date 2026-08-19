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

from src.tiny_market_llm.outcomes.outcome_engine import (
    OutcomeEngine,
)

from scripts.baseline_predictor import (
    BaselinePredictor,
)


def main():

    print("=" * 46)
    print(" TinyMarketLLM - Outcome Engine Test")
    print("=" * 46)
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
    print("[3] Running backtest...")

    predictor = BaselinePredictor()

    backtest = BacktestEngine(
        evaluation_horizon=10,
        sustain_bars=3,
        move_tolerance_pct=0.5,
    )

    results = backtest.run(
        features,
        predictor,
    )

    print(
        f"[OK] Backtest records: {len(results)}"
    )

    print()
    print("[4] Building learning dataset...")

    outcome_engine = OutcomeEngine()

    dataset = outcome_engine.build(
        features,
        results,
    )

    print(
        f"[OK] Learning records: {len(dataset)}"
    )

    print()
    print("[5] Validating learning records...")

    required = [
        "schema_version",
        "decision_timestamp",

        "structure",
        "price_direction",
        "rsi",
        "rsi_direction",
        "ma_direction",
        "price_vs_ma",

        "predicted_direction",

        "label_direction",
        "label_move_pct",
        "label_favorable_move_pct",
        "label_adverse_move_pct",
        "label_duration",
        "label_sustained",
        "label_target_reached",
        "label_passed",
    ]

    missing = [
        column
        for column in required
        if column not in dataset.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing learning columns: {missing}"
        )

    print("[OK] All learning fields present.")

    print()
    print("[6] Checking record alignment...")

    if len(dataset) != len(results):
        raise RuntimeError(
            "Learning record count does not match "
            "backtest result count."
        )

    if (
        dataset["decision_timestamp"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate decision timestamps found."
        )

    print("[OK] One learning record per decision.")

    print()
    print("[7] Learning labels...")

    print(
        dataset["label_direction"]
        .value_counts()
        .to_string()
    )

    print()
    print("[8] Label statistics...")

    print(
        f"Average move: "
        f"{dataset['label_move_pct'].mean():.3f}%"
    )

    print(
        f"Average favorable move: "
        f"{dataset['label_favorable_move_pct'].mean():.3f}%"
    )

    print(
        f"Average adverse move: "
        f"{dataset['label_adverse_move_pct'].mean():.3f}%"
    )

    print(
        f"Sustained: "
        f"{dataset['label_sustained'].sum()}"
    )

    print(
        f"Target reached: "
        f"{dataset['label_target_reached'].sum()}"
    )

    print(
        f"Passed: "
        f"{dataset['label_passed'].sum()}"
    )

    print()
    print("========== SAMPLE LEARNING RECORDS ==========")

    columns = [
        "decision_timestamp",
        "structure",
        "rsi",
        "rsi_direction",
        "ma_direction",
        "price_vs_ma",
        "predicted_direction",
        "label_direction",
        "label_move_pct",
        "label_favorable_move_pct",
        "label_adverse_move_pct",
        "label_sustained",
        "label_passed",
    ]

    print(
        dataset[columns]
        .tail(15)
        .to_string(index=False)
    )

    print()
    print("[9] Saving temporary learning dataset...")

    output = Path(
        "data"
    ) / "learning" / "GRASIM_1D_training.parquet"

    outcome_engine.save(
        dataset,
        output,
    )

    print(
        f"[OK] Saved: {output}"
    )

    print()
    print("========== RESULT ==========")
    print("PASS")


if __name__ == "__main__":
    main()