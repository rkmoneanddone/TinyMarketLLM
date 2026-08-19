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

from src.tiny_market_llm.features.feature_engine_v2 import (
    FeatureEngineV2,
)

from src.tiny_market_llm.backtest.backtest_engine import (
    BacktestEngine,
)

from src.tiny_market_llm.outcomes.outcome_engine_v2 import (
    OutcomeEngineV2,
)

from scripts.baseline_predictor import (
    BaselinePredictor,
)


def main():

    print("=" * 48)
    print(" TinyMarketLLM - Outcome Engine V2 Test")
    print("=" * 48)
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
    print("[2] Calculating V2 features...")

    feature_engine = FeatureEngineV2()

    features = feature_engine.calculate(
        data
    )

    print(
        f"[OK] Feature rows: {len(features)}"
    )

    print()
    print("[3] Confirming MA is absent...")

    ma_columns = [
        column
        for column in features.columns
        if "ma" in column.lower()
    ]

    if ma_columns:
        raise RuntimeError(
            f"MA columns detected: {ma_columns}"
        )

    print("[OK] No MA features.")

    print()
    print("[4] Running backtest...")

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
    print("[5] Building V2 learning dataset...")

    engine = OutcomeEngineV2()

    dataset = engine.build(
        features,
        results,
    )

    print(
        f"[OK] Learning records: {len(dataset)}"
    )

    print()
    print("[6] Checking V2 columns...")

    forbidden = [
    column
    for column in dataset.columns
    if column.lower() in {
        "ma",
        "ma_direction",
        "moving_average",
        "moving_average_direction",
    }
    ]

    if forbidden:
        raise RuntimeError(
            f"MA fields found in V2 dataset: "
            f"{forbidden}"
        )

    print(
        "[OK] Dataset contains no MA fields."
    )

    required = [
        "structure",
        "move_3_pct",
        "move_5_pct",
        "range_5_pct",
        "pullback_from_high_pct",
        "recovery_from_low_pct",

        "rsi",
        "rsi_direction",
        "rsi_zone",
        "rsi_change_3",

        "label_direction",
        "label_move_pct",
        "label_favorable_move_pct",
        "label_adverse_move_pct",
        "label_duration",
        "label_sustained",
    ]

    missing = [
        column
        for column in required
        if column not in dataset.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing V2 learning fields: {missing}"
        )

    print(
        "[OK] All V2 learning fields present."
    )

    print()
    print("[7] Checking alignment...")

    if len(dataset) != len(results):
        raise RuntimeError(
            "Dataset/result count mismatch."
        )

    if dataset[
        "decision_timestamp"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate decision timestamps."
        )

    print(
        "[OK] One learning record per decision."
    )

    print()
    print("[8] Outcome statistics...")

    print(
        dataset[
            "label_direction"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        f"Average move: "
        f"{dataset['label_move_pct'].mean():.3f}%"
    )

    print(
        f"Average favorable: "
        f"{dataset['label_favorable_move_pct'].mean():.3f}%"
    )

    print(
        f"Average adverse: "
        f"{dataset['label_adverse_move_pct'].mean():.3f}%"
    )

    print(
        f"Sustained: "
        f"{dataset['label_sustained'].sum()}"
    )

    print()
    print("[9] Saving V2 dataset...")

    output = Path(
        "data"
    ) / "learning" / "GRASIM_1D_training_v2.parquet"

    engine.save(
        dataset,
        output,
    )

    print(
        f"[OK] Saved: {output}"
    )

    print()
    print("========== SAMPLE V2 RECORDS ==========")

    columns = [
        "decision_timestamp",
        "structure",
        "price_direction",
        "move_3_pct",
        "move_5_pct",
        "rsi",
        "rsi_direction",
        "rsi_zone",
        "label_direction",
        "label_move_pct",
        "label_favorable_move_pct",
        "label_adverse_move_pct",
        "label_sustained",
    ]

    print(
        dataset[columns]
        .tail(15)
        .to_string(index=False)
    )

    print()
    print("========== RESULT ==========")
    print("PASS")


if __name__ == "__main__":
    main()