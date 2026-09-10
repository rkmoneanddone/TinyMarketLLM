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

from src.tiny_market_llm.backtest.backtest_engine_v2 import (
    BacktestEngineV2,
)

from src.tiny_market_llm.outcomes.outcome_engine_v2 import (
    OutcomeEngineV2,
)

from scripts.baseline_predictor import (
    BaselinePredictor,
)


def main():

    print("=" * 60)
    print(" TinyMarketLLM - Build 1-Candle V2 Dataset")
    print("=" * 60)
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
    print("[3] Running 1-candle backtest...")

    predictor = BaselinePredictor()

    backtest = BacktestEngineV2(
        evaluation_horizon=1,
        sustain_bars=1,
        move_tolerance_pct=0.5,
    )

    results = backtest.run(
        features,
        predictor,
    )

    print(
        f"[OK] Backtest records: {len(results)}"
    )

    if results.empty:
        raise RuntimeError(
            "1-candle backtest produced no records."
        )

    print()
    print("[4] Verifying horizon...")

    horizons = sorted(
        results["horizon_candles"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    print(
        f"[OK] Horizons found: {horizons}"
    )

    if horizons != [1]:
        raise RuntimeError(
            f"Expected only horizon 1, got {horizons}"
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
    print("[6] Verifying learning horizon...")

    dataset_horizons = sorted(
        dataset["horizon_candles"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    print(
        f"[OK] Dataset horizons: {dataset_horizons}"
    )

    if dataset_horizons != [1]:
        raise RuntimeError(
            f"Expected dataset horizon [1], "
            f"got {dataset_horizons}"
        )

    print()
    print("[7] Checking required fields...")

    required = [
        "decision_timestamp",
        "horizon_candles",

        "structure",
        "price_direction",

        "move_3_pct",
        "move_5_pct",
        "range_5_pct",

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
            f"Missing learning fields: {missing}"
        )

    print(
        "[OK] All required fields present."
    )

    print()
    print("[8] Checking decision timestamps...")

    if dataset[
        "decision_timestamp"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate decision timestamps."
        )

    print(
        "[OK] One record per decision timestamp."
    )

    print()
    print("[9] Outcome statistics...")

    print(
        dataset[
            "label_direction"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        f"Average 1-candle move     : "
        f"{dataset['label_move_pct'].mean():.3f}%"
    )

    print(
        f"Average favorable move   : "
        f"{dataset['label_favorable_move_pct'].mean():.3f}%"
    )

    print(
        f"Average adverse move     : "
        f"{dataset['label_adverse_move_pct'].mean():.3f}%"
    )

    print(
        f"Average duration          : "
        f"{dataset['label_duration'].mean():.3f}"
    )

    print()
    print("[10] Saving 1-candle dataset...")

    output = (
        Path("data")
        / "learning"
        / "GRASIM_1D_1C_training_v2.parquet"
    )

    engine.save(
        dataset,
        output,
    )

    print(
        f"[OK] Saved: {output}"
    )

    print()
    print("========== SAMPLE RECORDS ==========")

    columns = [
        "decision_timestamp",
        "horizon_candles",
        "structure",
        "price_direction",
        "rsi",
        "rsi_direction",
        "rsi_zone",
        "label_direction",
        "label_move_pct",
        "label_favorable_move_pct",
        "label_adverse_move_pct",
        "label_duration",
        "label_sustained",
    ]

    print(
        dataset[
            columns
        ]
        .tail(15)
        .to_string(index=False)
    )

    print()
    print("========== RESULT ==========")
    print("1-CANDLE V2 DATASET BUILD PASS")


if __name__ == "__main__":
    main()