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


def main():

    print("=" * 46)
    print(" TinyMarketLLM - Feature Engine V2 Test")
    print("=" * 46)
    print()

    repository = MarketRepository()

    print("[1] Loading GRASIM 1D...")

    data = repository.load(
        "GRASIM",
        "1D",
    )

    print(
        f"[OK] Input candles: {len(data)}"
    )

    print()
    print("[2] Calculating Price Action + RSI...")

    engine = FeatureEngineV2()

    features = engine.calculate(data)

    print(
        f"[OK] Output candles: {len(features)}"
    )

    print()
    print("[3] Checking MA is absent...")

    ma_columns = [
        column
        for column in features.columns
        if "ma" in column.lower()
    ]

    if ma_columns:
        raise RuntimeError(
            f"MA-related columns found: {ma_columns}"
        )

    print("[OK] No MA features.")

    print()
    print("[4] Checking required features...")

    required = [
        "structure_event",
        "structure",
        "price_direction",
        "move_3_pct",
        "move_5_pct",
        "range_5_pct",
        "candle_body_pct",
        "pullback_from_high_pct",
        "recovery_from_low_pct",
        "rsi",
        "rsi_direction",
        "rsi_overbought",
        "rsi_oversold",
        "rsi_change_3",
        "rsi_zone",
        "bullish_divergence_candidate",
        "bearish_divergence_candidate",
    ]

    missing = [
        column
        for column in required
        if column not in features.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing features: {missing}"
        )

    print("[OK] All V2 features present.")

    print()
    print("[5] Structure events...")

    print(
        features["structure_event"]
        .value_counts(dropna=True)
        .to_string()
    )

    print()
    print("[6] RSI range...")

    rsi = features["rsi"].dropna()

    print(
        f"[OK] RSI: "
        f"{rsi.min():.2f} -> "
        f"{rsi.max():.2f}"
    )

    print()
    print("========== LAST 15 V2 FEATURES ==========")

    columns = [
        "timestamp",
        "close",
        "structure_event",
        "structure",
        "price_direction",
        "move_3_pct",
        "move_5_pct",
        "pullback_from_high_pct",
        "recovery_from_low_pct",
        "rsi",
        "rsi_direction",
        "rsi_zone",
        "bullish_divergence_candidate",
        "bearish_divergence_candidate",
    ]

    print(
        features[columns]
        .tail(15)
        .to_string(index=False)
    )

    print()
    print("========== RESULT ==========")
    print("PASS")


if __name__ == "__main__":
    main()