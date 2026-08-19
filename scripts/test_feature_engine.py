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


def main():
    print("=" * 40)
    print(" TinyMarketLLM - Feature Engine Test")
    print("=" * 40)
    print()

    repository = MarketRepository()

    print("[TEST] Loading GRASIM 1D...")
    data = repository.load(
        "GRASIM",
        "1D",
    )

    print(
        f"[OK] Input candles: {len(data)}"
    )

    print()
    print("[TEST] Calculating features...")

    engine = FeatureEngine(
        rsi_period=14,
        ma_period=20,
    )

    features = engine.calculate(data)

    print(
        f"[OK] Output candles: {len(features)}"
    )

    required = [
        "structure_event",
        "structure",
        "price_direction",
        "rsi",
        "rsi_direction",
        "rsi_overbought",
        "rsi_oversold",
        "ma",
        "ma_direction",
        "price_vs_ma",
    ]

    print()
    print("[TEST] Feature columns...")

    missing = [
        column
        for column in required
        if column not in features.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing feature columns: {missing}"
        )

    print("[OK] All feature columns present.")

    print()
    print("[TEST] Structure events...")

    structure_counts = (
        features["structure_event"]
        .value_counts(dropna=True)
    )

    print(
        structure_counts.to_string()
    )

    print()
    print("[TEST] RSI...")

    valid_rsi = features["rsi"].dropna()

    if valid_rsi.empty:
        raise RuntimeError(
            "No RSI values were calculated."
        )

    if not (
        (valid_rsi >= 0)
        & (valid_rsi <= 100)
    ).all():
        raise RuntimeError(
            "RSI contains values outside 0-100."
        )

    print(
        f"[OK] RSI range: "
        f"{valid_rsi.min():.2f} -> "
        f"{valid_rsi.max():.2f}"
    )

    print()
    print("[TEST] Moving average...")

    valid_ma = features["ma"].dropna()

    if valid_ma.empty:
        raise RuntimeError(
            "No moving-average values were calculated."
        )

    print(
        f"[OK] MA range: "
        f"{valid_ma.min():.2f} -> "
        f"{valid_ma.max():.2f}"
    )

    print()
    print("========== LAST 15 FEATURE ROWS ==========")

    columns = [
        "timestamp",
        "close",
        "structure_event",
        "structure",
        "price_direction",
        "rsi",
        "rsi_direction",
        "ma",
        "ma_direction",
        "price_vs_ma",
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