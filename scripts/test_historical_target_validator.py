from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

import pandas as pd

from src.tiny_market_llm.validation.historical_target_validator import (
    HistoricalTargetValidator,
)


def main():

    symbol = "GRASIM"
    timeframe = "1D"

    ohlc_path = (
        PROJECT_ROOT
        / "data"
        / "market"
        / symbol
        / timeframe
        / "ohlc.parquet"
    )

    structure_path = (
        PROJECT_ROOT
        / "data"
        / "learning"
        / "GRASIM_1D_training_v2.parquet"
    )

    print("=" * 50)
    print(" TinyMarketLLM - Historical Target Validation")
    print("=" * 50)
    print()

    print("[1] Loading Dhan OHLC...")
    ohlc = pd.read_parquet(
        ohlc_path
    )

    print(
        f"[OK] OHLC records: {len(ohlc)}"
    )

    print()
    print("[2] Loading V2 structure data...")

    features = pd.read_parquet(
        structure_path
    )

    required = [
    "decision_timestamp",
    "structure",
    ]

    missing = [
        column
        for column in required
        if column not in features.columns
    ]

    if missing:
        raise ValueError(
            f"Missing structure columns: {missing}"
        )

    features = features.rename(
        columns={
            "decision_timestamp": "timestamp",
        }
    )

    print(
        f"[OK] Structure records: {len(features)}"
    )

    print()
    print("[3] Running historical validation...")

    validator = HistoricalTargetValidator(
        lookahead=10,
        pivot_bars=2,
    )

    report = validator.evaluate(
        ohlc=ohlc,
        structures=features[
            [
                "timestamp",
                "structure",
            ]
        ],
    )

    print("[OK] Validation complete.")

    print()
    print("========== RESULT ==========")
    print()

    print(
        "Symbol                  :",
        symbol,
    )

    print(
        "Timeframe               :",
        timeframe,
    )

    print(
        "OHLC candles            :",
        report["candles"],
    )

    print(
        "Qualified samples       :",
        report["samples"],
    )

    print(
        "Bullish samples         :",
        report["bullish_samples"],
    )

    print(
        "Bearish samples         :",
        report["bearish_samples"],
    )

    print()
    print("---- Bullish ----")

    print(
        "Previous High hit rate :",
        f"{report['bullish_previous_high_hit_rate']:.2f}%",
    )

    print(
        "Fib 1.618 hit rate     :",
        f"{report['bullish_fib_1618_hit_rate']:.2f}%",
    )

    print()
    print("---- Bearish ----")

    print(
        "Previous Low hit rate  :",
        f"{report['bearish_previous_low_hit_rate']:.2f}%",
    )

    print(
        "Fib 1.618 hit rate     :",
        f"{report['bearish_fib_1618_hit_rate']:.2f}%",
    )

    print()
    print("---- Combined ----")

    print(
        "Previous level hit rate:",
        f"{report['previous_level_hit_rate']:.2f}%",
    )

    print(
        "Fib 1.618 hit rate      :",
        f"{report['fib_1618_hit_rate']:.2f}%",
    )

    print()
    print("PASS")


if __name__ == "__main__":
    main()
