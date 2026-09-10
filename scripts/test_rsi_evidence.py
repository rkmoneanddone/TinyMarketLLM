from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tiny_market_llm.technical import TechnicalStateCalculator
from src.tiny_market_llm.validation import RSIBehaviorAnalyzer


DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "market"
    / "GRASIM"
    / "1D"
    / "ohlc.parquet"
)

PREDICTION_DATE = "2025-02-27"


def main():

    print("=" * 60)
    print(" TinyMarketLLM - RSI Evidence Test")
    print("=" * 60)
    print()

    print("[1] Loading historical OHLC...")

    df = pd.read_parquet(DATA_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    prediction_date = pd.Timestamp(
        PREDICTION_DATE
    ).date()

    history = df[
        df["timestamp"].dt.date <= prediction_date
    ].copy()

    print(
        f"[OK] History through: "
        f"{history['timestamp'].iloc[-1]}"
    )

    print()
    print("[2] Calculating RSI...")

    calculator = TechnicalStateCalculator()

    technical = calculator.calculate(
        history
    )

    print("[OK] RSI calculated.")

    print()
    print("[3] Analyzing RSI behavior...")

    analyzer = RSIBehaviorAnalyzer()

    evidence = analyzer.analyze(
        technical
    )

    print("[OK] RSI evidence calculated.")

    print()
    print("========== RESULT ==========")
    print()

    print(
        f"RSI                         : "
        f"{evidence.rsi}"
    )

    print(
        f"RSI Direction               : "
        f"{evidence.rsi_direction}"
    )

    print(
        f"RSI Zone                    : "
        f"{evidence.rsi_zone}"
    )

    print(
        f"Bullish Divergence          : "
        f"{evidence.bullish_divergence_candidate}"
    )

    print(
        f"Bearish Divergence          : "
        f"{evidence.bearish_divergence_candidate}"
    )

    print(
        f"Bullish Double Bottom      : "
        f"{evidence.bullish_double_bottom_candidate}"
    )

    print()
    print("Remarks:")

    for remark in evidence.remarks:
        print(f" - {remark}")

    print()
    print("PASS")


if __name__ == "__main__":
    main()
