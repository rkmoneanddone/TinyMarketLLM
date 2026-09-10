from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tiny_market_llm.technical import TechnicalStateCalculator
from src.tiny_market_llm.validation import SupportResistanceEvidence


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
    print(" TinyMarketLLM - Support/Resistance Evidence Test")
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

    if history.empty:
        raise ValueError(
            "No historical data available."
        )

    print(
        f"[OK] History through: "
        f"{history['timestamp'].iloc[-1]}"
    )

    print()
    print("[2] Calculating technical state...")

    calculator = TechnicalStateCalculator()

    technical = calculator.calculate(
        history
    )

    row = technical.iloc[-1]

    print("[OK] Technical state calculated.")

    print()
    print("[3] Analyzing previous levels...")

    analyzer = SupportResistanceEvidence()

    evidence = analyzer.analyze(
        current_price=float(row["close"]),
        previous_high=(
            None
            if pd.isna(row["previous_high"])
            else float(row["previous_high"])
        ),
        previous_low=(
            None
            if pd.isna(row["previous_low"])
            else float(row["previous_low"])
        ),
    )

    print("[OK] Evidence calculated.")

    print()
    print("========== RESULT ==========")
    print()

    print(
        f"Current Price          : "
        f"{evidence.current_price:.2f}"
    )

    print(
        f"Previous High          : "
        f"{evidence.previous_high}"
    )

    print(
        f"Previous Low           : "
        f"{evidence.previous_low}"
    )

    print(
        f"Distance to High       : "
        f"{evidence.distance_to_high_pct:.2f}%"
        if evidence.distance_to_high_pct is not None
        else "Distance to High       : None"
    )

    print(
        f"Distance to Low        : "
        f"{evidence.distance_to_low_pct:.2f}%"
        if evidence.distance_to_low_pct is not None
        else "Distance to Low        : None"
    )

    print()
    print(
        f"High Status            : "
        f"{evidence.high_status}"
    )

    print(
        f"Low Status             : "
        f"{evidence.low_status}"
    )

    print(
        f"Bounce Possible        : "
        f"{evidence.bounce_possible}"
    )

    print(
        f"Continuation Possible  : "
        f"{evidence.continuation_possible}"
    )

    print()
    print("Remarks:")

    for remark in evidence.remarks:
        print(f" - {remark}")

    print()
    print("PASS")


if __name__ == "__main__":
    main()
