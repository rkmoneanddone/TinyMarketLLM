from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tiny_market_llm.technical import (
    TechnicalStateCalculator,
)


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
    print(" TinyMarketLLM - Historical Technical State")
    print("=" * 60)
    print()

    print("[1] Loading full Dhan OHLC history...")

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

    print(f"[OK] Records: {len(df)}")

    prediction_date = pd.Timestamp(
        PREDICTION_DATE
    ).date()

    cutoff_rows = df[
        df["timestamp"].dt.date <= prediction_date
    ].copy()

    future_rows = df[
        df["timestamp"].dt.date > prediction_date
    ].copy()

    if cutoff_rows.empty:
        raise ValueError(
            f"No historical data through {PREDICTION_DATE}"
        )

    if future_rows.empty:
        raise ValueError(
            "No future candles available."
        )

    print()
    print("[2] Historical information boundary")

    print(
        f"[OK] Data available through : "
        f"{cutoff_rows['timestamp'].iloc[-1]}"
    )

    print(
        f"[OK] First hidden candle     : "
        f"{future_rows['timestamp'].iloc[0]}"
    )

    # --------------------------------------------------
    # IMPORTANT:
    # Technical indicators are calculated ONLY from
    # information available through 27-Feb.
    #
    # Therefore 28-Feb cannot influence EMA, RSI,
    # structure, previous levels, etc.
    # --------------------------------------------------

    print()
    print("[3] Calculating technical state...")

    calculator = TechnicalStateCalculator()

    technical = calculator.calculate(
        cutoff_rows
    )

    row = technical.iloc[-1]

    print("[OK] Technical state calculated.")

    print()
    print("=" * 60)
    print(" 27-FEB-2025 TECHNICAL STATE")
    print("=" * 60)

    print()
    print(f"Timestamp          : {row['timestamp']}")
    print(f"Price              : {row['close']:.2f}")

    print()
    print("EMA")
    print("-" * 40)

    print(
        f"21 EMA             : "
        f"{row['ema_21']:.2f}"
    )

    print(
        f"50 EMA             : "
        f"{row['ema_50']:.2f}"
    )

    if pd.isna(row["ema_200"]):
        print("200 EMA            : UNAVAILABLE")
    else:
        print(
            f"200 EMA            : "
            f"{row['ema_200']:.2f}"
        )

    print(
        f"EMA Alignment      : "
        f"{bool(row['ema_alignment'])}"
    )

    print(
        f"Alignment Type     : "
        f"{row['ema_alignment_type']}"
    )

    print()
    print("RSI")
    print("-" * 40)

    print(
        f"RSI                : "
        f"{row['rsi']:.2f}"
    )

    print(
        f"RSI Direction      : "
        f"{row['rsi_direction']}"
    )

    print(
        f"RSI Zone           : "
        f"{row['rsi_zone']}"
    )

    print()
    print("STRUCTURE")
    print("-" * 40)

    print(
        f"Structure           : "
        f"{row['structure']}"
    )

    print(
        f"Price Direction    : "
        f"{row['price_direction']}"
    )

    print()
    print("PREVIOUS LEVELS")
    print("-" * 40)

    print(
        f"Previous High      : "
        f"{row['previous_high']}"
    )

    print(
        f"Previous Low       : "
        f"{row['previous_low']}"
    )

    print()
    print("MARKET LOCATION")
    print("----------------------------------------")

    print(
        f"Distance to Previous High : "
        f"{row['distance_to_previous_high_pct']:.2f}%"
        if pd.notna(row["distance_to_previous_high_pct"])
        else "Distance to Previous High : None"
    )

    print(
        f"Distance to Previous Low  : "
        f"{row['distance_to_previous_low_pct']:.2f}%"
        if pd.notna(row["distance_to_previous_low_pct"])
        else "Distance to Previous Low  : None"
    )

    print(
        f"Above Previous High       : "
        f"{row['above_previous_high']}"
    )

    print(
        f"Below Previous Low        : "
        f"{row['below_previous_low']}"
    )

    print(
        f"Previous High Broken      : "
        f"{row['previous_high_broken']}"
    )

    print(
        f"Previous Low Broken       : "
        f"{row['previous_low_broken']}"
    )

    print(
        f"Near Previous High        : "
        f"{row['near_previous_high']}"
    )

    print(
        f"Near Previous Low         : "
        f"{row['near_previous_low']}"
    )

    print()
    print("SUPPORT / RESISTANCE")
    print("----------------------------------------")

    print(
        f"Support Status            : "
        f"{row['support_status']}"
    )

    print(
        f"Resistance Status         : "
        f"{row['resistance_status']}"
    )

    print()
    print("BOUNCE / CONTINUATION")
    print("----------------------------------------")

    print(
        f"Bounce From Low           : "
        f"{row['bounce_from_low_pct']:.2f}%"
    )

    print(
        f"Rejection From High       : "
        f"{row['rejection_from_high_pct']:.2f}%"
    )

    print(
        f"Bounce Signal             : "
        f"{row['bounce_signal']}"
    )

    print(
        f"Continuation Signal       : "
        f"{row['continuation_signal']}"
    )

    print()
    print("========== HIDDEN FUTURE ==========")

    future = future_rows.iloc[0]

    print(
        f"First hidden candle : "
        f"{future['timestamp']}"
    )

    print()
    print("The prediction state above was calculated")
    print("without using this future candle.")

    print()
    print("========== RESULT ==========")
    print()
    print("PASS")


if __name__ == "__main__":
    main()
