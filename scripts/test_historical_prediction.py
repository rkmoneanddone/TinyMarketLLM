from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from src.tiny_market_llm.learning.learning_engine import (
    LearningEngine,
)


SYMBOL = "GRASIM"
TIMEFRAME = "1D"

FROM_DATE = "2024-12-17"
TO_DATE = "2025-03-03"

LOOKAHEAD = 10


def pct(a, b):
    if b == 0:
        return 0.0

    return (
        (a - b)
        / b
        * 100.0
    )


def first_direction(
    future,
    entry_price,
):
    """
    Determine which direction happens first.

    We use the candle HIGH/LOW rather than close.
    """

    for _, row in future.iterrows():

        high_move = pct(
            row["high"],
            entry_price,
        )

        low_move = pct(
            row["low"],
            entry_price,
        )

        if high_move > 0 and low_move < 0:

            if abs(high_move) > abs(low_move):
                return "UP"

            if abs(low_move) > abs(high_move):
                return "DOWN"

            return "SAME"

        if high_move > 0:
            return "UP"

        if low_move < 0:
            return "DOWN"

    return "NONE"


def main():

    print("=" * 65)
    print(" TinyMarketLLM - Historical Model Prediction")
    print("=" * 65)

    market_path = (
        PROJECT_ROOT
        / "data"
        / "market"
        / SYMBOL
        / TIMEFRAME
        / "ohlc.parquet"
    )

    learning_path = (
        PROJECT_ROOT
        / "data"
        / "learning"
        / f"{SYMBOL}_1D_training_v2.parquet"
    )

    # --------------------------------------------------
    # 1. Load historical OHLC
    # --------------------------------------------------

    print()
    print("[1] Loading historical OHLC...")

    ohlc = pd.read_parquet(
        market_path
    )

    ohlc["timestamp"] = pd.to_datetime(
        ohlc["timestamp"],
        utc=True,
    )

    ohlc = (
        ohlc
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"[OK] OHLC candles: {len(ohlc)}"
    )

    # --------------------------------------------------
    # 2. Load V2 learning dataset
    # --------------------------------------------------

    print()
    print("[2] Loading V2 learning data...")

    learning = pd.read_parquet(
        learning_path
    )

    if "decision_timestamp" in learning.columns:

        learning["timestamp"] = pd.to_datetime(
            learning["decision_timestamp"],
            utc=True,
        )

    else:

        learning["timestamp"] = pd.to_datetime(
            learning["timestamp"],
            utc=True,
        )

    learning = (
        learning
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"[OK] Learning records: {len(learning)}"
    )

    # --------------------------------------------------
    # 3. Find cutoff candle
    # --------------------------------------------------

    cutoff = pd.Timestamp(
    CUTOFF_DATE,
    tz="UTC",
    )

# --------------------------------------------------
# Exact prediction candle.
# Do NOT silently use an earlier candle.
# --------------------------------------------------

    prediction_rows = ohlc[
        ohlc["timestamp"] == cutoff
    ].copy()

    if prediction_rows.empty:
        available = ohlc[
            ohlc["timestamp"] <= cutoff
        ]["timestamp"]

        raise ValueError(
            f"Exact prediction candle not found: {cutoff}. "
            f"Latest available candle at/before cutoff: "
            f"{available.max() if not available.empty else None}"
        )

    history = ohlc[
        ohlc["timestamp"] <= cutoff
    ].copy()

    future = ohlc[
        ohlc["timestamp"] > cutoff
    ].head(LOOKAHEAD).copy()

    if future.empty:
        raise ValueError(
            "No future candles found after prediction candle."
        )

    entry = prediction_rows.iloc[-1]

    entry_price = float(
        entry["close"]
    )

    future = ohlc[
        ohlc["timestamp"] > cutoff
    ].head(LOOKAHEAD).copy()

    if history.empty:
        raise ValueError(
            "No historical candles found before cutoff."
        )

    if future.empty:
        raise ValueError(
            "No future candles found after cutoff."
        )

    entry = history.iloc[-1]

    entry_price = float(
        entry["close"]
    )

    print()
    print("[3] Prediction point")
    print(
        "Cutoff candle:",
        entry["timestamp"],
    )

    print(
        "Entry price:",
        entry_price,
    )

    print(
        "Future candles:",
        len(future),
    )

    # --------------------------------------------------
    # 4. IMPORTANT:
    #    Train ONLY using information available
    #    before the prediction date.
    # --------------------------------------------------

    train_data = learning[
        learning["timestamp"] < cutoff
    ].copy()

    if train_data.empty:
        raise ValueError(
            "No training data before cutoff."
        )

    print()
    print("[4] Training model using historical data only...")

    print(
        "Training records:",
        len(train_data),
    )

    engine = LearningEngine(
        train_ratio=0.70,
        validation_ratio=0.15,
        minimum_samples=20,
        promotion_threshold=0.80,
    )

    model = engine.train(
        train_data
    )

    print(
        "[OK] Model trained."
    )

    # --------------------------------------------------
    # 5. Get model input at cutoff
    # --------------------------------------------------

    print()
    print("[5] Creating prediction...")

    candidates = learning[
        learning["timestamp"] == cutoff
    ]

    if candidates.empty:

        # Sometimes the learning dataset timestamp
        # does not exactly match the OHLC cutoff.
        candidates = learning[
            learning["timestamp"] <= cutoff
        ]

    if candidates.empty:
        raise ValueError(
            "No learning record available at cutoff."
        )

    row = candidates.iloc[-1]

    prediction = engine.predict(
        model,
        row,
    )

    print()
    print("========== MODEL PREDICTION ==========")
    print()

    print(
        "Direction :",
        prediction.direction,
    )

    print(
        "Confidence:",
        prediction.confidence,
    )

    print(
        "Probability:",
        prediction.probability,
    )

    # --------------------------------------------------
    # 6. Now reveal the future
    # --------------------------------------------------

    print()
    print("========== ACTUAL NEXT 10 CANDLES ==========")

    first = first_direction(
        future,
        entry_price,
    )

    highest = float(
        future["high"].max()
    )

    lowest = float(
        future["low"].min()
    )

    final_close = float(
        future.iloc[-1]["close"]
    )

    max_upside = pct(
        highest,
        entry_price,
    )

    max_downside = pct(
        lowest,
        entry_price,
    )

    final_move = pct(
        final_close,
        entry_price,
    )

    print()
    print(
        "First direction :",
        first,
    )

    print(
        "Highest price   :",
        highest,
    )

    print(
        "Max upside      :",
        f"{max_upside:.2f}%",
    )

    print(
        "Lowest price    :",
        lowest,
    )

    print(
        "Max downside    :",
        f"{max_downside:.2f}%",
    )

    print(
        "Final close     :",
        final_close,
    )

    print(
        "Final move      :",
        f"{final_move:.2f}%",
    )

    # --------------------------------------------------
    # 7. Candle-by-candle result
    # --------------------------------------------------

    print()
    print("CANDLE-BY-CANDLE")
    print("-" * 65)

    for i, (_, candle) in enumerate(
        future.iterrows(),
        start=1,
    ):

        high_move = pct(
            float(candle["high"]),
            entry_price,
        )

        low_move = pct(
            float(candle["low"]),
            entry_price,
        )

        close_move = pct(
            float(candle["close"]),
            entry_price,
        )

        print(
            f"{i:2} | "
            f"{candle['timestamp'].date()} | "
            f"High {high_move:+.2f}% | "
            f"Low {low_move:+.2f}% | "
            f"Close {close_move:+.2f}%"
        )

    # --------------------------------------------------
    # 8. Compare prediction vs reality
    # --------------------------------------------------

    predicted = prediction.direction

    correct = (
        predicted == first
    )

    print()
    print("========== VERDICT ==========")

    print(
        "Model prediction:",
        predicted,
    )

    print(
        "Actual first move:",
        first,
    )

    print(
        "Direction correct:",
        correct,
    )

    print()
    print("PASS")


if __name__ == "__main__":
    main()
