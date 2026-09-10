from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tiny_market_llm.learning.learning_engine import LearningEngine
from src.tiny_market_llm.technical import TechnicalStateCalculator


OHLC_FILE = (
    PROJECT_ROOT
    / "data"
    / "market"
    / "GRASIM"
    / "1D"
    / "ohlc.parquet"
)

LEARNING_FILE = (
    PROJECT_ROOT
    / "data"
    / "learning"
    / "GRASIM_1D_1C_training_v2.parquet"
)

PREDICTION_DATE = "2025-02-27"


def main():

    print("=" * 65)
    print(" TinyMarketLLM - Learned Forecast Test")
    print("=" * 65)
    print()

    print("[1] Loading OHLC...")

    ohlc = pd.read_parquet(OHLC_FILE)

    ohlc["timestamp"] = pd.to_datetime(
        ohlc["timestamp"],
        utc=True,
    )

    ohlc = (
        ohlc
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    cutoff_date = pd.Timestamp(
        PREDICTION_DATE
    ).date()

    history = ohlc[
        ohlc["timestamp"].dt.date <= cutoff_date
    ].copy()

    future = ohlc[
        ohlc["timestamp"].dt.date > cutoff_date
    ].copy()

    print(
        f"[OK] Historical candles: {len(history)}"
    )

    print(
        f"[OK] Hidden candles    : {len(future)}"
    )

    print()
    print("[2] Loading learning data...")

    learning = pd.read_parquet(
        LEARNING_FILE
    )

    print(
        f"[OK] Learning records: {len(learning)}"
    )

    print()
    print("[3] Training historical model...")

    # Only training records before prediction date.
    training = learning[
        pd.to_datetime(
            learning["decision_timestamp"],
            utc=True,
        ).dt.date < cutoff_date
    ].copy()

    print(
        f"[OK] Training records: {len(training)}"
    )

    engine = LearningEngine()

    model = engine.train(
        training
    )

    print("[OK] Model trained.")

    print()
    print("[4] Building 27-Feb technical state...")

    calculator = TechnicalStateCalculator()

    technical = calculator.calculate(
        history
    )

    row = technical.iloc[-1]

    print(
        f"[OK] Prediction candle: "
        f"{row['timestamp']}"
    )

    print(
        f"[OK] Entry price: "
        f"{row['close']:.2f}"
    )

    print()
    print("[5] Creating learned forecast...")

    prediction = engine.predict(
        model,
        row,
        timeframe="1D",
        horizon_candles=1,
    )

    print()
    print("=" * 65)
    print(" MODEL FORECAST FOR 28-FEB-2025")
    print("=" * 65)

    print()
    print(
        f"Timeframe             : "
        f"{prediction.timeframe}"
    )

    print(
        f"Horizon               : "
        f"{prediction.horizon_candles} candle"
    )

    print(
        f"Direction             : "
        f"{prediction.direction}"
    )

    print(
        f"Confidence            : "
        f"{prediction.confidence:.4f}"
    )

    print(
        f"Probability           : "
        f"{prediction.probability}"
    )

    print()
    print("MOVE")
    print("-" * 45)



    print()
    print("TARGET")
    print("-" * 45)

    print(
        f"Target price          : "
        f"{prediction.target_price:.2f}"
        if prediction.target_price is not None
        else "Target price          : None"
    )

    print(
        f"Target low            : "
        f"{prediction.target_low:.2f}"
        if prediction.target_low is not None
        else "Target low            : None"
    )

    print(
        f"Target high           : "
        f"{prediction.target_high:.2f}"
        if prediction.target_high is not None
        else "Target high           : None"
    )

    print()
    print("PROBABILITIES")
    print("-" * 45)

    print(
        f"Sustain probability   : "
        f"{prediction.sustain_probability:.4f}"
    )

    print(
        f"Target probability    : "
        f"{prediction.target_probability:.4f}"
    )

    print()
    print("REMARKS")

    if prediction.remarks:
        for remark in prediction.remarks:
            print(f" - {remark}")

    print()
    print("========== ACTUAL 28-FEB ==========")

    if future.empty:
        print("No future candle available.")
        return

    actual = future.iloc[0]

    print(
        f"Open                  : "
        f"{actual['open']:.2f}"
    )

    print(
        f"High                  : "
        f"{actual['high']:.2f}"
    )

    print(
        f"Low                   : "
        f"{actual['low']:.2f}"
    )

    print(
        f"Close                 : "
        f"{actual['close']:.2f}"
    )

    entry = float(row["close"])

    actual_close_move = (
        (float(actual["close"]) - entry)
        / entry
    ) * 100

    actual_high_move = (
        (float(actual["high"]) - entry)
        / entry
    ) * 100

    actual_low_move = (
        (float(actual["low"]) - entry)
        / entry
    ) * 100

    print()
    print(
        f"Actual close move     : "
        f"{actual_close_move:.2f}%"
    )

    print(
        f"Actual max upside     : "
        f"{actual_high_move:.2f}%"
    )

    print(
        f"Actual max downside   : "
        f"{actual_low_move:.2f}%"
    )

    print()
    print("PASS")


if __name__ == "__main__":
    main()