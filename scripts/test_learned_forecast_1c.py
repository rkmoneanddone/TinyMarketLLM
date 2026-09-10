from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from src.tiny_market_llm.learning.learning_engine import (
    LearningEngine,
)

from src.tiny_market_llm.features.feature_engine_v2 import (
    FeatureEngineV2,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OHLC_PATH = (
    PROJECT_ROOT
    / "data"
    / "market"
    / "GRASIM"
    / "1D"
    / "ohlc.parquet"
)

LEARNING_PATH = (
    PROJECT_ROOT
    / "data"
    / "learning"
    / "GRASIM_1D_1C_training_v2.parquet"
)

CUTOFF = pd.Timestamp(
    "2025-02-27 18:30:00+00:00"
)

TARGET_DATE = pd.Timestamp(
    "2025-02-28 18:30:00+00:00"
)


def pct(value: float) -> str:
    return f"{value:+.2f}%"


def main():

    print("=" * 65)
    print(" TinyMarketLLM - 1-Candle Historical Forecast")
    print("=" * 65)
    print()

    # ---------------------------------------------------------
    # 1. Load OHLC
    # ---------------------------------------------------------

    print("[1] Loading OHLC...")

    ohlc = pd.read_parquet(
        OHLC_PATH
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
        f"[OK] Historical candles: {len(ohlc)}"
    )

    # ---------------------------------------------------------
    # 2. Load 1-candle learning data
    # ---------------------------------------------------------

    print()
    print("[2] Loading 1-candle learning data...")

    learning = pd.read_parquet(
        LEARNING_PATH
    )

    learning["decision_timestamp"] = pd.to_datetime(
        learning["decision_timestamp"],
        utc=True,
    )

    print(
        f"[OK] Learning records: {len(learning)}"
    )

    horizons = sorted(
        learning["horizon_candles"]
        .astype(int)
        .unique()
        .tolist()
    )

    print(
        f"[OK] Horizons: {horizons}"
    )

    if horizons != [1]:
        raise RuntimeError(
            f"Expected only horizon 1, got {horizons}"
        )

    # ---------------------------------------------------------
    # 3. Cut training data at 27-Feb
    # ---------------------------------------------------------

    print()
    print("[3] Historical information boundary")

    training = learning[
        learning["decision_timestamp"] < CUTOFF
    ].copy()

    if training.empty:
        raise RuntimeError(
            "No historical training data before cutoff."
        )

    print(
        f"[OK] Training records: {len(training)}"
    )

    print(
        f"[OK] Training through  : "
        f"{training['decision_timestamp'].max()}"
    )

    print(
        f"[OK] Cutoff             : {CUTOFF}"
    )

    # ---------------------------------------------------------
    # 4. Build technical state from data through 27-Feb
    # ---------------------------------------------------------

    print()
    print("[4] Building 27-Feb technical state...")

    historical_ohlc = ohlc[
        ohlc["timestamp"] <= CUTOFF
    ].copy()

    if historical_ohlc.empty:
        raise RuntimeError(
            "No OHLC data available through cutoff."
        )

    feature_engine = FeatureEngineV2()

    features = feature_engine.calculate(
        historical_ohlc
    )

    prediction_rows = features[
        features["timestamp"] == CUTOFF
    ]

    if prediction_rows.empty:
        raise RuntimeError(
            f"Prediction candle not found: {CUTOFF}"
        )

    prediction_row = (
        prediction_rows
        .iloc[-1]
    )

    entry_price = float(
        prediction_row["close"]
    )

    print(
        f"[OK] Prediction candle: "
        f"{prediction_row['timestamp']}"
    )

    print(
        f"[OK] Entry price: "
        f"{entry_price:.2f}"
    )

    # ---------------------------------------------------------
    # 5. Train model using historical 1-candle outcomes
    # ---------------------------------------------------------

    print()
    print("[5] Training 1-candle historical model...")

    engine = LearningEngine()

    model = engine.train(
        training
    )

    print("[OK] Model trained.")

    # ---------------------------------------------------------
    # 6. Prediction
    # ---------------------------------------------------------

    print()
    print("[6] Creating 1-candle forecast...")

    prediction = engine.predict(
        model,
        prediction_row,
        timeframe="1D",
        horizon_candles=1,
    )

    print()
    print("=" * 65)
    print(" MODEL FORECAST FOR 28-FEB-2025")
    print("=" * 65)

    print()
    print(
        f"Timeframe             : 1D"
    )

    print(
        f"Horizon               : 1 candle"
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

    print(
        f"Expected move         : "
        f"{pct(prediction.expected_move_pct)}"
    )

    print(
        f"Favorable move        : "
        f"{pct(prediction.expected_favorable_move_pct)}"
    )

    print(
        f"Adverse move          : "
        f"{pct(prediction.expected_adverse_move_pct)}"
    )

    print(
        f"Expected duration     : "
        f"{prediction.expected_duration}"
    )

    # ---------------------------------------------------------
    # Target range
    # ---------------------------------------------------------

    expected_move = float(
        prediction.expected_move_pct
    )

    expected_favorable = float(
        prediction.expected_favorable_move_pct
    )

    expected_adverse = float(
        prediction.expected_adverse_move_pct
    )

    if prediction.direction == "DOWN":

        target_price = (
            entry_price
            * (
                1
                + expected_move / 100
            )
        )

        expected_high = (
            entry_price
            * (
                1
                + expected_favorable / 100
            )
        )

        expected_low = (
            entry_price
            * (
                1
                - expected_adverse / 100
            )
        )

    elif prediction.direction == "UP":

        target_price = (
            entry_price
            * (
                1
                + expected_move / 100
            )
        )

        expected_high = (
            entry_price
            * (
                1
                + expected_favorable / 100
            )
        )

        expected_low = (
            entry_price
            * (
                1
                - expected_adverse / 100
            )
        )

    else:

        target_price = entry_price

        expected_high = (
            entry_price
            * (
                1
                + expected_favorable / 100
            )
        )

        expected_low = (
            entry_price
            * (
                1
                - expected_adverse / 100
            )
        )

    print()
    print("EXPECTED PRICE RANGE")
    print("-" * 45)

    print(
        f"Entry price           : "
        f"{entry_price:.2f}"
    )

    print(
        f"Expected target       : "
        f"{target_price:.2f}"
    )

    print(
        f"Expected high         : "
        f"{expected_high:.2f}"
    )

    print(
        f"Expected low          : "
        f"{expected_low:.2f}"
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

    # ---------------------------------------------------------
    # 7. Actual 28-Feb
    # ---------------------------------------------------------

    print()
    print("=" * 65)
    print(" ACTUAL 28-FEB-2025")
    print("=" * 65)

    actual_rows = ohlc[
        ohlc["timestamp"] == TARGET_DATE
    ]

    if actual_rows.empty:

        print(
            "[WARNING] 28-Feb candle not available."
        )

        print()
        print(
            "========== RESULT =========="
        )

        print(
            "FORECAST CREATED"
        )

        return

    actual = (
        actual_rows
        .iloc[0]
    )

    actual_open = float(
        actual["open"]
    )

    actual_high = float(
        actual["high"]
    )

    actual_low = float(
        actual["low"]
    )

    actual_close = float(
        actual["close"]
    )

    actual_close_move = (
        (
            actual_close
            - entry_price
        )
        / entry_price
        * 100
    )

    actual_upside = (
        (
            actual_high
            - entry_price
        )
        / entry_price
        * 100
    )

    actual_downside = (
        (
            actual_low
            - entry_price
        )
        / entry_price
        * 100
    )

    print()
    print(
        f"Open                  : "
        f"{actual_open:.2f}"
    )

    print(
        f"High                  : "
        f"{actual_high:.2f}"
    )

    print(
        f"Low                   : "
        f"{actual_low:.2f}"
    )

    print(
        f"Close                 : "
        f"{actual_close:.2f}"
    )

    print()
    print(
        f"Actual close move     : "
        f"{pct(actual_close_move)}"
    )

    print(
        f"Actual max upside     : "
        f"{pct(actual_upside)}"
    )

    print(
        f"Actual max downside   : "
        f"{pct(actual_downside)}"
    )

    # ---------------------------------------------------------
    # 8. Verdict
    # ---------------------------------------------------------

    actual_direction = (
        "UP"
        if actual_close_move > 0.5
        else "DOWN"
        if actual_close_move < -0.5
        else "FLAT"
    )

    direction_correct = (
        prediction.direction
        == actual_direction
    )

    print()
    print("=" * 65)
    print(" VERDICT")
    print("=" * 65)

    print(
        f"Model direction       : "
        f"{prediction.direction}"
    )

    print(
        f"Actual direction      : "
        f"{actual_direction}"
    )

    print(
        f"Direction correct     : "
        f"{direction_correct}"
    )

    print(
        f"Predicted move        : "
        f"{pct(expected_move)}"
    )

    print(
        f"Actual close move     : "
        f"{pct(actual_close_move)}"
    )

    print()
    print("========== RESULT ==========")

    if direction_correct:
        print("DIRECTION PASS")
    else:
        print("DIRECTION FAIL")


if __name__ == "__main__":
    main()