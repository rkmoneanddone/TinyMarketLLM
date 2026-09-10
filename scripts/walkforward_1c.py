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

START_DATE = pd.Timestamp(
    "2024-12-17",
    tz="UTC",
)

END_DATE = pd.Timestamp(
    "2025-02-27 18:30:00",
    tz="UTC",
)


def actual_direction(move: float) -> str:
    if move > 0.5:
        return "UP"
    if move < -0.5:
        return "DOWN"
    return "FLAT"


def main():

    print("=" * 70)
    print(" TinyMarketLLM - Walk-Forward 1-Candle Evaluation")
    print("=" * 70)
    print()

    # ---------------------------------------------------------
    # 1. Load OHLC
    # ---------------------------------------------------------

    print("[1] Loading OHLC...")

    ohlc = pd.read_parquet(
        OHLC_FILE
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
        f"[OK] Candles: {len(ohlc)}"
    )

    # ---------------------------------------------------------
    # 2. Load 1-candle learning data
    # ---------------------------------------------------------

    print()
    print("[2] Loading 1-candle learning data...")

    learning = pd.read_parquet(
        LEARNING_FILE
    )

    learning["decision_timestamp"] = pd.to_datetime(
        learning["decision_timestamp"],
        utc=True,
    )

    learning = learning[
        learning["horizon_candles"] == 1
    ].copy()

    print(
        f"[OK] Learning records: {len(learning)}"
    )

    # ---------------------------------------------------------
    # 3. Calculate technical states
    # ---------------------------------------------------------

    print()
    print("[3] Calculating technical states...")

    calculator = TechnicalStateCalculator()

    technical = calculator.calculate(
        ohlc
    )

    technical["timestamp"] = pd.to_datetime(
        technical["timestamp"],
        utc=True,
    )

    print(
        f"[OK] Technical rows: {len(technical)}"
    )

    # ---------------------------------------------------------
    # 4. Select evaluation dates
    # ---------------------------------------------------------

    evaluation_rows = technical[
        (technical["timestamp"] >= START_DATE)
        & (technical["timestamp"] <= END_DATE)
    ].copy()

    print()
    print("[4] Evaluation window")

    print(
        f"[OK] From: {evaluation_rows['timestamp'].min()}"
    )

    print(
        f"[OK] To  : {evaluation_rows['timestamp'].max()}"
    )

    # ---------------------------------------------------------
    # 5. Walk forward
    # ---------------------------------------------------------

    print()
    print("[5] Running walk-forward predictions...")
    print()

    predictions = []

    total = len(evaluation_rows)

    for count, (_, row) in enumerate(
        evaluation_rows.iterrows(),
        start=1,
    ):

        cutoff = row["timestamp"]

        # Training must be strictly BEFORE
        # the prediction candle.
        training = learning[
            learning["decision_timestamp"] < cutoff
        ].copy()

        if len(training) < 100:
            continue

        engine = LearningEngine()

        try:
            model = engine.train(
                training
            )

            prediction = engine.predict(
                model,
                row,
                timeframe="1D",
                horizon_candles=1,
            )

        except Exception as exc:

            print(
                f"[SKIP] {cutoff.date()} "
                f"{type(exc).__name__}: {exc}"
            )

            continue

        # Find the actual NEXT candle.
        future = technical[
            technical["timestamp"] > cutoff
        ]

        if future.empty:
            continue

        actual = future.iloc[0]

        entry = float(
            row["close"]
        )

        close = float(
            actual["close"]
        )

        high = float(
            actual["high"]
        )

        low = float(
            actual["low"]
        )

        move = (
            (close - entry)
            / entry
            * 100
        )

        upside = (
            (high - entry)
            / entry
            * 100
        )

        downside = (
            (low - entry)
            / entry
            * 100
        )

        actual_dir = actual_direction(
            move
        )

        predicted_dir = prediction.direction

        predictions.append(
            {
                "timestamp": cutoff,
                "entry": entry,

                "prediction": predicted_dir,
                "confidence": float(
                    prediction.confidence
                ),

                "up_probability": float(
                    prediction.probability.get(
                        "UP",
                        0.0,
                    )
                ),

                "down_probability": float(
                    prediction.probability.get(
                        "DOWN",
                        0.0,
                    )
                ),

                "flat_probability": float(
                    prediction.probability.get(
                        "FLAT",
                        0.0,
                    )
                ),

                "expected_move": float(
                    prediction.expected_move_pct
                ),

                "favorable": float(
                    prediction.expected_favorable_move_pct
                ),

                "adverse": float(
                    prediction.expected_adverse_move_pct
                ),

                "target_price": (
                    float(prediction.target_price)
                    if prediction.target_price is not None
                    else None
                ),

                "actual": actual_dir,
                "actual_move": move,
                "actual_upside": upside,
                "actual_downside": downside,

                "correct": (
                    predicted_dir
                    == actual_dir
                ),

                # State evidence
                "structure": row.get(
                    "structure"
                ),

                "price_direction": row.get(
                    "price_direction"
                ),

                "rsi": row.get(
                    "rsi"
                ),

                "rsi_direction": row.get(
                    "rsi_direction"
                ),

                "rsi_zone": row.get(
                    "rsi_zone"
                ),

                "bullish_divergence": row.get(
                    "bullish_divergence_candidate"
                ),

                "bearish_divergence": row.get(
                    "bearish_divergence_candidate"
                ),
            }
        )

        if count % 25 == 0:

            print(
                f"[{count:4d}/{total}] "
                f"{cutoff.date()} "
                f"{predicted_dir:5s} -> "
                f"{actual_dir:5s} "
                f"{'PASS' if predicted_dir == actual_dir else 'FAIL'}"
            )

    results = pd.DataFrame(
        predictions
    )

    if results.empty:
        raise RuntimeError(
            "No walk-forward predictions generated."
        )

    # ---------------------------------------------------------
    # 6. Overall statistics
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print(" WALK-FORWARD RESULTS")
    print("=" * 70)

    total_predictions = len(
        results
    )

    correct = int(
        results["correct"].sum()
    )

    accuracy = (
        correct
        / total_predictions
        * 100
    )

    print()
    print(
        f"Predictions           : "
        f"{total_predictions}"
    )

    print(
        f"Correct directions    : "
        f"{correct}"
    )

    print(
        f"Direction accuracy    : "
        f"{accuracy:.2f}%"
    )

    print()
    print("PREDICTED DIRECTIONS")
    print("-" * 45)

    print(
        results[
            "prediction"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("ACTUAL DIRECTIONS")
    print("-" * 45)

    print(
        results[
            "actual"
        ]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # Per-direction accuracy
    # ---------------------------------------------------------

    print()
    print("DIRECTION ACCURACY")
    print("-" * 45)

    for direction in (
        "UP",
        "DOWN",
        "FLAT",
    ):

        subset = results[
            results["prediction"]
            == direction
        ]

        if subset.empty:

            print(
                f"{direction:5s}: no predictions"
            )

            continue

        acc = (
            subset["correct"].mean()
            * 100
        )

        print(
            f"{direction:5s}: "
            f"{acc:.2f}% "
            f"({len(subset)} predictions)"
        )

    # ---------------------------------------------------------
    # Move accuracy
    # ---------------------------------------------------------

    results["move_error"] = (
        results["expected_move"]
        - results["actual_move"]
    ).abs()

    print()
    print("MOVE STATISTICS")
    print("-" * 45)

    print(
        f"Average predicted move : "
        f"{results['expected_move'].mean():+.3f}%"
    )

    print(
        f"Average actual move    : "
        f"{results['actual_move'].mean():+.3f}%"
    )

    print(
        f"Average move error     : "
        f"{results['move_error'].mean():.3f}%"
    )

    # ---------------------------------------------------------
    # Confidence buckets
    # ---------------------------------------------------------

    print()
    print("CONFIDENCE ANALYSIS")
    print("-" * 45)

    for threshold in (
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ):

        subset = results[
            results["confidence"]
            >= threshold
        ]

        if subset.empty:

            print(
                f">={threshold:.0%}: no samples"
            )

            continue

        acc = (
            subset["correct"].mean()
            * 100
        )

        print(
            f">={threshold:.0%}: "
            f"{acc:.2f}% accuracy "
            f"({len(subset)} samples)"
        )

    # ---------------------------------------------------------
    # RSI analysis
    # ---------------------------------------------------------

    print()
    print("RSI STATE ANALYSIS")
    print("-" * 45)

    for state in (
        "OVERSOLD",
        "WEAK",
        "NEUTRAL",
        "STRONG",
        "OVERBOUGHT",
    ):

        subset = results[
            results["rsi_zone"]
            == state
        ]

        if subset.empty:
            continue

        acc = (
            subset["correct"].mean()
            * 100
        )

        print(
            f"{state:12s}: "
            f"{acc:.2f}% "
            f"({len(subset)})"
        )

    # ---------------------------------------------------------
    # RSI direction
    # ---------------------------------------------------------

    print()
    print("RSI DIRECTION")
    print("-" * 45)

    for direction in (
        "RISING",
        "FALLING",
        "STABLE",
    ):

        subset = results[
            results["rsi_direction"]
            == direction
        ]

        if subset.empty:
            continue

        acc = (
            subset["correct"].mean()
            * 100
        )

        print(
            f"{direction:8s}: "
            f"{acc:.2f}% "
            f"({len(subset)})"
        )

    # ---------------------------------------------------------
    # Important reversal combinations
    # ---------------------------------------------------------

    print()
    print("REVERSAL / CONTINUATION EVIDENCE")
    print("-" * 45)

    price_down_rsi_up = results[
        (results["price_direction"] == "DOWN")
        & (results["rsi_direction"] == "RISING")
    ]

    price_up_rsi_down = results[
        (results["price_direction"] == "UP")
        & (results["rsi_direction"] == "FALLING")
    ]

    if not price_down_rsi_up.empty:

        print(
            "Price DOWN + RSI RISING : "
            f"{price_down_rsi_up['correct'].mean() * 100:.2f}% "
            f"({len(price_down_rsi_up)})"
        )

    if not price_up_rsi_down.empty:

        print(
            "Price UP + RSI FALLING  : "
            f"{price_up_rsi_down['correct'].mean() * 100:.2f}% "
            f"({len(price_up_rsi_down)})"
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    output = (
        PROJECT_ROOT
        / "data"
        / "learning"
        / "GRASIM_1D_1C_walkforward.parquet"
    )

    results.to_parquet(
        output,
        index=False,
    )

    print()
    print(
        f"[OK] Detailed results saved:"
    )

    print(
        output
    )

    print()
    print("========== RESULT ==========")
    print("WALK-FORWARD PASS")


if __name__ == "__main__":
    main()