from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tiny_market_llm.data.market_repository import MarketRepository
from src.tiny_market_llm.features.feature_engine_v2 import FeatureEngineV2
from src.tiny_market_llm.learning.learning_engine import LearningEngine


START_DATE = pd.Timestamp("2024-12-17 18:30:00+00:00")
END_DATE = pd.Timestamp("2025-02-27 18:30:00+00:00")


def direction_from_move(move):
    if move > 0.5:
        return "UP"

    if move < -0.5:
        return "DOWN"

    return "FLAT"


def print_trade_quality(name, data):

    print()
    print(name)
    print("-" * 45)

    for direction in ["UP", "DOWN", "FLAT"]:

        subset = data[
            data["prediction"] == direction
        ]

        if subset.empty:
            print(f"{direction:<6}: no samples")
            continue

        avg_move = subset["actual_move"].mean()
        median_move = subset["actual_move"].median()

        if direction == "UP":

            win_rate = (
                subset["actual_move"] > 0.5
            ).mean() * 100

            large_move = (
                subset["actual_move"] >= 1.0
            ).mean() * 100

        elif direction == "DOWN":

            win_rate = (
                subset["actual_move"] < -0.5
            ).mean() * 100

            large_move = (
                subset["actual_move"] <= -1.0
            ).mean() * 100

        else:

            win_rate = (
                subset["actual_move"].abs() <= 0.5
            ).mean() * 100

            large_move = (
                subset["actual_move"].abs() >= 1.0
            ).mean() * 100

        print(
            f"{direction:<6}: "
            f"N={len(subset):2d} "
            f"Avg={avg_move:+.3f}% "
            f"Median={median_move:+.3f}% "
            f"Win={win_rate:6.2f}% "
            f"|1%|={large_move:6.2f}%"
        )


def print_confusion_matrix(name, data):

    print()
    print(name)
    print("-" * 45)

    matrix = pd.crosstab(
        data["prediction"],
        data["actual"],
        rownames=["Predicted"],
        colnames=["Actual"],
        dropna=False,
    )

    for direction in ["UP", "DOWN", "FLAT"]:

        if direction not in matrix.index:
            matrix.loc[direction] = 0

        for actual in ["UP", "DOWN", "FLAT"]:

            if actual not in matrix.columns:
                matrix[actual] = 0

    matrix = matrix[
        ["UP", "DOWN", "FLAT"]
    ].sort_index()

    print(matrix.to_string())


def print_directional_edge(name, data):

    print()
    print(name)
    print("-" * 45)

    for direction in ["UP", "DOWN"]:

        subset = data[
            data["prediction"] == direction
        ]

        if subset.empty:
            print(f"{direction:<6}: no samples")
            continue

        if direction == "UP":

            favorable = (
                subset["actual_move"] > 0
            ).mean() * 100

            strong = (
                subset["actual_move"] >= 1.0
            ).mean() * 100

        else:

            favorable = (
                subset["actual_move"] < 0
            ).mean() * 100

            strong = (
                subset["actual_move"] <= -1.0
            ).mean() * 100

        print(
            f"{direction:<6}: "
            f"favorable={favorable:6.2f}% "
            f"strong(|move|>=1%)={strong:6.2f}%"
        )


def main():

    print("=" * 70)
    print(" TinyMarketLLM - Core vs Evidence Walk-Forward 1-Candle")
    print("=" * 70)
    print()

    repository = MarketRepository()

    print("[1] Loading GRASIM 1D...")

    data = repository.load(
        "GRASIM",
        "1D",
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
    )

    data = (
        data
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"[OK] Candles: {len(data)}"
    )

    print()
    print("[2] Calculating V2 features...")

    features = FeatureEngineV2().calculate(
        data
    )

    print(
        f"[OK] Feature rows: {len(features)}"
    )

    print()
    print("[3] Loading learning dataset...")

    learning_path = (
        PROJECT_ROOT
        / "data"
        / "learning"
        / "GRASIM_1D_1C_training_v2.parquet"
    )

    learning = pd.read_parquet(
        learning_path
    )

    learning["decision_timestamp"] = pd.to_datetime(
        learning["decision_timestamp"],
        utc=True,
    )

    print(
        f"[OK] Learning records: {len(learning)}"
    )

    evaluation_dates = learning[
        (
            learning["decision_timestamp"]
            >= START_DATE
        )
        &
        (
            learning["decision_timestamp"]
            <= END_DATE
        )
    ]["decision_timestamp"].tolist()

    print()
    print("[4] Evaluation window")

    print(
        f"[OK] From: {START_DATE}"
    )

    print(
        f"[OK] To  : {END_DATE}"
    )

    print(
        f"[OK] Predictions: {len(evaluation_dates)}"
    )

    engine = LearningEngine()

    core_results = []
    evidence_results = []

    print()
    print("[5] Running walk-forward comparison...")

    for i, timestamp in enumerate(
        evaluation_dates,
        start=1,
    ):

        # -----------------------------------------------------
        # HISTORICAL INFORMATION BOUNDARY
        # -----------------------------------------------------

        training = learning[
            learning["decision_timestamp"]
            < timestamp
        ].copy()

        if training.empty:
            continue

        row_matches = features[
            features["timestamp"] == timestamp
        ]

        if row_matches.empty:
            continue

        row = row_matches.iloc[-1]

        # -----------------------------------------------------
        # CORE MODEL
        # -----------------------------------------------------

        core_model = engine.train(
            training
        )

        core_prediction = engine.predict(
            core_model,
            row,
            timeframe="1D",
            horizon_candles=1,
        )

        # -----------------------------------------------------
        # EVIDENCE MODEL
        # -----------------------------------------------------

        evidence_model = (
            engine.train_with_evidence(
                training,
                minimum_evidence_samples=20,
            )
        )

        evidence_pattern = None

        for pattern in evidence_model["patterns"]:

            state = pattern["state"]

            if (
                state["structure"]
                == row["structure"]
                and
                state["price_direction"]
                == row["price_direction"]
                and
                state["rsi_direction"]
                == row["rsi_direction"]
                and
                state["rsi_zone"]
                == row["rsi_zone"]
            ):

                evidence_pattern = pattern
                break

        if evidence_pattern is None:

            evidence_direction = "FLAT"
            evidence_confidence = 0.0

            evidence_probability = {
                "UP": 0.0,
                "DOWN": 0.0,
                "FLAT": 1.0,
            }

        else:

            evidence_direction = (
                evidence_pattern[
                    "predicted_direction"
                ]
            )

            evidence_confidence = float(
                evidence_pattern[
                    "confidence"
                ]
            )

            evidence_probability = (
                evidence_pattern[
                    "final_probability"
                ]
            )

        # -----------------------------------------------------
        # ACTUAL NEXT CANDLE
        # -----------------------------------------------------

        future = data[
            data["timestamp"] > timestamp
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

        actual_move = (
            (close - entry)
            / entry
            * 100
        )

        actual_direction = (
            direction_from_move(
                actual_move
            )
        )

        core_correct = (
            core_prediction.direction
            == actual_direction
        )

        evidence_correct = (
            evidence_direction
            == actual_direction
        )

        core_results.append(
            {
                "timestamp": timestamp,
                "prediction":
                    core_prediction.direction,
                "actual":
                    actual_direction,
                "correct":
                    core_correct,
                "confidence":
                    core_prediction.confidence,
                "move":
                    core_prediction.expected_move_pct,
                "actual_move":
                    actual_move,
            }
        )

        evidence_results.append(
            {
                "timestamp": timestamp,
                "prediction":
                    evidence_direction,
                "actual":
                    actual_direction,
                "correct":
                    evidence_correct,
                "confidence":
                    evidence_confidence,
                "move":
                    0.0,
                "actual_move":
                    actual_move,
            }
        )

        if (
            i % 25 == 0
            or i == len(evaluation_dates)
        ):

            print(
                f"[{i:3}/{len(evaluation_dates)}] "
                f"{timestamp.date()} "
                f"CORE={core_prediction.direction} "
                f"EVIDENCE={evidence_direction} "
                f"ACTUAL={actual_direction}"
            )

    core = pd.DataFrame(
        core_results
    )

    evidence = pd.DataFrame(
        evidence_results
    )

    # =========================================================
    # RESULTS
    # =========================================================

    print()
    print("=" * 70)
    print(" WALK-FORWARD COMPARISON")
    print("=" * 70)

    # ---------------------------------------------------------
    # CORE
    # ---------------------------------------------------------

    print()
    print("CORE MODEL")
    print("-" * 45)

    print(
        f"Predictions        : {len(core)}"
    )

    print(
        f"Correct            : "
        f"{core['correct'].sum()}"
    )

    print(
        f"Accuracy           : "
        f"{core['correct'].mean() * 100:.2f}%"
    )

    print()
    print(
        core["prediction"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # EVIDENCE
    # ---------------------------------------------------------

    print()
    print("EVIDENCE MODEL")
    print("-" * 45)

    print(
        f"Predictions        : "
        f"{len(evidence)}"
    )

    print(
        f"Correct            : "
        f"{evidence['correct'].sum()}"
    )

    print(
        f"Accuracy           : "
        f"{evidence['correct'].mean() * 100:.2f}%"
    )

    print()
    print(
        evidence["prediction"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # DIRECTION ACCURACY
    # ---------------------------------------------------------

    print()
    print("DIRECTION ACCURACY")
    print("-" * 45)

    for direction in [
        "UP",
        "DOWN",
        "FLAT",
    ]:

        core_subset = core[
            core["prediction"]
            == direction
        ]

        evidence_subset = evidence[
            evidence["prediction"]
            == direction
        ]

        core_accuracy = (
            core_subset["correct"].mean()
            * 100
            if len(core_subset)
            else 0.0
        )

        evidence_accuracy = (
            evidence_subset["correct"].mean()
            * 100
            if len(evidence_subset)
            else 0.0
        )

        print(
            f"{direction:<6} "
            f"Core={core_accuracy:6.2f}% "
            f"Evidence={evidence_accuracy:6.2f}%"
        )

    # ---------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------

    print_confusion_matrix(
        "CORE CONFUSION MATRIX",
        core,
    )

    print_confusion_matrix(
        "EVIDENCE CONFUSION MATRIX",
        evidence,
    )

    # ---------------------------------------------------------
    # TRADE QUALITY
    # ---------------------------------------------------------

    print_trade_quality(
        "CORE TRADE QUALITY",
        core,
    )

    print_trade_quality(
        "EVIDENCE TRADE QUALITY",
        evidence,
    )

    # ---------------------------------------------------------
    # DIRECTIONAL EDGE
    # ---------------------------------------------------------

    print_directional_edge(
        "CORE DIRECTIONAL EDGE",
        core,
    )

    print_directional_edge(
        "EVIDENCE DIRECTIONAL EDGE",
        evidence,
    )

    # ---------------------------------------------------------
    # MOVE ERROR
    # ---------------------------------------------------------

    print()
    print("AVERAGE MOVE ERROR")
    print("-" * 45)

    core_error = (
        core["move"]
        - core["actual_move"]
    ).abs().mean()

    print(
        f"Core               : "
        f"{core_error:.3f}%"
    )

    # Evidence currently does not produce
    # a move prediction.

    print(
        "Evidence           : "
        "N/A (direction-only)"
    )

    # ---------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------

    print()
    print("EVIDENCE CONFIDENCE")
    print("-" * 45)

    for threshold in [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]:

        subset = evidence[
            evidence["confidence"]
            >= threshold
        ]

        if subset.empty:

            print(
                f">={threshold:.0%}: "
                f"no samples"
            )

        else:

            print(
                f">={threshold:.0%}: "
                f"{subset['correct'].mean() * 100:.2f}% "
                f"accuracy "
                f"({len(subset)} samples)"
            )

    # ---------------------------------------------------------
    # BEARISH BIAS CHECK
    # ---------------------------------------------------------

    print()
    print("PREDICTION BIAS")
    print("-" * 45)

    evidence_down_pct = (
        (
            evidence["prediction"]
            == "DOWN"
        ).mean()
        * 100
    )

    actual_down_pct = (
        (
            evidence["actual"]
            == "DOWN"
        ).mean()
        * 100
    )

    print(
        f"Evidence DOWN predictions : "
        f"{evidence_down_pct:.2f}%"
    )

    print(
        f"Actual DOWN outcomes      : "
        f"{actual_down_pct:.2f}%"
    )

    print(
        f"DOWN prediction bias      : "
        f"{evidence_down_pct - actual_down_pct:+.2f}%"
    )

    # ---------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------

    print()
    print("========== RESULT ==========")

    core_accuracy = (
        core["correct"].mean()
    )

    evidence_accuracy = (
        evidence["correct"].mean()
    )

    if evidence_accuracy > core_accuracy:

        print(
            "EVIDENCE > CORE"
        )

    elif evidence_accuracy == core_accuracy:

        print(
            "EVIDENCE = CORE"
        )

    else:

        print(
            "CORE > EVIDENCE"
        )


if __name__ == "__main__":
    main()