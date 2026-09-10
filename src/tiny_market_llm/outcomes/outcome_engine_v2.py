from __future__ import annotations

from pathlib import Path

import pandas as pd

STATE_COLUMNS_V2 = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",

    # Price Action
    "structure_event",
    "structure",
    "price_direction",
    "move_3_pct",
    "move_5_pct",
    "range_5_pct",
    "candle_body_pct",
    "pullback_from_high_pct",
    "recovery_from_low_pct",

    # Market Location
    "previous_high",
    "previous_low",
    "distance_to_previous_high_pct",
    "distance_to_previous_low_pct",
    "above_previous_high",
    "below_previous_low",
    "previous_high_broken",
    "previous_low_broken",
    "near_previous_high",
    "near_previous_low",
    "new_high",
    "new_low",
    "bounce_from_low_pct",
    "rejection_from_high_pct",
    "support_status",
    "resistance_status",
    "bounce_signal",
    "continuation_signal",

    # RSI
    "rsi",
    "rsi_direction",
    "rsi_overbought",
    "rsi_oversold",
    "rsi_change_3",
    "rsi_zone",
    "bullish_divergence_candidate",
    "bearish_divergence_candidate",
]


RESULT_COLUMNS = [
    "decision_timestamp",
    "horizon_candles",

    "predicted_direction",
    "predicted_move_pct",
    "predicted_duration",
    "prediction_confidence",

    "actual_direction",
    "actual_move_pct",

    "actual_upside_move_pct",
    "actual_downside_move_pct",

    "time_to_upside_peak",
    "time_to_downside_peak",

    "upside_sustained",
    "downside_sustained",

    "favorable_move_pct",
    "adverse_move_pct",

    "passed",
]




class OutcomeEngineV2:
    """
    Builds the V2 learning dataset.

    V2 learning state:
        Price Action + RSI

    This module does NOT:
        - call Dhan
        - load market data
        - calculate features
        - make predictions
        - train models
        - promote models
    """

    def build(
        self,
        features: pd.DataFrame,
        results: pd.DataFrame,
    ) -> pd.DataFrame:

        self._validate_features(features)
        self._validate_results(results)

        state = features[STATE_COLUMNS_V2].copy()

        state = state.rename(columns={"timestamp": "decision_timestamp"})

        outcomes = results.copy()

        dataset = outcomes.merge(
            state,
            on="decision_timestamp",
            how="left",
            validate="one_to_one",
        )

        missing_state = dataset["close"].isna()

        if missing_state.any():
            count = int(missing_state.sum())

            raise ValueError(
                f"Could not attach market state " f"to {count} outcome records."
            )

        # Actual outcomes become the learning labels.
        dataset["label_direction"] = dataset["actual_direction"]

        dataset["horizon_candles"] = dataset["horizon_candles"].astype(int)
        dataset["label_move_pct"] = dataset["actual_move_pct"]
        dataset["label_favorable_move_pct"] = dataset["actual_upside_move_pct"]

        dataset["label_adverse_move_pct"] = dataset["actual_downside_move_pct"]

        if dataset["label_direction"].eq("UP").any():
            dataset["label_duration"] = dataset.apply(
                lambda row: (
                    row["time_to_upside_peak"]
                    if row["label_direction"] == "UP"
                    else row["time_to_downside_peak"]
                    if row["label_direction"] == "DOWN"
                    else 1
                ),
                axis=1,
        )

        dataset["label_sustained"] = (
            (
                (dataset["label_direction"] == "UP")
                & dataset["upside_sustained"]
            )
            |
            (
                (dataset["label_direction"] == "DOWN")
                & dataset["downside_sustained"]
            )
        )

        dataset["label_passed"] = dataset["passed"]

        dataset["label_target_reached"] = dataset["label_favorable_move_pct"] >= 2.0

        dataset["schema_version"] = "2.0"

        output_columns = [
            "schema_version",
            "decision_timestamp",
            "horizon_candles",
            # Market state
            "open",
            "high",
            "low",
            "close",
            "volume",
            # Price Action
            "structure_event",
            "structure",
            "price_direction",
            "move_3_pct",
            "move_5_pct",
            "range_5_pct",
            "candle_body_pct",
            "pullback_from_high_pct",
            "recovery_from_low_pct",
            # Market Location
            "previous_high",
            "previous_low",
            "distance_to_previous_high_pct",
            "distance_to_previous_low_pct",
            "above_previous_high",
            "below_previous_low",
            "previous_high_broken",
            "previous_low_broken",
            "near_previous_high",
            "near_previous_low",
            "new_high",
            "new_low",
            "bounce_from_low_pct",
            "rejection_from_high_pct",
            "support_status",
            "resistance_status",
            "bounce_signal",
            "continuation_signal",
            # RSI
            "rsi",
            "rsi_direction",
            "rsi_overbought",
            "rsi_oversold",
            "rsi_change_3",
            "rsi_zone",
            "bullish_divergence_candidate",
            "bearish_divergence_candidate",
            # Prediction
            "predicted_direction",
            "predicted_move_pct",
            "predicted_duration",
            "prediction_confidence",
            # Actual outcome / labels
            "label_direction",
            "label_move_pct",
            "label_favorable_move_pct",
            "label_adverse_move_pct",
            "label_duration",
            "label_sustained",
            "label_target_reached",
            "label_passed",
        ]

        dataset = dataset[output_columns]

        return dataset.reset_index(drop=True)

    def save(
        self,
        dataset: pd.DataFrame,
        path: Path,
    ) -> Path:

        if dataset.empty:
            raise ValueError("Cannot save empty V2 dataset.")

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset.to_parquet(
            path,
            index=False,
        )

        return path

    def _validate_features(
        self,
        features: pd.DataFrame,
    ) -> None:

        missing = set(STATE_COLUMNS_V2) - set(features.columns)

        if missing:
            raise ValueError(
                "Feature Engine V2 output is missing: " f"{sorted(missing)}"
            )

    def _validate_results(
        self,
        results: pd.DataFrame,
    ) -> None:

        missing = set(RESULT_COLUMNS) - set(results.columns)

        if missing:
            raise ValueError("Backtest results are missing: " f"{sorted(missing)}")

        if results.empty:
            raise ValueError("Backtest results are empty.")

        if (results["horizon_candles"] < 1).any():
            raise ValueError(
                "horizon_candles must be >= 1."
            )
