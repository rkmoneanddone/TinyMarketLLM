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
    "predicted_direction",
    "predicted_move_pct",
    "predicted_duration",
    "prediction_confidence",
    "actual_direction",
    "actual_move_pct",
    "actual_duration",
    "actual_upside_move_pct",
    "actual_downside_move_pct",
    "sustained",
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

        dataset["label_move_pct"] = dataset["actual_move_pct"]
        dataset["label_favorable_move_pct"] = dataset["actual_upside_move_pct"]

        dataset["label_adverse_move_pct"] = dataset["actual_downside_move_pct"]

        dataset["label_duration"] = dataset["actual_duration"]

        dataset["label_sustained"] = dataset["sustained"]

        dataset["label_passed"] = dataset["passed"]

        dataset["label_target_reached"] = dataset["label_favorable_move_pct"] >= 2.0

        dataset["schema_version"] = "2.0"

        output_columns = [
            "schema_version",
            "decision_timestamp",
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
