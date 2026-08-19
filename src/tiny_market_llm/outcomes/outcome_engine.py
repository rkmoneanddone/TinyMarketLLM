from __future__ import annotations

from pathlib import Path

import pandas as pd


STATE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "structure_event",
    "structure",
    "price_direction",
    "rsi",
    "rsi_direction",
    "rsi_overbought",
    "rsi_oversold",
    "ma",
    "ma_direction",
    "price_vs_ma",
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
    "max_favorable_move_pct",
    "max_adverse_move_pct",
    "sustained",
    "passed",
]


class OutcomeEngine:
    """
    Combines historical market state with backtest outcomes
    to create learning records.

    This module does NOT:
    - call Dhan
    - calculate indicators
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

        state = (
            features[STATE_COLUMNS]
            .copy()
        )

        outcomes = results.copy()

        state = state.rename(
            columns={
                "timestamp": "decision_timestamp"
            }
        )

        dataset = outcomes.merge(
            state,
            on="decision_timestamp",
            how="left",
            validate="one_to_one",
        )

        if dataset["close"].isna().any():
            missing = int(
                dataset["close"].isna().sum()
            )

            raise ValueError(
                f"Could not attach market state to "
                f"{missing} outcome records."
            )

        dataset["label_direction"] = (
            dataset["actual_direction"]
        )

        dataset["label_move_pct"] = (
            dataset["actual_move_pct"]
        )

        dataset["label_favorable_move_pct"] = (
            dataset["max_favorable_move_pct"]
        )

        dataset["label_adverse_move_pct"] = (
            dataset["max_adverse_move_pct"]
        )

        dataset["label_sustained"] = (
            dataset["sustained"]
        )

        dataset["label_passed"] = (
            dataset["passed"]
        )

        dataset["label_duration"] = (
            dataset["actual_duration"]
        )

        dataset["label_target_reached"] = (
            dataset["max_favorable_move_pct"]
            >= dataset["predicted_move_pct"]
        )

        dataset["schema_version"] = "1.0"

        dataset = dataset[
            [
                "schema_version",
                "decision_timestamp",

                "open",
                "high",
                "low",
                "close",
                "volume",

                "structure_event",
                "structure",
                "price_direction",

                "rsi",
                "rsi_direction",
                "rsi_overbought",
                "rsi_oversold",

                "ma",
                "ma_direction",
                "price_vs_ma",

                "predicted_direction",
                "predicted_move_pct",
                "predicted_duration",
                "prediction_confidence",

                "label_direction",
                "label_move_pct",
                "label_favorable_move_pct",
                "label_adverse_move_pct",
                "label_duration",
                "label_sustained",
                "label_target_reached",
                "label_passed",
            ]
        ]

        return dataset.reset_index(drop=True)

    def save(
        self,
        dataset: pd.DataFrame,
        path: Path,
    ) -> Path:

        if dataset.empty:
            raise ValueError(
                "Cannot save an empty learning dataset."
            )

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

        missing = set(STATE_COLUMNS) - set(
            features.columns
        )

        if missing:
            raise ValueError(
                f"Feature dataset missing columns: "
                f"{sorted(missing)}"
            )

    def _validate_results(
        self,
        results: pd.DataFrame,
    ) -> None:

        missing = set(RESULT_COLUMNS) - set(
            results.columns
        )

        if missing:
            raise ValueError(
                f"Backtest results missing columns: "
                f"{sorted(missing)}"
            )

        if results.empty:
            raise ValueError(
                "Backtest results are empty."
            )