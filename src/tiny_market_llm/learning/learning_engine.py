from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd

# ============================================================
# LEARNING STATE
# ============================================================
#
# These are the raw features that the learner understands.
#
# Numeric features are converted into buckets before learning.
# This prevents every tiny numeric difference from becoming
# a completely different pattern.
# ============================================================

STATE_COLUMNS = [
    "structure",
    "price_direction",
    "rsi_direction",
    "rsi_zone",
]


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class LearningEngine:
    """
    Transparent statistical learning engine.

    Responsibilities:
    - chronological dataset splitting
    - market-state bucketing
    - historical pattern learning
    - out-of-sample evaluation
    - candidate model serialization

    This module does NOT:
    - call Dhan
    - load market data
    - calculate indicators
    - run backtests
    - modify the active model

    The learner receives an already-prepared learning dataset.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        minimum_samples: int = 20,
        promotion_threshold: float = 0.80,
    ):

        if train_ratio <= 0:
            raise ValueError("Train ratio must be > 0.")

        if validation_ratio <= 0:
            raise ValueError("Validation ratio must be > 0.")

        if train_ratio + validation_ratio >= 1:
            raise ValueError("Train + validation ratios must be < 1.")

        if minimum_samples < 1:
            raise ValueError("Minimum samples must be >= 1.")

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.minimum_samples = minimum_samples
        self.promotion_threshold = promotion_threshold

    # ========================================================
    # DATASET SPLIT
    # ========================================================

    def split(
        self,
        dataset: pd.DataFrame,
    ) -> DatasetSplit:

        self._validate_dataset(dataset)

        data = dataset.sort_values("decision_timestamp").reset_index(drop=True)

        total = len(data)

        train_end = int(total * self.train_ratio)

        validation_end = train_end + int(total * self.validation_ratio)

        train = data.iloc[:train_end].copy()

        validation = data.iloc[train_end:validation_end].copy()

        test = data.iloc[validation_end:].copy()

        if train.empty or validation.empty or test.empty:
            raise ValueError("Dataset split produced " "an empty partition.")

        return DatasetSplit(
            train=train,
            validation=validation,
            test=test,
        )

    # ========================================================
    # TRAIN
    # ========================================================

    def train(
        self,
        train_data: pd.DataFrame,
    ) -> dict:

        self._validate_dataset(train_data)

        data = self._prepare_learning_states(train_data)

        grouped = data.groupby(
            STATE_COLUMNS,
            dropna=False,
        )

        patterns = []

        for state, group in grouped:

            sample_count = len(group)

            if sample_count < self.minimum_samples:
                continue

            direction_counts = group["label_direction"].value_counts()

            total = len(group)

            direction_probabilities = {
                direction: float(
                    direction_counts.get(
                        direction,
                        0,
                    )
                    / total
                )
                for direction in [
                    "UP",
                    "DOWN",
                    "FLAT",
                ]
            }

            predicted_direction = max(
                direction_probabilities,
                key=direction_probabilities.get,
            )

            selected = group[group["label_direction"] == predicted_direction]

            if selected.empty:
                continue

            target_probability = (
                float(selected["label_target_reached"].mean())
                if "label_target_reached" in selected.columns
                else 0.0
            )

            sustain_probability = float(selected["label_sustained"].mean())

            expected_move = self._safe_mean(
                selected,
                "label_move_pct",
            )

            expected_favorable = self._safe_mean(
                selected,
                "label_favorable_move_pct",
            )

            expected_adverse = self._safe_mean(
                selected,
                "label_adverse_move_pct",
            )

            expected_duration = self._safe_mean(
                selected,
                "label_duration",
            )

            patterns.append(
                {
                    "state": {
                        column: self._clean_value(value)
                        for column, value in zip(
                            STATE_COLUMNS,
                            state,
                        )
                    },
                    "samples": sample_count,
                    "predicted_direction": predicted_direction,
                    "direction_probability": direction_probabilities,
                    "expected_move_pct": expected_move,
                    "expected_favorable_move_pct": expected_favorable,
                    "expected_adverse_move_pct": expected_adverse,
                    "expected_duration": expected_duration,
                    "sustain_probability": sustain_probability,
                    "target_probability": target_probability,
                    "confidence": float(direction_probabilities[predicted_direction]),
                }
            )

        model = {
            "schema_version": "2.0",
            "model_type": "bucketed_historical_state_pattern",
            "state_columns": STATE_COLUMNS,
            "minimum_samples": self.minimum_samples,
            "promotion_threshold": self.promotion_threshold,
            "patterns": patterns,
        }

        return model

    # ========================================================
    # EVALUATE
    # ========================================================

    def evaluate(
        self,
        model: dict,
        data: pd.DataFrame,
    ) -> dict:

        self._validate_dataset(data)

        prepared = self._prepare_learning_states(data)

        predictions = []

        pattern_map = {
            self._state_key(pattern["state"]): pattern
            for pattern in model.get(
                "patterns",
                [],
            )
        }

        for _, row in prepared.iterrows():

            key = self._state_key_from_row(row)

            pattern = pattern_map.get(key)

            if pattern is None:
                continue

            predicted = pattern["predicted_direction"]

            actual = row["label_direction"]

            predictions.append(
                {
                    "predicted": predicted,
                    "actual": actual,
                    "correct": predicted == actual,
                    "confidence": pattern["confidence"],
                    "samples": pattern["samples"],
                }
            )

        if not predictions:
            return {
                "evaluated": 0,
                "correct": 0,
                "accuracy": 0.0,
                "accuracy_pct": 0.0,
                "coverage": 0.0,
                "qualifies": False,
                "reason": "No known patterns " "in evaluation data.",
            }

        evaluated = len(predictions)

        correct = sum(item["correct"] for item in predictions)

        total = len(prepared)

        accuracy = correct / evaluated

        coverage = evaluated / total if total > 0 else 0.0

        return {
            "evaluated": evaluated,
            "correct": correct,
            "accuracy": accuracy,
            "accuracy_pct": accuracy * 100,
            "coverage": coverage,
            "coverage_pct": coverage * 100,
            "qualifies": accuracy >= self.promotion_threshold,
            "threshold": self.promotion_threshold,
        }

    # ========================================================
    # SAVE
    # ========================================================

    def save_candidate(
        self,
        model: dict,
        path: Path,
    ) -> Path:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                model,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return path
    
    def save(
        self,
        model: dict,
        path: Path,
    ) -> Path:

        return self.save_candidate(
            model,
            path,
        )

    # ========================================================
    # STATE PREPARATION
    # ========================================================

    def _prepare_learning_states(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        prepared = data.copy()

        # -----------------------------------------------
        # Categorical price-action state
        # -----------------------------------------------

        prepared["structure"] = prepared["structure"].fillna("UNKNOWN").astype(str)

        prepared["price_direction"] = (
            prepared["price_direction"].fillna("UNKNOWN").astype(str)
        )

        # -----------------------------------------------
        # Price-action buckets
        # -----------------------------------------------

        prepared["move_3_bucket"] = prepared["move_3_pct"].apply(self._bucket_move)

        prepared["move_5_bucket"] = prepared["move_5_pct"].apply(self._bucket_move)

        prepared["range_5_bucket"] = prepared["range_5_pct"].apply(self._bucket_range)

        prepared["candle_body_bucket"] = prepared["candle_body_pct"].apply(
            self._bucket_body
        )

        prepared["pullback_bucket"] = prepared["pullback_from_high_pct"].apply(
            self._bucket_pullback
        )

        prepared["recovery_bucket"] = prepared["recovery_from_low_pct"].apply(
            self._bucket_recovery
        )

        # -----------------------------------------------
        # RSI buckets
        # -----------------------------------------------

        prepared["rsi_bucket"] = prepared["rsi"].apply(self._bucket_rsi)

        prepared["rsi_direction"] = (
            prepared["rsi_direction"].fillna("UNKNOWN").astype(str)
        )

        prepared["rsi_zone"] = prepared["rsi_zone"].fillna("UNKNOWN").astype(str)

        prepared["rsi_change_3_bucket"] = prepared["rsi_change_3"].apply(
            self._bucket_rsi_change
        )

        prepared["rsi_overbought"] = (
            prepared["rsi_overbought"].fillna(False).astype(bool)
        )

        prepared["rsi_oversold"] = prepared["rsi_oversold"].fillna(False).astype(bool)

        prepared["bullish_divergence_candidate"] = (
            prepared["bullish_divergence_candidate"].fillna(False).astype(bool)
        )

        prepared["bearish_divergence_candidate"] = (
            prepared["bearish_divergence_candidate"].fillna(False).astype(bool)
        )

        return prepared

    # ========================================================
    # BUCKET FUNCTIONS
    # ========================================================

    @staticmethod
    def _bucket_move(value) -> str:

        if pd.isna(value):
            return "UNKNOWN"

        value = float(value)

        if value <= -5:
            return "STRONG_DOWN"

        if value <= -2:
            return "DOWN"

        if value < 2:
            return "FLAT"

        if value < 5:
            return "UP"

        return "STRONG_UP"

    @staticmethod
    def _bucket_range(value) -> str:

        if pd.isna(value):
            return "UNKNOWN"

        value = abs(float(value))

        if value < 2:
            return "LOW"

        if value < 5:
            return "MEDIUM"

        return "HIGH"

    @staticmethod
    def _bucket_body(value) -> str:

        if pd.isna(value):
            return "UNKNOWN"

        value = abs(float(value))

        if value < 1:
            return "SMALL"

        if value < 3:
            return "MEDIUM"

        return "LARGE"

    @staticmethod
    def _bucket_pullback(value) -> str:

        if pd.isna(value):
            return "UNKNOWN"

        value = abs(float(value))

        if value < 1:
            return "NONE"

        if value < 3:
            return "SHALLOW"

        if value < 6:
            return "MEDIUM"

        return "DEEP"

    @staticmethod
    def _bucket_recovery(value) -> str:

        if pd.isna(value):
            return "UNKNOWN"

        value = abs(float(value))

        if value < 1:
            return "LOW"

        if value < 3:
            return "MEDIUM"

        if value < 6:
            return "STRONG"

        return "VERY_STRONG"

    @staticmethod
    def _bucket_rsi(value) -> str:

        if pd.isna(value):
            return "UNKNOWN"

        value = float(value)

        if value < 30:
            return "OVERSOLD"

        if value < 40:
            return "WEAK"

        if value < 50:
            return "NEUTRAL_LOW"

        if value < 60:
            return "NEUTRAL_HIGH"

        if value < 70:
            return "STRONG"

        return "OVERBOUGHT"

    @staticmethod
    def _bucket_rsi_change(value) -> str:

        if pd.isna(value):
            return "UNKNOWN"

        value = float(value)

        if value <= -10:
            return "STRONG_FALL"

        if value <= -3:
            return "FALL"

        if value < 3:
            return "STABLE"

        if value < 10:
            return "RISE"

        return "STRONG_RISE"

    # ========================================================
    # HELPERS
    # ========================================================

    def _state_key(
        self,
        state: dict,
    ) -> tuple:

        return tuple(self._clean_value(state.get(column)) for column in STATE_COLUMNS)

    def _state_key_from_row(
        self,
        row: pd.Series,
    ) -> tuple:

        return tuple(self._clean_value(row.get(column)) for column in STATE_COLUMNS)

    @staticmethod
    def _clean_value(value):

        if pd.isna(value):
            return None

        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        return value

    @staticmethod
    def _safe_mean(
        data: pd.DataFrame,
        column: str,
    ) -> float:

        if column not in data.columns:
            return 0.0

        values = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if values.dropna().empty:
            return 0.0

        return float(values.mean())

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_dataset(
        self,
        dataset: pd.DataFrame,
    ) -> None:

        required = {
            "decision_timestamp",
            # Price action
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
            "rsi_zone",
            "rsi_change_3",
            "rsi_overbought",
            "rsi_oversold",
            "bullish_divergence_candidate",
            "bearish_divergence_candidate",
            # Labels
            "label_direction",
            "label_move_pct",
            "label_favorable_move_pct",
            "label_adverse_move_pct",
            "label_duration",
            "label_sustained",
        }

        missing = required - set(dataset.columns)

        if missing:
            raise ValueError("Learning dataset missing columns: " f"{sorted(missing)}")

        if dataset.empty:
            raise ValueError("Learning dataset is empty.")
