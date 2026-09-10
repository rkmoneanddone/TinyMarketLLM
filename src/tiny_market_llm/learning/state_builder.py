from __future__ import annotations

import pandas as pd
from .schemas import STATE_COLUMNS, EVIDENCE_COLUMNS

class StateBuilder:
    REQUIRED_COLUMNS = {
        "decision_timestamp", "structure", "price_direction",
        "move_3_pct", "move_5_pct", "range_5_pct", "candle_body_pct",
        "pullback_from_high_pct", "recovery_from_low_pct",
        "rsi", "rsi_direction", "rsi_zone", "rsi_change_3",
        "rsi_overbought", "rsi_oversold",
        "bullish_divergence_candidate", "bearish_divergence_candidate",
        "label_direction", "label_move_pct",
        "label_favorable_move_pct", "label_adverse_move_pct",
        "label_duration", "label_sustained","previous_high",
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
    }

    def validate(self, dataset: pd.DataFrame) -> None:
        missing=self.REQUIRED_COLUMNS-set(dataset.columns)
        if missing:
            raise ValueError(f"Learning dataset missing columns: {sorted(missing)}")
        if dataset.empty:
            raise ValueError("Learning dataset is empty.")

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate(data)
        prepared=data.copy()
        prepared["structure"]=prepared["structure"].fillna("UNKNOWN").astype(str)
        prepared["price_direction"]=prepared["price_direction"].fillna("UNKNOWN").astype(str)
        prepared["move_3_bucket"]=prepared["move_3_pct"].apply(self._bucket_move)
        prepared["move_5_bucket"]=prepared["move_5_pct"].apply(self._bucket_move)
        prepared["range_5_bucket"]=prepared["range_5_pct"].apply(self._bucket_range)
        prepared["candle_body_bucket"]=prepared["candle_body_pct"].apply(self._bucket_body)
        prepared["pullback_bucket"]=prepared["pullback_from_high_pct"].apply(self._bucket_pullback)
        prepared["recovery_bucket"]=prepared["recovery_from_low_pct"].apply(self._bucket_recovery)
        prepared["rsi_bucket"]=prepared["rsi"].apply(self._bucket_rsi)
        prepared["rsi_direction"]=prepared["rsi_direction"].fillna("UNKNOWN").astype(str)
        prepared["rsi_zone"]=prepared["rsi_zone"].fillna("UNKNOWN").astype(str)
        prepared["rsi_change_3_bucket"]=prepared["rsi_change_3"].apply(self._bucket_rsi_change)
        prepared["rsi_overbought"]=prepared["rsi_overbought"].fillna(False).astype(bool)
        prepared["rsi_oversold"]=prepared["rsi_oversold"].fillna(False).astype(bool)
        prepared["bullish_divergence_candidate"]=prepared["bullish_divergence_candidate"].fillna(False).astype(bool)
        prepared["bearish_divergence_candidate"]=prepared["bearish_divergence_candidate"].fillna(False).astype(bool)
        prepared["distance_to_previous_high_bucket"] = (prepared["distance_to_previous_high_pct"].apply(self._bucket_level_distance)
        )

        prepared["below_previous_low"] = (
            prepared["below_previous_low"]
            .fillna(False)
            .astype(bool)
        )

        prepared["previous_low_broken"] = (
            prepared["previous_low_broken"]
            .fillna(False)
            .astype(bool)
        )

        prepared["near_previous_low"] = (
            prepared["near_previous_low"]
            .fillna(False)
            .astype(bool)
        )

        prepared["support_status"] = (
            prepared["support_status"]
            .fillna("UNKNOWN")
            .astype(str)
        )

        prepared["resistance_status"] = (
            prepared["resistance_status"]
            .fillna("UNKNOWN")
            .astype(str)
        )

        prepared["bounce_signal"] = (
            prepared["bounce_signal"]
            .fillna(False)
            .astype(bool)
        )

        prepared["continuation_signal"] = (
            prepared["continuation_signal"]
            .fillna(False)
            .astype(bool)
        )

        return prepared

    @staticmethod
    def _bucket_level_distance(value):
        if pd.isna(value):
            return "UNKNOWN"

        value = abs(float(value))

        if value <= 1:
            return "VERY_NEAR"

        if value <= 2:
            return "NEAR"

        if value <= 5:
            return "MEDIUM"

        return "FAR"


    @staticmethod
    def _bucket_bounce(value):
        if pd.isna(value):
            return "UNKNOWN"

        value = float(value)

        if value < 0.5:
            return "WEAK"

        if value < 1.5:
            return "MODERATE"

        if value < 3:
            return "STRONG"

        return "VERY_STRONG"


    @staticmethod
    def _bucket_rejection(value):
        if pd.isna(value):
            return "UNKNOWN"

        value = float(value)

        if value < 0.5:
            return "WEAK"

        if value < 1.5:
            return "MODERATE"

        if value < 3:
            return "STRONG"

        return "VERY_STRONG"

    @staticmethod
    def _bucket_move(value):
        if pd.isna(value): return "UNKNOWN"
        value=float(value)
        if value <= -5: return "STRONG_DOWN"
        if value <= -2: return "DOWN"
        if value < 2: return "FLAT"
        if value < 5: return "UP"
        return "STRONG_UP"

    @staticmethod
    def _bucket_range(value):
        if pd.isna(value): return "UNKNOWN"
        value=abs(float(value))
        if value < 2: return "LOW"
        if value < 5: return "MEDIUM"
        return "HIGH"

    @staticmethod
    def _bucket_body(value):
        if pd.isna(value): return "UNKNOWN"
        value=abs(float(value))
        if value < 1: return "SMALL"
        if value < 3: return "MEDIUM"
        return "LARGE"

    @staticmethod
    def _bucket_pullback(value):
        if pd.isna(value): return "UNKNOWN"
        value=abs(float(value))
        if value < 1: return "NONE"
        if value < 3: return "SHALLOW"
        if value < 6: return "MEDIUM"
        return "DEEP"

    @staticmethod
    def _bucket_recovery(value):
        if pd.isna(value): return "UNKNOWN"
        value=abs(float(value))
        if value < 1: return "LOW"
        if value < 3: return "MEDIUM"
        if value < 6: return "STRONG"
        return "VERY_STRONG"

    @staticmethod
    def _bucket_rsi(value):
        if pd.isna(value): return "UNKNOWN"
        value=float(value)
        if value < 30: return "OVERSOLD"
        if value < 40: return "WEAK"
        if value < 50: return "NEUTRAL_LOW"
        if value < 60: return "NEUTRAL_HIGH"
        if value < 70: return "STRONG"
        return "OVERBOUGHT"

    @staticmethod
    def _bucket_rsi_change(value):
        if pd.isna(value): return "UNKNOWN"
        value=float(value)
        if value <= -10: return "STRONG_FALL"
        if value <= -3: return "FALL"
        if value < 3: return "STABLE"
        if value < 10: return "RISE"
        return "STRONG_RISE"
