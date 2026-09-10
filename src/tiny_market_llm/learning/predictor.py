from __future__ import annotations

import pandas as pd

from .learning_math import LearningMath
from .schemas import PredictionResult


class Predictor:
    """
    Converts a trained model into a forecast.

    Responsibilities:
    - learned direction
    - learned probabilities
    - learned move magnitude
    - learned favorable/adverse move
    - learned duration
    - target calculation

    Does NOT:
    - fetch market data
    - calculate EMA
    - calculate RSI
    - calculate support/resistance
    - evaluate future candles
    """

    def predict(
        self,
        model: dict,
        row: pd.Series,
        timeframe: str = "1D",
        horizon_candles: int = 1,
    ) -> PredictionResult:

        key = LearningMath.state_key_from_row(row)

        for pattern in model.get("patterns", []):

            if LearningMath.state_key(
                pattern["state"]
            ) != key:
                continue

            direction = pattern.get(
                "predicted_direction",
                "FLAT",
            )

            confidence = float(
                pattern.get(
                    "confidence",
                    0.0,
                )
            )

            probability = pattern.get(
                "direction_probability",
                {
                    "UP": 0.0,
                    "DOWN": 0.0,
                    "FLAT": 1.0,
                },
            )

            expected_move_pct = float(
                pattern.get(
                    "expected_move_pct",
                    0.0,
                )
            )

            expected_favorable_move_pct = float(
                pattern.get(
                    "expected_favorable_move_pct",
                    0.0,
                )
            )

            expected_adverse_move_pct = float(
                pattern.get(
                    "expected_adverse_move_pct",
                    0.0,
                )
            )

            expected_duration = int(
                pattern.get(
                    "expected_duration",
                    0,
                )
            )

            sustain_probability = float(
                pattern.get(
                    "sustain_probability",
                    0.0,
                )
            )

            target_probability = float(
                pattern.get(
                    "target_probability",
                    0.0,
                )
            )

            current_price = float(
                row["close"]
            )

            target_price = self._target_price(
                current_price,
                direction,
                expected_move_pct,
            )

            target_low, target_high = (
                self._target_range(
                    current_price,
                    direction,
                    expected_move_pct,
                )
            )

            remarks = [
                "Forecast generated from a learned historical pattern.",
                (
                    f"Horizon: {horizon_candles} "
                    f"{timeframe} candle(s)."
                ),
            ]

            return PredictionResult(
                direction=direction,
                confidence=confidence,
                probability=probability,
                timeframe=timeframe,
                horizon_candles=horizon_candles,
                expected_move_pct=expected_move_pct,
                expected_favorable_move_pct=(
                    expected_favorable_move_pct
                ),
                expected_adverse_move_pct=(
                    expected_adverse_move_pct
                ),
                expected_duration=expected_duration,
                target_price=target_price,
                target_low=target_low,
                target_high=target_high,
                sustain_probability=sustain_probability,
                target_probability=target_probability,
                remarks=remarks,
            )

        return PredictionResult(
            direction="FLAT",
            confidence=0.0,
            probability={
                "UP": 0.0,
                "DOWN": 0.0,
                "FLAT": 1.0,
            },
            timeframe=timeframe,
            horizon_candles=horizon_candles,
            remarks=[
                "No matching learned pattern was found."
            ],
        )

    def predict_with_evidence(
        self,
        model: dict,
        row: pd.Series,
        timeframe: str = "1D",
        horizon_candles: int = 1,
    ) -> PredictionResult:

        result = self.predict(
            model=model,
            row=row,
            timeframe=timeframe,
            horizon_candles=horizon_candles,
        )

        if result.remarks is None:
            result.remarks = []

        result.remarks.append(
            "Supporting evidence is supplied by separate modules."
        )

        return result

    @staticmethod
    def _target_price(
        current_price: float,
        direction: str,
        move_pct: float,
    ) -> float:

        if direction == "UP":
            return current_price * (
                1.0 + abs(move_pct) / 100.0
            )

        if direction == "DOWN":
            return current_price * (
                1.0 - abs(move_pct) / 100.0
            )

        return current_price

    @staticmethod
    def _target_range(
        current_price: float,
        direction: str,
        move_pct: float,
    ) -> tuple[float, float]:

        target = Predictor._target_price(
            current_price,
            direction,
            move_pct,
        )

        if direction == "UP":
            return current_price, target

        if direction == "DOWN":
            return target, current_price

        return current_price, current_price