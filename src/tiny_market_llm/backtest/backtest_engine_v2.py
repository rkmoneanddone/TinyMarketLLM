from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass
class Prediction:
    direction: str
    expected_move_pct: float
    expected_duration: int
    confidence: float


class Predictor(Protocol):

    def predict(
        self,
        row: pd.Series,
    ) -> Prediction:
        ...


@dataclass
class BacktestResultV2:

    decision_timestamp: object

    # Prediction
    predicted_direction: str
    predicted_move_pct: float
    predicted_duration: int
    prediction_confidence: float

    # Actual final outcome
    actual_direction: str
    actual_move_pct: float

    # Actual movement in BOTH directions
    actual_upside_move_pct: float
    actual_downside_move_pct: float

    # Timing
    time_to_upside_peak: int
    time_to_downside_peak: int

    # Sustainability
    upside_sustained: bool
    downside_sustained: bool

    # Prediction-relative outcome
    favorable_move_pct: float
    adverse_move_pct: float

    passed: bool


class BacktestEngineV2:
    """
    Walk-forward historical evaluator.

    IMPORTANT:

    Actual market outcomes are calculated independently
    of the prediction.

    This module does NOT:
        - call Dhan
        - load files
        - calculate indicators
        - train models
        - promote models
    """

    def __init__(
        self,
        evaluation_horizon: int = 10,
        sustain_bars: int = 3,
        move_tolerance_pct: float = 0.5,
    ):
        self.evaluation_horizon = evaluation_horizon
        self.sustain_bars = sustain_bars
        self.move_tolerance_pct = move_tolerance_pct

    def run(
        self,
        features: pd.DataFrame,
        predictor: Predictor,
    ) -> pd.DataFrame:

        self._validate(features)

        data = (
            features
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        results = []

        last_index = (
            len(data)
            - self.evaluation_horizon
            - 1
        )

        for index in range(
            0,
            last_index + 1,
        ):

            current = data.iloc[index]

            future = data.iloc[
                index + 1:
                index + 1 + self.evaluation_horizon
            ]

            prediction = predictor.predict(
                current
            )

            result = self._evaluate(
                current,
                future,
                prediction,
            )

            results.append(result)

        return pd.DataFrame(
            [
                result.__dict__
                for result in results
            ]
        )

    def _evaluate(
        self,
        current: pd.Series,
        future: pd.DataFrame,
        prediction: Prediction,
    ) -> BacktestResultV2:

        entry = float(current["close"])

        upside = (
            (
                future["high"]
                - entry
            )
            / entry
            * 100
        )

        downside = (
            (
                future["low"]
                - entry
            )
            / entry
            * 100
        )

        final_move = (
            (
                float(
                    future["close"].iloc[-1]
                )
                - entry
            )
            / entry
            * 100
        )

        max_upside = float(
            upside.max()
        )

        max_downside = float(
            downside.min()
        )

        if final_move > self.move_tolerance_pct:

            actual_direction = "UP"

        elif final_move < -self.move_tolerance_pct:

            actual_direction = "DOWN"

        else:

            actual_direction = "FLAT"

        time_up = int(
            upside.values.argmax() + 1
        )

        time_down = int(
            downside.values.argmin() + 1
        )

        upside_sustained = (
            self._sustained_up(
                entry,
                future,
            )
        )

        downside_sustained = (
            self._sustained_down(
                entry,
                future,
            )
        )

        if prediction.direction == "UP":

            favorable = max_upside
            adverse = max_downside

        elif prediction.direction == "DOWN":

            favorable = -max_downside
            adverse = -max_upside

        else:

            favorable = 0.0
            adverse = min(
                max_downside,
                -max_upside,
            )

        passed = self._prediction_passed(
            prediction,
            actual_direction,
            favorable,
            upside_sustained,
            downside_sustained,
        )

        return BacktestResultV2(

            decision_timestamp=
                current["timestamp"],

            predicted_direction=
                prediction.direction,

            predicted_move_pct=
                prediction.expected_move_pct,

            predicted_duration=
                prediction.expected_duration,

            prediction_confidence=
                prediction.confidence,

            actual_direction=
                actual_direction,

            actual_move_pct=
                final_move,

            actual_upside_move_pct=
                max_upside,

            actual_downside_move_pct=
                max_downside,

            time_to_upside_peak=
                time_up,

            time_to_downside_peak=
                time_down,

            upside_sustained=
                upside_sustained,

            downside_sustained=
                downside_sustained,

            favorable_move_pct=
                favorable,

            adverse_move_pct=
                adverse,

            passed=
                passed,
        )

    def _sustained_up(
        self,
        entry: float,
        future: pd.DataFrame,
    ) -> bool:

        for start in range(
            0,
            len(future)
            - self.sustain_bars
            + 1,
        ):

            window = future[
                "close"
            ].iloc[
                start:
                start + self.sustain_bars
            ]

            if (
                window > entry
            ).all():

                return True

        return False

    def _sustained_down(
        self,
        entry: float,
        future: pd.DataFrame,
    ) -> bool:

        for start in range(
            0,
            len(future)
            - self.sustain_bars
            + 1,
        ):

            window = future[
                "close"
            ].iloc[
                start:
                start + self.sustain_bars
            ]

            if (
                window < entry
            ).all():

                return True

        return False

    def _prediction_passed(
        self,
        prediction: Prediction,
        actual_direction: str,
        favorable: float,
        upside_sustained: bool,
        downside_sustained: bool,
    ) -> bool:

        if prediction.direction == "FLAT":

            return (
                actual_direction == "FLAT"
            )

        if (
            prediction.direction
            != actual_direction
        ):

            return False

        if (
            favorable
            < prediction.expected_move_pct
        ):

            return False

        if prediction.direction == "UP":

            if not upside_sustained:
                return False

        if prediction.direction == "DOWN":

            if not downside_sustained:
                return False

        return True

    def _validate(
        self,
        features: pd.DataFrame,
    ) -> None:

        required = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        }

        missing = (
            required
            - set(features.columns)
        )

        if missing:

            raise ValueError(
                "Backtest V2 missing columns: "
                f"{sorted(missing)}"
            )

        if features.empty:

            raise ValueError(
                "Backtest V2 received empty data."
            )