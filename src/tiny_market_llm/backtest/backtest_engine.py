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
    def predict(self, row: pd.Series) -> Prediction:
        ...


@dataclass
class BacktestResult:
    decision_timestamp: object

    # Prediction
    predicted_direction: str
    predicted_move_pct: float
    predicted_duration: int
    prediction_confidence: float

    # Actual market outcome
    actual_direction: str
    actual_move_pct: float
    actual_duration: int

    actual_upside_move_pct: float
    actual_downside_move_pct: float

    # Prediction performance
    prediction_favorable_move_pct: float
    prediction_adverse_move_pct: float

    sustained: bool
    passed: bool


class BacktestEngine:
    """
    Walk-forward historical backtesting.

    Responsibilities:
    - receive feature data
    - ask a Predictor for a prediction
    - look only at future rows after the decision point
    - calculate actual market movement
    - calculate prediction performance
    - produce evaluation records

    This module does NOT:
    - call Dhan
    - load files
    - save datasets
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

        self._validate_input(features)

        features = (
            features
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        results = []

        first_index = 0

        last_index = (
            len(features)
            - self.evaluation_horizon
            - 1
        )

        for index in range(
            first_index,
            last_index + 1,
        ):
            current = features.iloc[index]

            prediction = predictor.predict(
                current
            )

            future = features.iloc[
                index + 1:
                index + 1 + self.evaluation_horizon
            ]

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
    ) -> BacktestResult:

        entry_price = float(
            current["close"]
        )

        future_high = future["high"].max()
        future_low = future["low"].min()

        future_final_close = float(
            future["close"].iloc[-1]
        )

        # ==================================================
        # ACTUAL MARKET MOVEMENT
        # ==================================================
        #
        # These values are calculated ONLY from future
        # OHLC data.
        #
        # They NEVER depend on the prediction.
        # ==================================================

        upward_move = (
            (
                float(future_high)
                - entry_price
            )
            / entry_price
            * 100
        )

        downward_move = (
            (
                entry_price
                - float(future_low)
            )
            / entry_price
            * 100
        )

        final_move = (
            (
                future_final_close
                - entry_price
            )
            / entry_price
            * 100
        )

        # ==================================================
        # ACTUAL MARKET DIRECTION
        # ==================================================

        if final_move > self.move_tolerance_pct:

            actual_direction = "UP"

        elif final_move < -self.move_tolerance_pct:

            actual_direction = "DOWN"

        else:

            actual_direction = "FLAT"

        # ==================================================
        # ACTUAL MARKET OUTCOME
        # ==================================================

        actual_upside_move = upward_move

        actual_downside_move = downward_move

        actual_move_pct = final_move

        # Determine which side produced the larger move.
        if actual_upside_move >= actual_downside_move:

            actual_peak_direction = "UP"

        else:

            actual_peak_direction = "DOWN"

        actual_duration = self._time_to_peak(
            current,
            future,
            actual_peak_direction,
        )

        # ==================================================
        # ACTUAL SUSTAINED MOVEMENT
        # ==================================================

        actual_sustained = self._is_sustained(
            current,
            future,
            actual_peak_direction,
        )

        # ==================================================
        # PREDICTION PERFORMANCE
        # ==================================================

        prediction_favorable = (
            self._prediction_favorable_move(
                prediction,
                actual_upside_move,
                actual_downside_move,
            )
        )

        prediction_adverse = (
            actual_downside_move
            if prediction.direction == "UP"
            else actual_upside_move
            if prediction.direction == "DOWN"
            else 0.0
        )

        prediction_sustained = self._is_sustained(
            current,
            future,
            prediction.direction,
        )

        passed = self._prediction_passed(
            prediction,
            actual_direction,
            prediction_favorable,
            prediction_sustained,
        )

        # ==================================================
        # RESULT
        # ==================================================

        return BacktestResult(

            decision_timestamp=(
                current["timestamp"]
            ),

            # Prediction
            predicted_direction=(
                prediction.direction
            ),

            predicted_move_pct=(
                prediction.expected_move_pct
            ),

            predicted_duration=(
                prediction.expected_duration
            ),

            prediction_confidence=(
                prediction.confidence
            ),

            # Actual market outcome
            actual_direction=(
                actual_direction
            ),

            actual_move_pct=(
                actual_move_pct
            ),

            actual_duration=(
                actual_duration
            ),

            actual_upside_move_pct=(
                actual_upside_move
            ),

            actual_downside_move_pct=(
                actual_downside_move
            ),

            # Prediction performance
            prediction_favorable_move_pct=(
                prediction_favorable
            ),

            prediction_adverse_move_pct=(
                prediction_adverse
            ),

            sustained=(
                actual_sustained
            ),

            passed=(
                passed
            ),
        )

    def _time_to_peak(
        self,
        current: pd.Series,
        future: pd.DataFrame,
        direction: str,
    ) -> int:

        entry = float(
            current["close"]
        )

        if direction == "UP":

            moves = (
                (
                    future["high"]
                    - entry
                )
                / entry
                * 100
            )

        elif direction == "DOWN":

            moves = (
                (
                    entry
                    - future["low"]
                )
                / entry
                * 100
            )

        else:

            return 0

        if moves.empty:

            return 0

        return int(
            moves.values.argmax() + 1
        )

    def _is_sustained(
        self,
        current: pd.Series,
        future: pd.DataFrame,
        direction: str,
    ) -> bool:

        if len(future) < self.sustain_bars:

            return False

        entry = float(
            current["close"]
        )

        if direction == "UP":

            closes = future["close"].iloc[
                :self.sustain_bars
            ]

            return bool(
                (
                    closes > entry
                ).all()
            )

        if direction == "DOWN":

            closes = future["close"].iloc[
                :self.sustain_bars
            ]

            return bool(
                (
                    closes < entry
                ).all()
            )

        return False

    def _prediction_favorable_move(
        self,
        prediction: Prediction,
        upward_move: float,
        downward_move: float,
    ) -> float:

        if prediction.direction == "UP":

            return upward_move

        if prediction.direction == "DOWN":

            return downward_move

        return 0.0

    def _prediction_passed(
        self,
        prediction: Prediction,
        actual_direction: str,
        favorable_move: float,
        sustained: bool,
    ) -> bool:

        if prediction.direction == "FLAT":

            return (
                actual_direction
                == "FLAT"
            )

        if prediction.direction != actual_direction:

            return False

        if (
            favorable_move
            < prediction.expected_move_pct
        ):

            return False

        if not sustained:

            return False

        return True

    def _validate_input(
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
                "Backtest input missing columns: "
                f"{sorted(missing)}"
            )

        if features.empty:

            raise ValueError(
                "Backtest received empty data."
            )