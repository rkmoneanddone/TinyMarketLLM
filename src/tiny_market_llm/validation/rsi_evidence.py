from __future__ import annotations

from dataclasses import dataclass, field

from typing import List, Optional

import pandas as pd


@dataclass
class RSIEvidence:
    rsi: Optional[float]
    rsi_direction: str
    rsi_zone: str

    bullish_divergence_candidate: bool
    bearish_divergence_candidate: bool
    bullish_double_bottom_candidate: bool

    remarks: List[str] = field(default_factory=list)


class RSIBehaviorAnalyzer:
    """
    Analyzes RSI behavior as supporting evidence.

    Responsibilities:
    - RSI level
    - RSI direction
    - RSI zone
    - possible bullish divergence
    - possible bearish divergence
    - possible RSI double bottom
    - descriptive remarks

    Does NOT:
    - make predictions
    - calculate EMA
    - calculate support/resistance
    - train models
    """

    def __init__(
        self,
        oversold: float = 30.0,
        overbought: float = 70.0,
        divergence_lookback: int = 10,
        double_bottom_tolerance: float = 3.0,
    ):
        self.oversold = oversold
        self.overbought = overbought
        self.divergence_lookback = divergence_lookback
        self.double_bottom_tolerance = double_bottom_tolerance

    def analyze(
        self,
        data: pd.DataFrame,
    ) -> RSIEvidence:

        self._validate(data)

        row = data.iloc[-1]

        rsi = (
            None
            if pd.isna(row["rsi"])
            else float(row["rsi"])
        )

        rsi_direction = str(
            row.get("rsi_direction", "UNKNOWN")
        )

        rsi_zone = self._zone(rsi)

        bullish_divergence = (
            self._bullish_divergence(data)
        )

        bearish_divergence = (
            self._bearish_divergence(data)
        )

        bullish_double_bottom = (
            self._bullish_double_bottom(data)
        )

        remarks = []

        if rsi is not None:

            if rsi <= self.oversold:

                remarks.append(
                    "RSI is oversold."
                )

            elif rsi < 40:

                remarks.append(
                    "RSI is weak and approaching oversold."
                )

            elif rsi >= self.overbought:

                remarks.append(
                    "RSI is overbought."
                )

        if rsi_direction == "FALLING":

            remarks.append(
                "RSI is falling."
            )

        elif rsi_direction == "RISING":

            remarks.append(
                "RSI is rising."
            )

        if bullish_divergence:

            remarks.append(
                "Bullish RSI-price divergence candidate detected."
            )

        if bearish_divergence:

            remarks.append(
                "Bearish RSI-price divergence candidate detected."
            )

        if bullish_double_bottom:

            remarks.append(
                "RSI double-bottom candidate detected."
            )

        return RSIEvidence(
            rsi=rsi,
            rsi_direction=rsi_direction,
            rsi_zone=rsi_zone,
            bullish_divergence_candidate=(
                bullish_divergence
            ),
            bearish_divergence_candidate=(
                bearish_divergence
            ),
            bullish_double_bottom_candidate=(
                bullish_double_bottom
            ),
            remarks=remarks,
        )

    def _bullish_divergence(
        self,
        data: pd.DataFrame,
    ) -> bool:

        if len(data) < self.divergence_lookback:
            return False

        window = data.tail(
            self.divergence_lookback
        ).copy()

        if "rsi" not in window.columns:
            return False

        window = window.dropna(
            subset=["low", "rsi"]
        )

        if len(window) < 4:
            return False

        midpoint = len(window) // 2

        first = window.iloc[:midpoint]
        second = window.iloc[midpoint:]

        first_low_idx = first["low"].idxmin()
        second_low_idx = second["low"].idxmin()

        first_low = float(
            window.loc[first_low_idx, "low"]
        )

        second_low = float(
            window.loc[second_low_idx, "low"]
        )

        first_rsi = float(
            window.loc[first_low_idx, "rsi"]
        )

        second_rsi = float(
            window.loc[second_low_idx, "rsi"]
        )

        # Price makes a lower low while RSI makes
        # a higher low.
        return (
            second_low < first_low
            and second_rsi > first_rsi
        )

    def _bearish_divergence(
        self,
        data: pd.DataFrame,
    ) -> bool:

        if len(data) < self.divergence_lookback:
            return False

        window = data.tail(
            self.divergence_lookback
        ).copy()

        if "rsi" not in window.columns:
            return False

        window = window.dropna(
            subset=["high", "rsi"]
        )

        if len(window) < 4:
            return False

        midpoint = len(window) // 2

        first = window.iloc[:midpoint]
        second = window.iloc[midpoint:]

        first_high_idx = first["high"].idxmax()
        second_high_idx = second["high"].idxmax()

        first_high = float(
            window.loc[first_high_idx, "high"]
        )

        second_high = float(
            window.loc[second_high_idx, "high"]
        )

        first_rsi = float(
            window.loc[first_high_idx, "rsi"]
        )

        second_rsi = float(
            window.loc[second_high_idx, "rsi"]
        )

        # Price makes a higher high while RSI makes
        # a lower high.
        return (
            second_high > first_high
            and second_rsi < first_rsi
        )

    def _bullish_double_bottom(
        self,
        data: pd.DataFrame,
    ) -> bool:

        if len(data) < self.divergence_lookback:
            return False

        window = data.tail(
            self.divergence_lookback
        ).copy()

        window = window.dropna(
            subset=["rsi"]
        )

        if len(window) < 5:
            return False

        values = window["rsi"].tolist()

        # Find two local RSI lows.
        lows = []

        for i in range(1, len(values) - 1):

            if (
                values[i] <= values[i - 1]
                and values[i] <= values[i + 1]
            ):
                lows.append(i)

        if len(lows) < 2:
            return False

        first = lows[-2]
        second = lows[-1]

        first_value = values[first]
        second_value = values[second]

        if abs(
            first_value - second_value
        ) > self.double_bottom_tolerance:
            return False

        # Require recovery after second low.
        if second >= len(values) - 1:
            return False

        recovery = values[-1] > second_value

        return recovery

    def _zone(
        self,
        rsi: Optional[float],
    ) -> str:

        if rsi is None:
            return "UNKNOWN"

        if rsi <= self.oversold:
            return "OVERSOLD"

        if rsi < 40:
            return "WEAK"

        if rsi < 60:
            return "NEUTRAL"

        if rsi < self.overbought:
            return "STRONG"

        return "OVERBOUGHT"

    @staticmethod
    def _validate(
        data: pd.DataFrame,
    ) -> None:

        required = {
            "low",
            "high",
            "rsi",
            "rsi_direction",
        }

        missing = (
            required - set(data.columns)
        )

        if missing:
            raise ValueError(
                "RSI evidence data missing columns: "
                f"{sorted(missing)}"
            )

        if data.empty:
            raise ValueError(
                "RSI evidence received empty data."
            )
