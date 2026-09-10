from __future__ import annotations

from typing import Optional

import pandas as pd

from .schemas import LevelEvidence


class SupportResistanceEvidence:
    """
    Identifies previous price levels and their current status.

    Responsibilities:
    - previous high / low
    - distance from current price
    - broken / nearby / above level classification
    - bounce / continuation evidence
    - human-readable remarks

    Does NOT:
    - fetch market data
    - calculate EMA
    - calculate RSI
    - make UP/DOWN predictions
    - train models
    """

    def __init__(
        self,
        nearby_threshold_pct: float = 2.0,
    ):
        self.nearby_threshold_pct = nearby_threshold_pct

    def analyze(
        self,
        current_price: float,
        previous_high: Optional[float],
        previous_low: Optional[float],
    ) -> LevelEvidence:

        if current_price <= 0:
            raise ValueError(
                "Current price must be greater than zero."
            )

        distance_to_high = self._distance_pct(
            current_price,
            previous_high,
        )

        distance_to_low = self._distance_pct(
            current_price,
            previous_low,
        )

        high_status = self._classify_high(
            current_price,
            previous_high,
        )

        low_status = self._classify_low(
            current_price,
            previous_low,
        )

        remarks = []

        bounce_possible = False
        continuation_possible = False

        # ---------------------------------------------
        # Previous HIGH
        # ---------------------------------------------

        if high_status == "BELOW_NEAR":

            remarks.append(
                "Price is approaching previous major high."
            )

            remarks.append(
                "Previous high may act as resistance."
            )

        elif high_status == "ABOVE":

            remarks.append(
                "Price has broken above the previous major high."
            )

            remarks.append(
                "Previous high may change from resistance to support."
            )

        elif high_status == "BELOW_FAR":

            remarks.append(
                "Previous major high is above current price."
            )

        # ---------------------------------------------
        # Previous LOW
        # ---------------------------------------------

        if low_status == "ABOVE_NEAR":

            bounce_possible = True

            remarks.append(
                "Price is approaching previous major low."
            )

            remarks.append(
                "Previous low may act as support."
            )

            remarks.append(
                "Bounce is possible near previous support."
            )

        elif low_status == "BELOW":

            continuation_possible = True

            remarks.append(
                "Price is below the previous major low."
            )

            remarks.append(
                "Previous low has already been broken."
            )

            remarks.append(
                "Downside continuation remains possible."
            )

            remarks.append(
                "A bounce may still occur after a new low."
            )

        elif low_status == "ABOVE_FAR":

            remarks.append(
                "Previous major low is below current price."
            )

        return LevelEvidence(
            current_price=current_price,
            previous_high=previous_high,
            previous_low=previous_low,
            distance_to_high_pct=distance_to_high,
            distance_to_low_pct=distance_to_low,
            high_status=high_status,
            low_status=low_status,
            bounce_possible=bounce_possible,
            continuation_possible=continuation_possible,
            remarks=remarks,
        )

    def _classify_high(
        self,
        current_price: float,
        previous_high: Optional[float],
    ) -> str:

        if previous_high is None:
            return "UNKNOWN"

        if current_price >= previous_high:
            return "ABOVE"

        distance = self._distance_pct(
            current_price,
            previous_high,
        )

        if distance is not None and distance <= self.nearby_threshold_pct:
            return "BELOW_NEAR"

        return "BELOW_FAR"

    def _classify_low(
        self,
        current_price: float,
        previous_low: Optional[float],
    ) -> str:

        if previous_low is None:
            return "UNKNOWN"

        if current_price < previous_low:
            return "BELOW"

        distance = self._distance_pct(
            current_price,
            previous_low,
        )

        if distance is not None and distance <= self.nearby_threshold_pct:
            return "ABOVE_NEAR"

        return "ABOVE_FAR"

    @staticmethod
    def _distance_pct(
        current_price: float,
        level: Optional[float],
    ) -> Optional[float]:

        if level is None or pd.isna(level):
            return None

        return (
            abs(current_price - float(level))
            / current_price
        ) * 100.0
