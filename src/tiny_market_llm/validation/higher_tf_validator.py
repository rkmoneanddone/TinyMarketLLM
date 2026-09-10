from __future__ import annotations

from typing import Optional

from .schemas import ValidationResult


class HigherTFValidator:
    """
    Independent higher-timeframe validation module.

    MA alignment:

        Bullish:
            21 MA > 50 MA > 200 MA

        Bearish:
            21 MA < 50 MA < 200 MA

    Structure:

        Bullish:
            HH / HL

        Bearish:
            LH / LL

    This module receives already-calculated values.
    It does not load market data or calculate indicators.
    """

    def validate(
        self,
        *,
        current_price: float,
        direction: str,
        structure: str,
        ma_21: float,
        ma_50: float,
        ma_200: float,
        previous_high: Optional[float] = None,
        previous_low: Optional[float] = None,
        swing_high: Optional[float] = None,
        swing_low: Optional[float] = None,
    ) -> ValidationResult:

        direction = str(direction).upper()
        structure = str(structure).upper()

        self._validate_price(current_price)
        self._validate_ma(ma_21, ma_50, ma_200)

        reasons = []

        # --------------------------------------------------
        # MA ALIGNMENT
        # --------------------------------------------------

        bullish_ma = (
            ma_21 > ma_50
            and ma_50 > ma_200
        )

        bearish_ma = (
            ma_21 < ma_50
            and ma_50 < ma_200
        )

        if bullish_ma:
            ma_alignment_type = "BULLISH"
        elif bearish_ma:
            ma_alignment_type = "BEARISH"
        else:
            ma_alignment_type = "MIXED"

        ma_alignment = (
            direction == "UP" and bullish_ma
        ) or (
            direction == "DOWN" and bearish_ma
        )

        if bullish_ma:
            reasons.append(
                "21 MA > 50 MA > 200 MA"
            )
        elif bearish_ma:
            reasons.append(
                "21 MA < 50 MA < 200 MA"
            )
        else:
            reasons.append(
                "MA alignment is mixed"
            )

        # --------------------------------------------------
        # STRUCTURE
        # --------------------------------------------------

        bullish_structure = structure in {
            "HH",
            "HL",
        }

        bearish_structure = structure in {
            "LH",
            "LL",
        }

        structure_valid = (
            direction == "UP"
            and bullish_structure
        ) or (
            direction == "DOWN"
            and bearish_structure
        )

        if direction == "UP":
            if structure == "HH":
                reasons.append("Bullish HH structure")
            elif structure == "HL":
                reasons.append("Bullish HL structure")
            else:
                reasons.append(
                    "Structure is not bullish"
                )

        elif direction == "DOWN":
            if structure == "LH":
                reasons.append("Bearish LH structure")
            elif structure == "LL":
                reasons.append("Bearish LL structure")
            else:
                reasons.append(
                    "Structure is not bearish"
                )

        # --------------------------------------------------
        # PREVIOUS HIGH / LOW
        # --------------------------------------------------

        previous_level = None
        previous_level_distance_pct = None
        target = None
        target_type = None

        if direction == "UP":

            if previous_high is not None:
                self._validate_price(previous_high)

                if previous_high > current_price:

                    previous_level = float(
                        previous_high
                    )

                    previous_level_distance_pct = (
                        (
                            previous_high
                            - current_price
                        )
                        / current_price
                        * 100.0
                    )

                    target = previous_level
                    target_type = "PREVIOUS_HIGH"

                    reasons.append(
                        "Previous high is above current price"
                    )

                else:
                    reasons.append(
                        "Previous high is not above current price"
                    )

        elif direction == "DOWN":

            if previous_low is not None:
                self._validate_price(previous_low)

                if previous_low < current_price:

                    previous_level = float(
                        previous_low
                    )

                    previous_level_distance_pct = (
                        (
                            current_price
                            - previous_low
                        )
                        / current_price
                        * 100.0
                    )

                    target = previous_level
                    target_type = "PREVIOUS_LOW"

                    reasons.append(
                        "Previous low is below current price"
                    )

                else:
                    reasons.append(
                        "Previous low is not below current price"
                    )

        # --------------------------------------------------
        # FIBONACCI 1.618
        # --------------------------------------------------

        fib_1618 = None

        if (
            swing_high is not None
            and swing_low is not None
        ):

            self._validate_price(swing_high)
            self._validate_price(swing_low)

            if swing_high > swing_low:

                swing_range = (
                    swing_high - swing_low
                )

                if direction == "UP":

                    fib_1618 = (
                        swing_low
                        + swing_range * 1.618
                    )

                    reasons.append(
                        "Bullish 1.618 Fibonacci extension calculated"
                    )

                elif direction == "DOWN":

                    fib_1618 = (
                        swing_high
                        - swing_range * 1.618
                    )

                    reasons.append(
                        "Bearish 1.618 Fibonacci extension calculated"
                    )

        fib_target = fib_1618

        if (
            structure_valid
            and fib_1618 is not None
        ):
            reasons.append(
                "Structure supports Fibonacci extension validation"
            )

        # --------------------------------------------------
        # FINAL VALIDATION
        # --------------------------------------------------

        valid = (
            ma_alignment
            and structure_valid
            and previous_level is not None
        )

        return ValidationResult(
            valid=valid,
            direction=direction,
            ma_alignment=ma_alignment,
            ma_alignment_type=ma_alignment_type,
            structure_valid=structure_valid,
            previous_level=previous_level,
            previous_level_distance_pct=(
                previous_level_distance_pct
            ),
            fib_1618=fib_1618,
            target=target,
            target_type=target_type,
            fib_target=fib_target,
            reasons=reasons,
        )

    @staticmethod
    def _validate_price(
        value: float,
    ) -> None:

        if value is None:
            return

        if value <= 0:
            raise ValueError(
                "Price values must be greater than zero."
            )

    @staticmethod
    def _validate_ma(
        ma_21: float,
        ma_50: float,
        ma_200: float,
    ) -> None:

        if ma_21 <= 0:
            raise ValueError(
                "21 MA must be greater than zero."
            )

        if ma_50 <= 0:
            raise ValueError(
                "50 MA must be greater than zero."
            )

        if ma_200 <= 0:
            raise ValueError(
                "200 MA must be greater than zero."
            )
