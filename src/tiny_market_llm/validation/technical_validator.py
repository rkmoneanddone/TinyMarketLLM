from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class TechnicalValidationResult:
    valid: bool

    direction: str

    price: float

    ema_21: float
    ema_50: float
    ema_200: float

    ema_alignment: bool
    ema_alignment_type: str

    rsi: float
    rsi_direction: str

    swing_low: Optional[float]
    swing_high: Optional[float]

    fib_1618: Optional[float]

    reasons: list[str]


class TechnicalValidator:
    """
    Technical validation using ONLY:

        - Price
        - EMA 21
        - EMA 50
        - EMA 200
        - RSI
        - Fibonacci 1.618

    No 100 EMA.
    No additional indicators.

    The validator does not make a BUY/SELL decision.
    It returns the measured technical state.
    """

    REQUIRED_COLUMNS = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    def __init__(
        self,
        rsi_period: int = 14,
    ):
        self.rsi_period = rsi_period

    def evaluate(
        self,
        data: pd.DataFrame,
        direction: Optional[str] = None,
    ) -> TechnicalValidationResult:

        frame = self._prepare(data)

        if len(frame) < 200:
            raise ValueError(
                "At least 200 candles are required."
            )

        frame["ema_21"] = (
            frame["close"]
            .ewm(
                span=21,
                adjust=False,
            )
            .mean()
        )

        frame["ema_50"] = (
            frame["close"]
            .ewm(
                span=50,
                adjust=False,
            )
            .mean()
        )

        frame["ema_200"] = (
            frame["close"]
            .ewm(
                span=200,
                adjust=False,
            )
            .mean()
        )

        frame["rsi"] = self._calculate_rsi(
            frame["close"]
        )

        frame["rsi_direction"] = (
            frame["rsi"]
            .diff()
            .apply(
                lambda x:
                    "RISING"
                    if x > 0
                    else (
                        "FALLING"
                        if x < 0
                        else "FLAT"
                    )
            )
        )

        row = frame.iloc[-1]

        price = float(row["close"])

        ema_21 = float(row["ema_21"])
        ema_50 = float(row["ema_50"])
        ema_200 = float(row["ema_200"])

        rsi = float(row["rsi"])

        rsi_direction = str(
            row["rsi_direction"]
        )

        bullish_alignment = (
            ema_21
            > ema_50
            > ema_200
        )

        bearish_alignment = (
            ema_21
            < ema_50
            < ema_200
        )

        if bullish_alignment:
            alignment_type = "BULLISH"
        elif bearish_alignment:
            alignment_type = "BEARISH"
        else:
            alignment_type = "MIXED"

        if direction is None:

            if bullish_alignment:
                detected_direction = "UP"

            elif bearish_alignment:
                detected_direction = "DOWN"

            else:
                detected_direction = "FLAT"

        else:
            detected_direction = direction.upper()

        reasons = []

        if bullish_alignment:
            reasons.append(
                "21 EMA > 50 EMA > 200 EMA"
            )

        elif bearish_alignment:
            reasons.append(
                "21 EMA < 50 EMA < 200 EMA"
            )

        else:
            reasons.append(
                "EMA alignment is mixed"
            )

        if rsi_direction == "RISING":
            reasons.append(
                "RSI is rising"
            )

        elif rsi_direction == "FALLING":
            reasons.append(
                "RSI is falling"
            )

        swing_low, swing_high = (
            self._find_recent_bullish_swing(
                frame
            )
        )

        fib_1618 = None

        if (
            swing_low is not None
            and swing_high is not None
        ):

            fib_1618 = (
                swing_low
                + 1.618
                * (
                    swing_high
                    - swing_low
                )
            )

            reasons.append(
                "Bullish 1.618 Fibonacci extension calculated"
            )

        valid = (
            bullish_alignment
            if detected_direction == "UP"
            else (
                bearish_alignment
                if detected_direction == "DOWN"
                else False
            )
        )

        return TechnicalValidationResult(
            valid=valid,
            direction=detected_direction,
            price=price,

            ema_21=ema_21,
            ema_50=ema_50,
            ema_200=ema_200,

            ema_alignment=(
                bullish_alignment
                or bearish_alignment
            ),

            ema_alignment_type=alignment_type,

            rsi=rsi,
            rsi_direction=rsi_direction,

            swing_low=swing_low,
            swing_high=swing_high,

            fib_1618=fib_1618,

            reasons=reasons,
        )

    # --------------------------------------------------
    # RSI
    # --------------------------------------------------

    def _calculate_rsi(
        self,
        close: pd.Series,
    ) -> pd.Series:

        delta = close.diff()

        gain = delta.clip(
            lower=0
        )

        loss = -delta.clip(
            upper=0
        )

        average_gain = (
            gain
            .ewm(
                alpha=1 / self.rsi_period,
                adjust=False,
            )
            .mean()
        )

        average_loss = (
            loss
            .ewm(
                alpha=1 / self.rsi_period,
                adjust=False,
            )
            .mean()
        )

        rs = (
            average_gain
            / average_loss.replace(
                0,
                float("nan"),
            )
        )

        rsi = (
            100
            - (
                100
                / (1 + rs)
            )
        )

        return rsi.fillna(50.0)

    # --------------------------------------------------
    # SWING
    # --------------------------------------------------

    def _find_recent_bullish_swing(
        self,
        data: pd.DataFrame,
    ) -> tuple[
        Optional[float],
        Optional[float],
    ]:

        if len(data) < 10:
            return None, None

        recent = data.tail(
            min(100, len(data))
        ).copy()

        swing_low = None
        swing_high = None

        low_index = None
        high_index = None

        for i in range(
            2,
            len(recent) - 2,
        ):

            low = recent["low"].iloc[i]

            if (
                low
                < recent["low"].iloc[i - 1]
                and low
                <= recent["low"].iloc[i + 1]
                and low
                < recent["low"].iloc[i - 2]
                and low
                <= recent["low"].iloc[i + 2]
            ):
                swing_low = float(low)
                low_index = i

        if low_index is None:
            return None, None

        for i in range(
            low_index + 1,
            len(recent) - 2,
        ):

            high = recent["high"].iloc[i]

            if (
                high
                > recent["high"].iloc[i - 1]
                and high
                >= recent["high"].iloc[i + 1]
                and high
                > recent["high"].iloc[i - 2]
                and high
                >= recent["high"].iloc[i + 2]
            ):
                swing_high = float(high)
                high_index = i

        if high_index is None:
            return None, None

        if high_index <= low_index:
            return None, None

        return swing_low, swing_high

    # --------------------------------------------------
    # PREPARE
    # --------------------------------------------------

    def _prepare(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        frame = data.copy()

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            utc=True,
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for column in numeric_columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        frame = (
            frame
            .dropna(
                subset=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                ]
            )
            .sort_values("timestamp")
            .drop_duplicates(
                "timestamp"
            )
            .reset_index(drop=True)
        )

        return frame
