from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class TechnicalState:
    timestamp: object
    price: float

    ema_21: float | None
    ema_50: float | None
    ema_200: float | None

    ema_alignment: bool
    ema_alignment_type: str

    rsi: float | None
    rsi_direction: str
    rsi_zone: str

    structure: str | None
    price_direction: str

    previous_high: float | None
    previous_low: float | None

    distance_to_previous_high_pct: float | None
    distance_to_previous_low_pct: float | None

    above_previous_high: bool
    below_previous_low: bool

    previous_high_broken: bool
    previous_low_broken: bool

    near_previous_high: bool
    near_previous_low: bool

    new_high: bool
    new_low: bool

    bounce_from_low_pct: float | None
    rejection_from_high_pct: float | None

    support_status: str
    resistance_status: str

    continuation_signal: bool
    bounce_signal: bool


class TechnicalStateCalculator:
    """
    Calculates technical state from OHLC data.

    Responsibilities:
    - EMA 21 / 50 / 200
    - RSI
    - RSI direction / zone
    - price direction
    - HH / HL / LH / LL structure
    - previous confirmed swing high / low
    - support / resistance location
    - bounce / continuation evidence

    Does NOT:
    - fetch Dhan data
    - load/save files
    - train models
    - make predictions
    """

    def __init__(self, rsi_period: int = 14):
        self.rsi_period = rsi_period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate(data)

        result = (
            data.copy()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # =========================================================
        # PRICE DIRECTION
        # =========================================================

        result["price_direction"] = (
            result["close"]
            .diff()
            .apply(self._price_direction)
        )

        # =========================================================
        # EMA
        # =========================================================

        result["ema_21"] = result["close"].ewm(
            span=21,
            adjust=False,
            min_periods=21,
        ).mean()

        result["ema_50"] = result["close"].ewm(
            span=50,
            adjust=False,
            min_periods=50,
        ).mean()

        result["ema_200"] = result["close"].ewm(
            span=200,
            adjust=False,
            min_periods=200,
        ).mean()

        result["ema_alignment"] = (
            (result["ema_21"] > result["ema_50"])
            & (result["ema_50"] > result["ema_200"])
        )

        result["ema_alignment_type"] = "NONE"

        result.loc[
            result["ema_alignment"],
            "ema_alignment_type",
        ] = "BULLISH"

        result.loc[
            (result["ema_21"] < result["ema_50"])
            & (result["ema_50"] < result["ema_200"]),
            "ema_alignment_type",
        ] = "BEARISH"

        # =========================================================
        # RSI
        # =========================================================

        delta = result["close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / self.rsi_period,
            adjust=False,
            min_periods=self.rsi_period,
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / self.rsi_period,
            adjust=False,
            min_periods=self.rsi_period,
        ).mean()

        rs = avg_gain / avg_loss

        result["rsi"] = 100 - (
            100 / (1 + rs)
        )

        result.loc[
            (avg_loss == 0) & (avg_gain > 0),
            "rsi",
        ] = 100

        result["rsi_direction"] = "FLAT"

        result.loc[
            result["rsi"].diff() > 0,
            "rsi_direction",
        ] = "RISING"

        result.loc[
            result["rsi"].diff() < 0,
            "rsi_direction",
        ] = "FALLING"

        result["rsi_zone"] = "NEUTRAL"

        result.loc[
            result["rsi"] <= 30,
            "rsi_zone",
        ] = "OVERSOLD"

        result.loc[
            result["rsi"] >= 70,
            "rsi_zone",
        ] = "OVERBOUGHT"

        # =========================================================
        # CONFIRMED SWINGS
        #
        # IMPORTANT:
        # We do NOT use shift(-1) to decide the current candle's
        # market state.
        #
        # A swing from candle i-1 is confirmed only when candle i
        # arrives.
        # =========================================================

        raw_swing_high = (
            (result["high"] > result["high"].shift(1))
            & (result["high"] > result["high"].shift(-1))
        )

        raw_swing_low = (
            (result["low"] < result["low"].shift(1))
            & (result["low"] < result["low"].shift(-1))
        )

        confirmed_swing_high = raw_swing_high.shift(1).fillna(False)
        confirmed_swing_low = raw_swing_low.shift(1).fillna(False)

        previous_high = None
        previous_low = None

        structure_values = []
        previous_high_values = []
        previous_low_values = []

        for index, row in result.iterrows():

            structure = None

            # A swing from the previous candle is now confirmed.
            if bool(confirmed_swing_high.iloc[index]):

                swing_price = float(
                    result["high"].iloc[index - 1]
                )

                if previous_high is None:
                    structure = "INITIAL_H"
                elif swing_price > previous_high:
                    structure = "HH"
                else:
                    structure = "LH"

                previous_high = swing_price

            if bool(confirmed_swing_low.iloc[index]):

                swing_price = float(
                    result["low"].iloc[index - 1]
                )

                if previous_low is None:
                    if structure is None:
                        structure = "INITIAL_L"
                elif swing_price > previous_low:
                    if structure is None:
                        structure = "HL"
                else:
                    if structure is None:
                        structure = "LL"

                previous_low = swing_price

            structure_values.append(structure)
            previous_high_values.append(previous_high)
            previous_low_values.append(previous_low)

        result["structure_event"] = structure_values

        result["structure"] = (
            result["structure_event"]
            .ffill()
        )

        result["previous_high"] = previous_high_values
        result["previous_low"] = previous_low_values

        # =========================================================
        # MARKET LOCATION
        # =========================================================

        result["distance_to_previous_high_pct"] = (
            (
                result["close"]
                - result["previous_high"]
            )
            / result["previous_high"]
            * 100
        )

        result["distance_to_previous_low_pct"] = (
            (
                result["close"]
                - result["previous_low"]
            )
            / result["previous_low"]
            * 100
        )

        result["above_previous_high"] = (
            result["previous_high"].notna()
            & (
                result["close"]
                > result["previous_high"]
            )
        )

        result["below_previous_low"] = (
            result["previous_low"].notna()
            & (
                result["close"]
                < result["previous_low"]
            )
        )

        # A break is considered valid when price closes beyond
        # the previously established level.
        result["previous_high_broken"] = (
            result["above_previous_high"]
        )

        result["previous_low_broken"] = (
            result["below_previous_low"]
        )

        # =========================================================
        # NEAR SUPPORT / RESISTANCE
        # =========================================================

        result["near_previous_high"] = (
            result["previous_high"].notna()
            & (
                result["distance_to_previous_high_pct"]
                .abs()
                <= 2.0
            )
        )

        result["near_previous_low"] = (
            result["previous_low"].notna()
            & (
                result["distance_to_previous_low_pct"]
                .abs()
                <= 2.0
            )
        )

        # =========================================================
        # NEW HIGH / NEW LOW
        # =========================================================

        prior_high = result["high"].shift(1).cummax()
        prior_low = result["low"].shift(1).cummin()

        result["new_high"] = (
            result["high"] > prior_high
        ).fillna(False)

        result["new_low"] = (
            result["low"] < prior_low
        ).fillna(False)

        # =========================================================
        # BOUNCE / REJECTION
        # =========================================================

        result["bounce_from_low_pct"] = (
            (
                result["close"]
                - result["low"]
            )
            / result["low"]
            * 100
        )

        result["rejection_from_high_pct"] = (
            (
                result["high"]
                - result["close"]
            )
            / result["high"]
            * 100
        )

        # =========================================================
        # SUPPORT STATUS
        # =========================================================

        result["support_status"] = "UNKNOWN"

        result.loc[
            result["near_previous_low"],
            "support_status",
        ] = "NEAR_SUPPORT"

        result.loc[
            result["previous_low_broken"],
            "support_status",
        ] = "BROKEN"

        # =========================================================
        # RESISTANCE STATUS
        # =========================================================

        result["resistance_status"] = "UNKNOWN"

        result.loc[
            result["near_previous_high"],
            "resistance_status",
        ] = "NEAR_RESISTANCE"

        result.loc[
            result["previous_high_broken"],
            "resistance_status",
        ] = "BROKEN"

        # =========================================================
        # BOUNCE SIGNAL
        #
        # Price is near/under support but recovering.
        # RSI recovery is additional confirmation.
        # =========================================================

        result["bounce_signal"] = (
            (
                result["near_previous_low"]
                | result["below_previous_low"]
            )
            & (
                result["price_direction"] == "UP"
            )
            & (
                result["rsi_direction"] == "RISING"
            )
        )

        # =========================================================
        # CONTINUATION SIGNAL
        #
        # Support broken + downward pressure.
        # =========================================================

        result["continuation_signal"] = (
            result["previous_low_broken"]
            & (
                result["price_direction"] == "DOWN"
            )
            & (
                result["rsi_direction"] == "FALLING"
            )
        )

        return result

    @staticmethod
    def _price_direction(value):
        if pd.isna(value):
            return "FLAT"

        if value > 0:
            return "UP"

        if value < 0:
            return "DOWN"

        return "FLAT"

    @staticmethod
    def _validate(data: pd.DataFrame):
        required = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        }

        missing = required - set(data.columns)

        if missing:
            raise ValueError(
                f"Technical data missing columns: {sorted(missing)}"
            )

        if data.empty:
            raise ValueError(
                "Technical calculator received empty data."
            )