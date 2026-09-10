from __future__ import annotations

import pandas as pd


class FeatureEngineV2:
    """
    Price Action + RSI + Market Location feature engine.

    Includes:
    - Price action
    - Confirmed swing structure
    - Support / resistance context
    - Bounce / continuation evidence
    - RSI

    No Moving Average.
    No Dhan dependency.
    No prediction.
    No backtesting.
    """

    MARKET_LOCATION_COLUMNS = [
        "previous_high",
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
    ]

    def __init__(self, rsi_period: int = 14):
        self.rsi_period = rsi_period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:

        self._validate(data)

        result = (
            data.copy()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        result = self._price_action(result)
        result = self._market_location(result)
        result = self._rsi(result)

        return result

    # =========================================================
    # PRICE ACTION
    # =========================================================

    def _price_action(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        data["price_change"] = (
            data["close"].diff()
        )

        data["price_direction"] = "FLAT"

        data.loc[
            data["price_change"] > 0,
            "price_direction",
        ] = "UP"

        data.loc[
            data["price_change"] < 0,
            "price_direction",
        ] = "DOWN"

        # -----------------------------------------------------
        # Raw swings
        # -----------------------------------------------------

        swing_high = (
            (data["high"] > data["high"].shift(1))
            &
            (data["high"] > data["high"].shift(-1))
        )

        swing_low = (
            (data["low"] < data["low"].shift(1))
            &
            (data["low"] < data["low"].shift(-1))
        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # A swing becomes usable only after the following candle
        # confirms it.
        #
        # Therefore:
        #
        # confirmed swing at candle i
        # comes from raw swing at candle i-1.
        #
        # This prevents current prediction state from using
        # the future candle.
        # -----------------------------------------------------

        confirmed_high = (
            swing_high
            .shift(1)
            .fillna(False)
            .astype(bool)
        )

        confirmed_low = (
            swing_low
            .shift(1)
            .fillna(False)
            .astype(bool)
        )

        previous_high = None
        previous_low = None

        events = []
        previous_high_values = []
        previous_low_values = []

        for index in range(len(data)):

            event = None

            if confirmed_high.iloc[index]:

                swing_price = float(
                    data["high"].iloc[index - 1]
                )

                if previous_high is None:

                    event = "INITIAL_H"

                elif swing_price > previous_high:

                    event = "HH"

                else:

                    event = "LH"

                previous_high = swing_price

            if confirmed_low.iloc[index]:

                swing_price = float(
                    data["low"].iloc[index - 1]
                )

                if previous_low is None:

                    if event is None:
                        event = "INITIAL_L"

                elif swing_price > previous_low:

                    if event is None:
                        event = "HL"

                else:

                    if event is None:
                        event = "LL"

                previous_low = swing_price

            events.append(event)
            previous_high_values.append(previous_high)
            previous_low_values.append(previous_low)

        data["structure_event"] = events

        data["structure"] = (
            data["structure_event"]
            .ffill()
        )

        data["previous_high"] = (
            previous_high_values
        )

        data["previous_low"] = (
            previous_low_values
        )

        # -----------------------------------------------------
        # Recent price movement
        # -----------------------------------------------------

        data["move_3_pct"] = (
            (
                data["close"]
                - data["close"].shift(3)
            )
            / data["close"].shift(3)
            * 100
        )

        data["move_5_pct"] = (
            (
                data["close"]
                - data["close"].shift(5)
            )
            / data["close"].shift(5)
            * 100
        )

        # -----------------------------------------------------
        # Recent range
        # -----------------------------------------------------

        data["range_5_pct"] = (
            (
                data["high"].rolling(5).max()
                -
                data["low"].rolling(5).min()
            )
            / data["close"]
            * 100
        )

        # -----------------------------------------------------
        # Candle body
        # -----------------------------------------------------

        data["candle_body_pct"] = (
            (
                data["close"]
                - data["open"]
            )
            / data["open"]
            * 100
        )

        # -----------------------------------------------------
        # Pullback from recent high
        # -----------------------------------------------------

        recent_high = (
            data["high"]
            .rolling(10)
            .max()
        )

        data["pullback_from_high_pct"] = (
            (
                data["close"]
                - recent_high
            )
            / recent_high
            * 100
        )

        # -----------------------------------------------------
        # Recovery from recent low
        # -----------------------------------------------------

        recent_low = (
            data["low"]
            .rolling(10)
            .min()
        )

        data["recovery_from_low_pct"] = (
            (
                data["close"]
                - recent_low
            )
            / recent_low
            * 100
        )

        return data

    # =========================================================
    # MARKET LOCATION
    # =========================================================

    def _market_location(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        # -----------------------------------------------------
        # Distance to previous levels
        # -----------------------------------------------------

        data["distance_to_previous_high_pct"] = (
            (
                data["close"]
                - data["previous_high"]
            )
            / data["previous_high"]
            * 100
        )

        data["distance_to_previous_low_pct"] = (
            (
                data["close"]
                - data["previous_low"]
            )
            / data["previous_low"]
            * 100
        )

        # -----------------------------------------------------
        # Level position
        # -----------------------------------------------------

        data["above_previous_high"] = (
            data["previous_high"].notna()
            &
            (
                data["close"]
                > data["previous_high"]
            )
        )

        data["below_previous_low"] = (
            data["previous_low"].notna()
            &
            (
                data["close"]
                < data["previous_low"]
            )
        )

        data["previous_high_broken"] = (
            data["above_previous_high"]
        )

        data["previous_low_broken"] = (
            data["below_previous_low"]
        )

        # -----------------------------------------------------
        # Near levels
        # -----------------------------------------------------

        data["near_previous_high"] = (
            data["previous_high"].notna()
            &
            (
                data[
                    "distance_to_previous_high_pct"
                ].abs()
                <= 2.0
            )
        )

        data["near_previous_low"] = (
            data["previous_low"].notna()
            &
            (
                data[
                    "distance_to_previous_low_pct"
                ].abs()
                <= 2.0
            )
        )

        # -----------------------------------------------------
        # New high / new low
        # -----------------------------------------------------

        prior_high = (
            data["high"]
            .shift(1)
            .cummax()
        )

        prior_low = (
            data["low"]
            .shift(1)
            .cummin()
        )

        data["new_high"] = (
            data["high"]
            > prior_high
        ).fillna(False)

        data["new_low"] = (
            data["low"]
            < prior_low
        ).fillna(False)

        # -----------------------------------------------------
        # Intracandle recovery / rejection
        # -----------------------------------------------------

        data["bounce_from_low_pct"] = (
            (
                data["close"]
                - data["low"]
            )
            / data["low"]
            * 100
        )

        data["rejection_from_high_pct"] = (
            (
                data["high"]
                - data["close"]
            )
            / data["high"]
            * 100
        )

        # -----------------------------------------------------
        # Support status
        # -----------------------------------------------------

        data["support_status"] = "UNKNOWN"

        data.loc[
            data["near_previous_low"],
            "support_status",
        ] = "NEAR_SUPPORT"

        data.loc[
            data["previous_low_broken"],
            "support_status",
        ] = "BROKEN"

        # -----------------------------------------------------
        # Resistance status
        # -----------------------------------------------------

        data["resistance_status"] = "UNKNOWN"

        data.loc[
            data["near_previous_high"],
            "resistance_status",
        ] = "NEAR_RESISTANCE"

        data.loc[
            data["previous_high_broken"],
            "resistance_status",
        ] = "BROKEN"

        # -----------------------------------------------------
        # Bounce / continuation
        #
        # RSI is intentionally not used here because _rsi()
        # runs afterward.
        #
        # The initial signals are price-action signals.
        # RSI confirmation will be added after RSI calculation.
        # -----------------------------------------------------

        data["bounce_signal"] = (
            (
                data["near_previous_low"]
                |
                data["below_previous_low"]
            )
            &
            (
                data["price_direction"] == "UP"
            )
        )

        data["continuation_signal"] = (
            data["previous_low_broken"]
            &
            (
                data["price_direction"] == "DOWN"
            )
        )

        return data

    # =========================================================
    # RSI
    # =========================================================

    def _rsi(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        delta = data["close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / self.rsi_period,
            min_periods=self.rsi_period,
            adjust=False,
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / self.rsi_period,
            min_periods=self.rsi_period,
            adjust=False,
        ).mean()

        rs = avg_gain / avg_loss

        data["rsi"] = 100 - (
            100 / (1 + rs)
        )

        data.loc[
            (avg_loss == 0)
            &
            (avg_gain > 0),
            "rsi",
        ] = 100

        data["rsi_direction"] = "FLAT"

        data.loc[
            data["rsi"].diff() > 0,
            "rsi_direction",
        ] = "RISING"

        data.loc[
            data["rsi"].diff() < 0,
            "rsi_direction",
        ] = "FALLING"

        data["rsi_overbought"] = (
            data["rsi"] >= 70
        )

        data["rsi_oversold"] = (
            data["rsi"] <= 30
        )

        data["rsi_change_3"] = (
            data["rsi"]
            - data["rsi"].shift(3)
        )

        data["rsi_zone"] = "NEUTRAL"

        data.loc[
            data["rsi"] >= 70,
            "rsi_zone",
        ] = "OVERBOUGHT"

        data.loc[
            data["rsi"] <= 30,
            "rsi_zone",
        ] = "OVERSOLD"

        # -----------------------------------------------------
        # Divergence candidates
        # -----------------------------------------------------

        data["bullish_divergence_candidate"] = (
            (data["low"] < data["low"].shift(5))
            &
            (data["rsi"] > data["rsi"].shift(5))
        )

        data["bearish_divergence_candidate"] = (
            (data["high"] > data["high"].shift(5))
            &
            (data["rsi"] < data["rsi"].shift(5))
        )

        # -----------------------------------------------------
        # RSI-confirmed bounce / continuation
        # -----------------------------------------------------

        data["bounce_signal"] = (
            data["bounce_signal"]
            &
            (
                data["rsi_direction"] == "RISING"
            )
        )

        data["continuation_signal"] = (
            data["continuation_signal"]
            &
            (
                data["rsi_direction"] == "FALLING"
            )
        )

        return data

    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate(
        self,
        data: pd.DataFrame,
    ):

        required = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

        missing = (
            required
            - set(data.columns)
        )

        if missing:

            raise ValueError(
                f"Missing OHLC columns: "
                f"{sorted(missing)}"
            )

        if data.empty:

            raise ValueError(
                "Feature Engine received empty data."
            )