from __future__ import annotations

import pandas as pd


class FeatureEngineV2:
    """
    Price Action + RSI feature engine.

    No Moving Average.
    No Dhan dependency.
    No prediction.
    No backtesting.
    """

    def __init__(self, rsi_period: int = 14):
        self.rsi_period = rsi_period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate(data)

        result = data.copy()
        result = result.sort_values("timestamp").reset_index(drop=True)

        result = self._price_action(result)
        result = self._rsi(result)

        return result

    def _price_action(self, data: pd.DataFrame) -> pd.DataFrame:

        data["price_change"] = data["close"].diff()

        data["price_direction"] = "FLAT"

        data.loc[
            data["price_change"] > 0,
            "price_direction",
        ] = "UP"

        data.loc[
            data["price_change"] < 0,
            "price_direction",
        ] = "DOWN"

        data["swing_high"] = (
            (data["high"] > data["high"].shift(1))
            & (data["high"] > data["high"].shift(-1))
        )

        data["swing_low"] = (
            (data["low"] < data["low"].shift(1))
            & (data["low"] < data["low"].shift(-1))
        )

        previous_high = None
        previous_low = None

        events = []

        for _, row in data.iterrows():

            event = None

            if row["swing_high"]:
                current = row["high"]

                if previous_high is None:
                    event = "INITIAL_H"
                elif current > previous_high:
                    event = "HH"
                else:
                    event = "LH"

                previous_high = current

            elif row["swing_low"]:
                current = row["low"]

                if previous_low is None:
                    event = "INITIAL_L"
                elif current > previous_low:
                    event = "HL"
                else:
                    event = "LL"

                previous_low = current

            events.append(event)

        data["structure_event"] = events

        data["structure"] = (
            data["structure_event"].ffill()
        )

        # Recent price movement
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

        # Recent range
        data["range_5_pct"] = (
            (
                data["high"].rolling(5).max()
                -
                data["low"].rolling(5).min()
            )
            / data["close"]
            * 100
        )

        # Candle body
        data["candle_body_pct"] = (
            (
                data["close"]
                - data["open"]
            )
            / data["open"]
            * 100
        )

        # Pullback from recent high
        recent_high = data["high"].rolling(10).max()

        data["pullback_from_high_pct"] = (
            (
                data["close"]
                - recent_high
            )
            / recent_high
            * 100
        )

        # Recovery from recent low
        recent_low = data["low"].rolling(10).min()

        data["recovery_from_low_pct"] = (
            (
                data["close"]
                - recent_low
            )
            / recent_low
            * 100
        )

        return data

    def _rsi(self, data: pd.DataFrame) -> pd.DataFrame:

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
            (avg_loss == 0) & (avg_gain > 0),
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

        # RSI movement
        data["rsi_change_3"] = (
            data["rsi"]
            - data["rsi"].shift(3)
        )

        # RSI zone
        data["rsi_zone"] = "NEUTRAL"

        data.loc[
            data["rsi"] >= 70,
            "rsi_zone",
        ] = "OVERBOUGHT"

        data.loc[
            data["rsi"] <= 30,
            "rsi_zone",
        ] = "OVERSOLD"

        # Simple divergence candidates.
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

        return data

    def _validate(self, data: pd.DataFrame):

        required = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

        missing = required - set(data.columns)

        if missing:
            raise ValueError(
                f"Missing OHLC columns: {sorted(missing)}"
            )

        if data.empty:
            raise ValueError(
                "Feature Engine received empty data."
            )