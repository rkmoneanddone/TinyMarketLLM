from __future__ import annotations

import pandas as pd


class FeatureEngine:
    """
    Converts normalized OHLC data into market-behavior features.

    This module does NOT:
    - call Dhan
    - load files
    - save files
    - make predictions
    - backtest
    - train models

    Input:
        timestamp, open, high, low, close, volume

    Output:
        OHLC + price structure + RSI + MA features
    """

    def __init__(
        self,
        rsi_period: int = 14,
        ma_period: int = 20,
    ):
        self.rsi_period = rsi_period
        self.ma_period = ma_period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(data)

        result = data.copy()
        result = result.sort_values("timestamp")
        result = result.reset_index(drop=True)

        result = self._price_structure(result)
        result = self._rsi(result)
        result = self._moving_average(result)

        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
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
                f"Feature Engine missing columns: {sorted(missing)}"
            )

        if data.empty:
            raise ValueError(
                "Feature Engine received empty market data."
            )

    def _price_structure(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

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

        previous_swing_high = None
        previous_swing_low = None

        structure = []

        for index, row in data.iterrows():

            value = None

            if row["swing_high"]:
                current = row["high"]

                if previous_swing_high is None:
                    value = "INITIAL_H"
                elif current > previous_swing_high:
                    value = "HH"
                else:
                    value = "LH"

                previous_swing_high = current

            elif row["swing_low"]:
                current = row["low"]

                if previous_swing_low is None:
                    value = "INITIAL_L"
                elif current > previous_swing_low:
                    value = "HL"
                else:
                    value = "LL"

                previous_swing_low = current

            structure.append(value)

        data["structure_event"] = structure

        data["structure"] = (
            data["structure_event"]
            .ffill()
        )

        return data

    def _rsi(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        delta = data["close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        average_gain = gain.ewm(
            alpha=1 / self.rsi_period,
            min_periods=self.rsi_period,
            adjust=False,
        ).mean()

        average_loss = loss.ewm(
            alpha=1 / self.rsi_period,
            min_periods=self.rsi_period,
            adjust=False,
        ).mean()

        rs = average_gain / average_loss

        data["rsi"] = 100 - (
            100 / (1 + rs)
        )

        data.loc[
            (average_loss == 0)
            & (average_gain > 0),
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

        return data

    def _moving_average(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        data["ma"] = (
            data["close"]
            .rolling(self.ma_period)
            .mean()
        )

        data["ma_direction"] = "FLAT"

        data.loc[
            data["ma"].diff() > 0,
            "ma_direction",
        ] = "RISING"

        data.loc[
            data["ma"].diff() < 0,
            "ma_direction",
        ] = "FALLING"

        data["price_vs_ma"] = "UNKNOWN"

        data.loc[
            data["close"] > data["ma"],
            "price_vs_ma",
        ] = "ABOVE"

        data.loc[
            data["close"] < data["ma"],
            "price_vs_ma",
        ] = "BELOW"

        return data