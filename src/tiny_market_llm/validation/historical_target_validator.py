from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd


@dataclass
class HistoricalValidationResult:
    direction: str
    structure: str
    timestamp: object

    current_price: float

    ma_21: float
    ma_50: float
    ma_200: float
    ma_aligned: bool

    previous_level: Optional[float]
    previous_level_hit: bool
    previous_level_candles: Optional[int]

    fib_1618: Optional[float]
    fib_1618_hit: bool
    fib_1618_candles: Optional[int]


class HistoricalTargetValidator:
    """
    Tests higher-timeframe target hypotheses against historical OHLC.

    Setup information comes only from candle T and earlier.

    Outcome information comes only from candles T+1 ... T+lookahead.

    Supported MA alignment:

        Bullish:
            MA21 > MA50 > MA200

        Bearish:
            MA21 < MA50 < MA200

    Supported structure:

        Bullish:
            HH / HL

        Bearish:
            LH / LL

    Targets:

        Bullish:
            previous swing high
            Fibonacci 1.618 extension

        Bearish:
            previous swing low
            Fibonacci 1.618 extension
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
        lookahead: int = 10,
        pivot_bars: int = 2,
    ):
        if lookahead <= 0:
            raise ValueError(
                "lookahead must be greater than zero."
            )

        if pivot_bars <= 0:
            raise ValueError(
                "pivot_bars must be greater than zero."
            )

        self.lookahead = lookahead
        self.pivot_bars = pivot_bars

    def evaluate(
        self,
        ohlc: pd.DataFrame,
        structures: Optional[pd.DataFrame] = None,
    ) -> dict:

        data = self._prepare_ohlc(ohlc)

        data["ma_21"] = (
            data["close"]
            .rolling(21)
            .mean()
        )

        data["ma_50"] = (
            data["close"]
            .rolling(50)
            .mean()
        )

        data["ma_200"] = (
            data["close"]
            .rolling(200)
            .mean()
        )

        data["pivot_high"] = self._pivot_highs(
            data["high"]
        )

        data["pivot_low"] = self._pivot_lows(
            data["low"]
        )

        data["previous_pivot_high"] = (
            data["pivot_high"]
            .ffill()
            .shift(1)
        )

        data["previous_pivot_low"] = (
            data["pivot_low"]
            .ffill()
            .shift(1)
        )

        data["last_swing_high"] = (
            data["pivot_high"]
            .ffill()
            .shift(1)
        )

        data["last_swing_low"] = (
            data["pivot_low"]
            .ffill()
            .shift(1)
        )

        if structures is not None:
            data = self._attach_structures(
                data,
                structures,
            )
        else:
            data["structure"] = None

        results = []

        for index in range(len(data)):

            row = data.iloc[index]

            if pd.isna(row["ma_200"]):
                continue

            structure = row["structure"]

            if not isinstance(structure, str):
                continue

            structure = structure.upper()

            if structure not in {
                "HH",
                "HL",
                "LH",
                "LL",
            }:
                continue

            if structure in {"HH", "HL"}:

                if not (
                    row["ma_21"]
                    > row["ma_50"]
                    > row["ma_200"]
                ):
                    continue

                result = self._evaluate_bullish(
                    data,
                    index,
                )

            else:

                if not (
                    row["ma_21"]
                    < row["ma_50"]
                    < row["ma_200"]
                ):
                    continue

                result = self._evaluate_bearish(
                    data,
                    index,
                )

            if result is not None:
                results.append(result)

        return self._summarize(
            data,
            results,
        )

    # --------------------------------------------------
    # BULLISH
    # --------------------------------------------------

    def _evaluate_bullish(
        self,
        data: pd.DataFrame,
        index: int,
    ) -> Optional[HistoricalValidationResult]:

        row = data.iloc[index]

        previous_high = row[
            "previous_pivot_high"
        ]

        swing_high = row[
            "last_swing_high"
        ]

        swing_low = row[
            "last_swing_low"
        ]

        if pd.isna(previous_high):
            return None

        if pd.isna(swing_high) or pd.isna(swing_low):
            return None

        current_price = float(
            row["close"]
        )

        previous_high = float(
            previous_high
        )

        swing_high = float(
            swing_high
        )

        swing_low = float(
            swing_low
        )

        if previous_high <= current_price:
            return None

        fib_1618 = (
            swing_low
            + 1.618
            * (
                swing_high
                - swing_low
            )
        )

        previous_hit, previous_candles = (
            self._future_upside_hit(
                data,
                index,
                previous_high,
            )
        )

        fib_hit, fib_candles = (
            self._future_upside_hit(
                data,
                index,
                fib_1618,
            )
        )

        return HistoricalValidationResult(
            direction="UP",
            structure=str(row["structure"]),
            timestamp=row["timestamp"],
            current_price=current_price,
            ma_21=float(row["ma_21"]),
            ma_50=float(row["ma_50"]),
            ma_200=float(row["ma_200"]),
            ma_aligned=True,
            previous_level=previous_high,
            previous_level_hit=previous_hit,
            previous_level_candles=previous_candles,
            fib_1618=fib_1618,
            fib_1618_hit=fib_hit,
            fib_1618_candles=fib_candles,
        )

    # --------------------------------------------------
    # BEARISH
    # --------------------------------------------------

    def _evaluate_bearish(
        self,
        data: pd.DataFrame,
        index: int,
    ) -> Optional[HistoricalValidationResult]:

        row = data.iloc[index]

        previous_low = row[
            "previous_pivot_low"
        ]

        swing_high = row[
            "last_swing_high"
        ]

        swing_low = row[
            "last_swing_low"
        ]

        if pd.isna(previous_low):
            return None

        if pd.isna(swing_high) or pd.isna(swing_low):
            return None

        current_price = float(
            row["close"]
        )

        previous_low = float(
            previous_low
        )

        swing_high = float(
            swing_high
        )

        swing_low = float(
            swing_low
        )

        if previous_low >= current_price:
            return None

        fib_1618 = (
            swing_high
            - 1.618
            * (
                swing_high
                - swing_low
            )
        )

        previous_hit, previous_candles = (
            self._future_downside_hit(
                data,
                index,
                previous_low,
            )
        )

        fib_hit, fib_candles = (
            self._future_downside_hit(
                data,
                index,
                fib_1618,
            )
        )

        return HistoricalValidationResult(
            direction="DOWN",
            structure=str(row["structure"]),
            timestamp=row["timestamp"],
            current_price=current_price,
            ma_21=float(row["ma_21"]),
            ma_50=float(row["ma_50"]),
            ma_200=float(row["ma_200"]),
            ma_aligned=True,
            previous_level=previous_low,
            previous_level_hit=previous_hit,
            previous_level_candles=previous_candles,
            fib_1618=fib_1618,
            fib_1618_hit=fib_hit,
            fib_1618_candles=fib_candles,
        )

    # --------------------------------------------------
    # FUTURE OUTCOME
    # --------------------------------------------------

    def _future_upside_hit(
        self,
        data: pd.DataFrame,
        index: int,
        target: float,
    ) -> tuple[bool, Optional[int]]:

        end = min(
            len(data),
            index + 1 + self.lookahead,
        )

        future = data.iloc[
            index + 1:end
        ]

        for offset, high in enumerate(
            future["high"],
            start=1,
        ):
            if high >= target:
                return True, offset

        return False, None

    def _future_downside_hit(
        self,
        data: pd.DataFrame,
        index: int,
        target: float,
    ) -> tuple[bool, Optional[int]]:

        end = min(
            len(data),
            index + 1 + self.lookahead,
        )

        future = data.iloc[
            index + 1:end
        ]

        for offset, low in enumerate(
            future["low"],
            start=1,
        ):
            if low <= target:
                return True, offset

        return False, None

    # --------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------

    def _attach_structures(
        self,
        data: pd.DataFrame,
        structures: pd.DataFrame,
    ) -> pd.DataFrame:

        if "timestamp" not in structures.columns:
            raise ValueError(
                "Structure data must contain timestamp."
            )

        if "structure" not in structures.columns:
            raise ValueError(
                "Structure data must contain structure."
            )

        structure_data = structures[
            [
                "timestamp",
                "structure",
            ]
        ].copy()

        structure_data["timestamp"] = (
            pd.to_datetime(
                structure_data["timestamp"],
                utc=True,
            )
        )

        return data.merge(
            structure_data,
            on="timestamp",
            how="left",
        )

    # --------------------------------------------------
    # PIVOTS
    # --------------------------------------------------

    def _pivot_highs(
        self,
        series: pd.Series,
    ) -> pd.Series:

        n = self.pivot_bars

        result = pd.Series(
            float("nan"),
            index=series.index,
        )

        for i in range(
            n,
            len(series) - n,
        ):

            value = series.iloc[i]

            left = series.iloc[
                i - n:i
            ]

            right = series.iloc[
                i + 1:i + n + 1
            ]

            if (
                value > left.max()
                and value >= right.max()
            ):
                result.iloc[i] = value

        return result

    def _pivot_lows(
        self,
        series: pd.Series,
    ) -> pd.Series:

        n = self.pivot_bars

        result = pd.Series(
            float("nan"),
            index=series.index,
        )

        for i in range(
            n,
            len(series) - n,
        ):

            value = series.iloc[i]

            left = series.iloc[
                i - n:i
            ]

            right = series.iloc[
                i + 1:i + n + 1
            ]

            if (
                value < left.min()
                and value <= right.min()
            ):
                result.iloc[i] = value

        return result

    # --------------------------------------------------
    # PREPARE
    # --------------------------------------------------

    def _prepare_ohlc(
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
                f"Missing OHLC columns: {missing}"
            )

        result = data.copy()

        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            utc=True,
        )

        numeric = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for column in numeric:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

        result = (
            result
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

        return result

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    def _summarize(
        self,
        data: pd.DataFrame,
        results: list[HistoricalValidationResult],
    ) -> dict:

        records = [
            asdict(result)
            for result in results
        ]

        frame = pd.DataFrame(records)

        if frame.empty:
            return {
                "candles": len(data),
                "samples": 0,
                "bullish_samples": 0,
                "bearish_samples": 0,
                "previous_level_hit_rate": 0.0,
                "fib_1618_hit_rate": 0.0,
                "results": [],
            }

        bullish = frame[
            frame["direction"] == "UP"
        ]

        bearish = frame[
            frame["direction"] == "DOWN"
        ]

        return {
            "candles": len(data),
            "samples": len(frame),

            "bullish_samples": len(
                bullish
            ),

            "bearish_samples": len(
                bearish
            ),

            "bullish_previous_high_hit_rate": (
                self._rate(
                    bullish,
                    "previous_level_hit",
                )
            ),

            "bullish_fib_1618_hit_rate": (
                self._rate(
                    bullish,
                    "fib_1618_hit",
                )
            ),

            "bearish_previous_low_hit_rate": (
                self._rate(
                    bearish,
                    "previous_level_hit",
                )
            ),

            "bearish_fib_1618_hit_rate": (
                self._rate(
                    bearish,
                    "fib_1618_hit",
                )
            ),

            "previous_level_hit_rate": (
                self._rate(
                    frame,
                    "previous_level_hit",
                )
            ),

            "fib_1618_hit_rate": (
                self._rate(
                    frame,
                    "fib_1618_hit",
                )
            ),

            "results": records,
        }

    @staticmethod
    def _rate(
        frame: pd.DataFrame,
        column: str,
    ) -> float:

        if frame.empty:
            return 0.0

        return float(
            frame[column].mean()
            * 100.0
        )
