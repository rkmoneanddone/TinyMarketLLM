from __future__ import annotations

import pandas as pd

from src.tiny_market_llm.backtest.backtest_engine import (
    Prediction,
)


class BaselinePredictor:
    """
    Temporary predictor used only to test
    the Backtest Engine.

    This is NOT the TinyMarketLLM.
    """

    def predict(
        self,
        row: pd.Series,
    ) -> Prediction:

        structure = row.get(
            "structure",
            None,
        )

        rsi_direction = row.get(
            "rsi_direction",
            "FLAT",
        )

        ma_direction = row.get(
            "ma_direction",
            "FLAT",
        )

        price_vs_ma = row.get(
            "price_vs_ma",
            "UNKNOWN",
        )

        bullish = (
            structure in {"HH", "HL"}
            and rsi_direction == "RISING"
            and ma_direction == "RISING"
            and price_vs_ma == "ABOVE"
        )

        bearish = (
            structure in {"LH", "LL"}
            and rsi_direction == "FALLING"
            and ma_direction == "FALLING"
            and price_vs_ma == "BELOW"
        )

        if bullish:
            return Prediction(
                direction="UP",
                expected_move_pct=1.0,
                expected_duration=5,
                confidence=0.60,
            )

        if bearish:
            return Prediction(
                direction="DOWN",
                expected_move_pct=1.0,
                expected_duration=5,
                confidence=0.60,
            )

        return Prediction(
            direction="FLAT",
            expected_move_pct=0.0,
            expected_duration=0,
            confidence=0.40,
        )