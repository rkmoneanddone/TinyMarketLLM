from __future__ import annotations

import unittest
import pandas as pd

from src.tiny_market_llm.scanner.high_value_order_block import (
    higher_timeframe_bullish_order_blocks,
    lower_timeframe_retest_trades,
)


class HighValueOrderBlockTests(unittest.TestCase):
    def fixture(self):
        timestamp = pd.date_range("2025-01-01", periods=13, freq="D", tz="UTC")
        higher = pd.DataFrame({
            "timestamp": timestamp,
            "open":  [8, 9, 10, 11, 10, 9, 10, 11, 13, 14, 13, 12, 12],
            "high":  [9,10, 12, 11, 10,10, 11, 13, 15, 14, 13, 13, 13],
            "low":   [7, 8,  9, 10,  9, 8,  9, 10, 12, 12, 11, 11, 11],
            "close": [8, 9, 11, 10,  9,10, 11, 13, 14, 13, 12, 12, 12],
            "volume": 100,
        })
        return higher

    def test_structure_is_not_tradable_until_b_is_confirmed(self):
        structures = higher_timeframe_bullish_order_blocks(self.fixture())
        self.assertEqual(len(structures), 1)
        self.assertEqual(structures.iloc[0]["a_price"], 12)
        self.assertEqual(structures.iloc[0]["b_price"], 15)
        self.assertEqual(structures.iloc[0]["tradable_from"], pd.Timestamp("2025-01-11", tz="UTC"))

    def test_lower_timeframe_return_trades_c_to_b(self):
        higher = self.fixture()
        lower = higher.copy()
        lower.loc[11, ["open", "high", "low", "close"]] = [13, 14, 10, 12]
        lower.loc[12, ["open", "high", "low", "close"]] = [12, 15, 11, 15]
        trades = lower_timeframe_retest_trades(higher, lower, "TEST", "1D", "1H")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["entry_price"], 10)
        self.assertEqual(trades.iloc[0]["target_price"], 15)
        self.assertEqual(trades.iloc[0]["outcome"], "TARGET")


if __name__ == "__main__": unittest.main()
