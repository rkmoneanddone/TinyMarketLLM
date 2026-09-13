from __future__ import annotations

import unittest
import pandas as pd

from src.tiny_market_llm.scanner.order_block_experiment import VARIANTS, select_variant


class OrderBlockExperimentTests(unittest.TestCase):
    def test_variants_are_incremental_filters(self):
        row = {name: True for name in ("liquidity_sweep", "strong_displacement", "fvg_created", "rejection", "price_volume", "rsi_confirmation")}
        ledger = pd.DataFrame([row, {**row, "fvg_created": False}])
        self.assertEqual(len(select_variant(ledger, "A_ORDER_BLOCK_RETEST")), 2)
        self.assertEqual(len(select_variant(ledger, "D_PLUS_FVG")), 1)
        self.assertEqual(len(select_variant(ledger, "H_COMPLETE_RSI_VOLUME")), 1)

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            select_variant(pd.DataFrame(), "NOT_A_TEST")

    def test_all_eight_variants_are_named(self):
        self.assertEqual(len(VARIANTS), 8)


if __name__ == "__main__": unittest.main()
