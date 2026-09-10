from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tiny_market_llm.scanner import ScannerConfig, TinyMarketScanner


def sample(seed: int, rows: int = 500) -> pd.DataFrame:
    random = np.random.default_rng(seed)
    returns = random.normal(0.0004, 0.012, rows)
    close = 100 * np.cumprod(1 + returns)
    open_ = close * (1 + random.normal(0, 0.003, rows))
    high = np.maximum(open_, close) * (1 + random.uniform(0.001, 0.012, rows))
    low = np.minimum(open_, close) * (1 - random.uniform(0.001, 0.012, rows))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": random.integers(100_000, 2_000_000, rows),
    })


class ScannerTests(unittest.TestCase):
    def setUp(self):
        self.scanner = TinyMarketScanner(ScannerConfig(minimum_training_rows=100))
        self.frames = {"GRASIM": sample(1), "RELIANCE": sample(2), "TCS": sample(3)}

    def test_latest_scan_has_one_row_per_stock(self):
        result = self.scanner.scan_latest(self.frames)
        self.assertEqual(set(result["symbol"]), set(self.frames))
        self.assertTrue(set(result["decision"]).issubset({"BUY", "SELL", "WAIT"}))

    def test_fixed_cutoff_keeps_future_unseen(self):
        result, summary = self.scanner.historical_test(
            self.frames,
            train_start="2024-03-01",
            train_end="2024-09-30",
            test_start="2024-10-01",
            test_end="2025-03-31",
        )
        cutoff = pd.Timestamp(summary["train_end"])
        self.assertTrue((pd.to_datetime(result["timestamp"], utc=True) > cutoff).all())
        self.assertGreater(summary["unseen_rows"], 0)
        self.assertIn(summary["model_status"], {"CANDIDATE", "REJECTED"})

    def test_missing_ohlc_is_rejected(self):
        with self.assertRaises(ValueError):
            self.scanner.prepare(pd.DataFrame({"close": [1.0]}))


if __name__ == "__main__":
    unittest.main()
