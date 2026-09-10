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
        self.assertIn("target_before_stop_rate", summary)
        self.assertIn("accuracy_ci_95_low", summary)
        self.assertIn("target_rate_ci_95_low", summary)
        self.assertIn("symbol_metrics", summary)
        self.assertTrue(set(result["trade_outcome"]).issubset(
            {"TARGET", "STOP", "AMBIGUOUS", "NEITHER", "NO_TRADE"}
        ))

    def test_multi_horizon_and_evidence_are_reported(self):
        result = self.scanner.scan_latest(self.frames)
        self.assertTrue({"horizon_agreement", "horizon_votes", "evidence"}.issubset(result.columns))
        traded = result[result["decision"].isin(["BUY", "SELL"])]
        if not traded.empty:
            expected = traded["decision"].map({"BUY": "UP", "SELL": "DOWN"})
            self.assertTrue((traded["evidence"] == expected).all())
            self.assertTrue((traded["horizon_agreement"] >= 2).all())

    def test_training_labels_do_not_cross_cutoff(self):
        prepared = self.scanner._combine(self.frames)
        cutoff = pd.Timestamp("2024-09-30", tz="UTC")
        train = prepared[(prepared["timestamp"] <= cutoff) & prepared["actual"].notna()]
        for horizon in self.scanner.config.horizons:
            crossing = train[train[f"label_timestamp_{horizon}"] > cutoff]
            eligible = self.scanner._training_subset(train, horizon, cutoff)
            self.assertGreater(len(crossing), 0)
            self.assertTrue((eligible[f"label_timestamp_{horizon}"] <= cutoff).all())

    def test_missing_ohlc_is_rejected(self):
        with self.assertRaises(ValueError):
            self.scanner.prepare(pd.DataFrame({"close": [1.0]}))

    def test_small_trade_sample_cannot_be_promoted(self):
        low, high = self.scanner._wilson_interval(4, 5)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)


if __name__ == "__main__":
    unittest.main()
