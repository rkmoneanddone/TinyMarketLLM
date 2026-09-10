from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "return_1", "return_3", "return_5", "body_pct", "upper_wick_pct",
    "lower_wick_pct", "range_pct", "ema_21_distance", "ema_50_distance",
    "ema_21_slope", "ema_50_slope", "rsi_14", "rsi_change_3",
    "atr_14_pct", "volume_ratio_20", "distance_high_20",
    "distance_low_20", "breakout_high_20", "breakdown_low_20",
]


@dataclass(frozen=True)
class ScannerConfig:
    horizon: int = 1
    flat_threshold_pct: float = 0.5
    minimum_confidence: float = 0.58
    minimum_training_rows: int = 120


class TinyMarketScanner:
    """Transparent UP/DOWN/FLAT classifier plus deterministic trade gate.

    Every feature uses the current or an earlier candle. Labels use only future
    closes and are removed from the row used for a live prediction.
    """

    def __init__(self, config: ScannerConfig | None = None):
        self.config = config or ScannerConfig()

    def prepare(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        self._validate_ohlc(ohlc)
        d = ohlc.copy().sort_values("timestamp").drop_duplicates("timestamp")
        d = d.reset_index(drop=True)
        for column in ("open", "high", "low", "close", "volume"):
            d[column] = pd.to_numeric(d[column], errors="coerce")

        close = d["close"]
        high = d["high"]
        low = d["low"]
        open_ = d["open"]
        safe_close = close.replace(0, np.nan)
        safe_open = open_.replace(0, np.nan)

        d["return_1"] = close.pct_change(fill_method=None) * 100
        d["return_3"] = close.pct_change(3, fill_method=None) * 100
        d["return_5"] = close.pct_change(5, fill_method=None) * 100
        d["body_pct"] = (close - open_) / safe_open * 100
        d["upper_wick_pct"] = (high - np.maximum(open_, close)) / safe_close * 100
        d["lower_wick_pct"] = (np.minimum(open_, close) - low) / safe_close * 100
        d["range_pct"] = (high - low) / safe_close * 100

        ema21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
        ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
        d["ema_21_distance"] = (close - ema21) / ema21 * 100
        d["ema_50_distance"] = (close - ema50) / ema50 * 100
        d["ema_21_slope"] = ema21.pct_change(3, fill_method=None) * 100
        d["ema_50_slope"] = ema50.pct_change(5, fill_method=None) * 100

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        d["rsi_14"] = (100 - 100 / (1 + rs)).fillna(
            pd.Series(np.where(avg_gain > 0, 100.0, np.nan), index=d.index)
        )
        d["rsi_change_3"] = d["rsi_14"].diff(3)

        previous_close = close.shift(1)
        true_range = pd.concat(
            [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        d["atr_14_pct"] = true_range.rolling(14).mean() / safe_close * 100
        d["volume_ratio_20"] = d["volume"] / d["volume"].rolling(20).mean()

        prior_high = high.shift(1).rolling(20).max()
        prior_low = low.shift(1).rolling(20).min()
        d["distance_high_20"] = (close - prior_high) / prior_high * 100
        d["distance_low_20"] = (close - prior_low) / prior_low * 100
        d["breakout_high_20"] = (close > prior_high).astype(float)
        d["breakdown_low_20"] = (close < prior_low).astype(float)

        future_close = close.shift(-self.config.horizon)
        d["future_move_pct"] = (future_close - close) / safe_close * 100
        d["actual"] = "FLAT"
        d.loc[d["future_move_pct"] > self.config.flat_threshold_pct, "actual"] = "UP"
        d.loc[d["future_move_pct"] < -self.config.flat_threshold_pct, "actual"] = "DOWN"
        d.loc[future_close.isna(), "actual"] = None
        return d

    def historical_test(
        self,
        frames: dict[str, pd.DataFrame],
        train_end: str | pd.Timestamp,
        train_start: str | pd.Timestamp | None = None,
        test_start: str | pd.Timestamp | None = None,
        test_end: str | pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        prepared = self._combine(frames)
        cutoff = self._coerce_cutoff(train_end, prepared["timestamp"])
        train = prepared[(prepared["timestamp"] <= cutoff) & prepared["actual"].notna()]
        if train_start is not None:
            beginning = self._coerce_cutoff(train_start, prepared["timestamp"])
            train = train[train["timestamp"] >= beginning]

        test_boundary = cutoff
        if test_start is not None:
            test_boundary = self._coerce_cutoff(test_start, prepared["timestamp"])
            test = prepared[(prepared["timestamp"] >= test_boundary) & prepared["actual"].notna()]
        else:
            test = prepared[(prepared["timestamp"] > test_boundary) & prepared["actual"].notna()]
        if test_end is not None:
            end = self._coerce_cutoff(test_end, prepared["timestamp"])
            test = test[test["timestamp"] <= end]
        self._ensure_trainable(train)
        if test.empty:
            raise ValueError("No unseen rows exist after the training cutoff.")

        model = self._new_model().fit(train[FEATURE_COLUMNS], train["actual"])
        result = self._predictions(model, test)
        accuracy = float((result["prediction"] == result["actual"]).mean())
        balanced_accuracy = float(balanced_accuracy_score(result["actual"], result["prediction"]))
        majority_baseline = float(result["actual"].value_counts(normalize=True).max())
        directional = result[result["decision"].isin(["BUY", "SELL"])]
        directional_accuracy = (
            float((directional["prediction"] == directional["actual"]).mean())
            if not directional.empty else None
        )
        summary = {
            "train_start": str(train["timestamp"].min()),
            "train_end": str(cutoff),
            "test_start": str(test["timestamp"].min()),
            "test_end": str(test["timestamp"].max()),
            "training_rows": int(len(train)),
            "unseen_rows": int(len(result)),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "majority_baseline_accuracy": majority_baseline,
            "beats_majority_baseline": accuracy > majority_baseline,
            "model_status": "CANDIDATE" if accuracy > majority_baseline else "REJECTED",
            "directional_rows": int(len(directional)),
            "directional_accuracy": directional_accuracy,
        }
        return result, summary

    def scan_latest(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        prepared = self._combine(frames)
        labelled = prepared[prepared["actual"].notna()]
        self._ensure_trainable(labelled)
        model = self._new_model().fit(labelled[FEATURE_COLUMNS], labelled["actual"])
        latest = prepared.sort_values("timestamp").groupby("symbol", as_index=False).tail(1)
        return self._predictions(model, latest, include_actual=False).sort_values(
            ["score", "confidence"], ascending=False
        ).reset_index(drop=True)

    def _predictions(self, model: Pipeline, rows: pd.DataFrame, include_actual: bool = True) -> pd.DataFrame:
        probabilities = model.predict_proba(rows[FEATURE_COLUMNS])
        classes = list(model.classes_)
        out = rows[["timestamp", "symbol", "close"]].copy()
        for label in ("UP", "DOWN", "FLAT"):
            out[f"probability_{label.lower()}"] = (
                probabilities[:, classes.index(label)] if label in classes else 0.0
            )
        predicted_index = probabilities.argmax(axis=1)
        out["prediction"] = [classes[index] for index in predicted_index]
        out["confidence"] = probabilities.max(axis=1)
        out["decision"] = "WAIT"
        qualified = out["confidence"] >= self.config.minimum_confidence
        out.loc[qualified & (out["prediction"] == "UP"), "decision"] = "BUY"
        out.loc[qualified & (out["prediction"] == "DOWN"), "decision"] = "SELL"
        out["score"] = (out["confidence"] * 100).round(1)
        out["reason"] = np.where(
            out["decision"] == "WAIT",
            "Confidence below threshold or FLAT is most probable",
            "Direction passed the configured probability threshold",
        )
        if include_actual:
            out["actual"] = rows["actual"].values
            out["future_move_pct"] = rows["future_move_pct"].values
            out["correct"] = out["prediction"] == out["actual"]
        return out

    def _combine(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        items = []
        for symbol, frame in frames.items():
            prepared = self.prepare(frame)
            prepared["symbol"] = symbol.upper()
            items.append(prepared)
        if not items:
            raise ValueError("At least one stock dataset is required.")
        return pd.concat(items, ignore_index=True).dropna(subset=FEATURE_COLUMNS)

    def _ensure_trainable(self, train: pd.DataFrame) -> None:
        if len(train) < self.config.minimum_training_rows:
            raise ValueError(
                f"Need at least {self.config.minimum_training_rows} training rows; got {len(train)}."
            )
        if train["actual"].nunique() < 2:
            raise ValueError("Training data needs at least two outcome classes.")

    @staticmethod
    def _new_model() -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ])

    @staticmethod
    def _coerce_cutoff(value: str | pd.Timestamp, series: pd.Series) -> pd.Timestamp:
        cutoff = pd.Timestamp(value)
        timezone = getattr(series.dt, "tz", None)
        if timezone is not None and cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(timezone)
        elif timezone is None and cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)
        return cutoff

    @staticmethod
    def _validate_ohlc(data: pd.DataFrame) -> None:
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
        if data.empty:
            raise ValueError("OHLC dataset is empty.")
