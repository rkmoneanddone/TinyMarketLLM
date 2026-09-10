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
    horizon: int = 3
    horizons: tuple[int, ...] = (1, 3, 5)
    flat_threshold_pct: float = 0.5
    minimum_confidence: float = 0.58
    minimum_training_rows: int = 120
    minimum_horizon_agreement: int = 2
    target_atr_multiple: float = 1.0
    stop_atr_multiple: float = 0.75


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

        for horizon in self.config.horizons:
            horizon_close = close.shift(-horizon)
            d[f"label_timestamp_{horizon}"] = d["timestamp"].shift(-horizon)
            move = (horizon_close - close) / safe_close * 100
            label = pd.Series("FLAT", index=d.index, dtype="object")
            label.loc[move > self.config.flat_threshold_pct] = "UP"
            label.loc[move < -self.config.flat_threshold_pct] = "DOWN"
            label.loc[horizon_close.isna()] = None
            d[f"future_move_{horizon}_pct"] = move
            d[f"actual_{horizon}"] = label

        future_close = close.shift(-self.config.horizon)
        d["future_move_pct"] = (future_close - close) / safe_close * 100
        d["actual"] = d[f"actual_{self.config.horizon}"]
        self._add_trade_path_outcomes(d)
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

        models = self._fit_horizon_models(train, label_cutoff=cutoff)
        result = self._predictions(models, test)
        accuracy = float((result["prediction"] == result["actual"]).mean())
        balanced_accuracy = float(balanced_accuracy_score(result["actual"], result["prediction"]))
        majority_baseline = float(result["actual"].value_counts(normalize=True).max())
        directional = result[result["decision"].isin(["BUY", "SELL"])]
        directional_accuracy = (
            float((directional["prediction"] == directional["actual"]).mean())
            if not directional.empty else None
        )
        resolved_trades = directional[directional["trade_outcome"].isin(["TARGET", "STOP"])]
        target_before_stop_rate = (
            float((resolved_trades["trade_outcome"] == "TARGET").mean())
            if not resolved_trades.empty else None
        )
        predictive_edge = accuracy > majority_baseline and balanced_accuracy > (1 / 3)
        trade_edge = (
            len(resolved_trades) >= 5
            and target_before_stop_rate is not None
            and target_before_stop_rate > 0.5
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
            "predictive_edge": predictive_edge,
            "trade_edge": trade_edge,
            "model_status": "CANDIDATE" if predictive_edge and trade_edge else "REJECTED",
            "directional_rows": int(len(directional)),
            "directional_accuracy": directional_accuracy,
            "resolved_trades": int(len(resolved_trades)),
            "target_before_stop_rate": target_before_stop_rate,
            "ambiguous_trades": int((directional["trade_outcome"] == "AMBIGUOUS").sum()),
        }
        return result, summary

    def scan_latest(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        prepared = self._combine(frames)
        labelled = prepared[prepared["actual"].notna()]
        self._ensure_trainable(labelled)
        models = self._fit_horizon_models(labelled)
        latest = prepared.sort_values("timestamp").groupby("symbol", as_index=False).tail(1)
        return self._predictions(models, latest, include_actual=False).sort_values(
            ["score", "confidence"], ascending=False
        ).reset_index(drop=True)

    def _fit_horizon_models(
        self,
        train: pd.DataFrame,
        label_cutoff: pd.Timestamp | None = None,
    ) -> dict[int, Pipeline]:
        models = {}
        for horizon in self.config.horizons:
            label = f"actual_{horizon}"
            subset = self._training_subset(train, horizon, label_cutoff)
            self._ensure_trainable(subset, label)
            models[horizon] = self._new_model().fit(subset[FEATURE_COLUMNS], subset[label])
        return models

    @staticmethod
    def _training_subset(
        train: pd.DataFrame,
        horizon: int,
        label_cutoff: pd.Timestamp | None,
    ) -> pd.DataFrame:
        label = f"actual_{horizon}"
        subset = train[train[label].notna()]
        if label_cutoff is not None:
            subset = subset[subset[f"label_timestamp_{horizon}"] <= label_cutoff]
        return subset

    def _predictions(self, models: dict[int, Pipeline], rows: pd.DataFrame, include_actual: bool = True) -> pd.DataFrame:
        probability_sets = []
        votes = []
        labels = ("UP", "DOWN", "FLAT")
        for horizon, model in models.items():
            raw = model.predict_proba(rows[FEATURE_COLUMNS])
            classes = list(model.classes_)
            aligned = np.column_stack([
                raw[:, classes.index(label)] if label in classes else np.zeros(len(rows))
                for label in labels
            ])
            probability_sets.append(aligned)
            votes.append(np.array(labels)[aligned.argmax(axis=1)])
        probabilities = np.mean(probability_sets, axis=0)
        out = rows[["timestamp", "symbol", "close"]].copy()
        for index, label in enumerate(labels):
            out[f"probability_{label.lower()}"] = probabilities[:, index]
        predicted_index = probabilities.argmax(axis=1)
        out["prediction"] = [labels[index] for index in predicted_index]
        out["confidence"] = probabilities.max(axis=1)
        vote_matrix = np.column_stack(votes)
        out["horizon_agreement"] = [int((row == prediction).sum()) for row, prediction in zip(vote_matrix, out["prediction"])]
        out["horizon_votes"] = ["|".join(f"{h}:{vote}" for h, vote in zip(models, row)) for row in vote_matrix]
        evidence = rows.apply(self._evidence_direction, axis=1)
        out["evidence"] = evidence.values
        out["decision"] = "WAIT"
        qualified = (
            (out["confidence"] >= self.config.minimum_confidence)
            & (out["horizon_agreement"] >= self.config.minimum_horizon_agreement)
            & (out["prediction"] == out["evidence"])
        )
        out.loc[qualified & (out["prediction"] == "UP"), "decision"] = "BUY"
        out.loc[qualified & (out["prediction"] == "DOWN"), "decision"] = "SELL"
        out["score"] = (out["confidence"] * 100).round(1)
        out["reason"] = out.apply(self._decision_reason, axis=1)
        if include_actual:
            out["actual"] = rows["actual"].values
            out["future_move_pct"] = rows["future_move_pct"].values
            out["correct"] = out["prediction"] == out["actual"]
            outcomes = []
            for decision, buy_outcome, sell_outcome in zip(
                out["decision"], rows["buy_trade_outcome"], rows["sell_trade_outcome"]
            ):
                outcomes.append(buy_outcome if decision == "BUY" else sell_outcome if decision == "SELL" else "NO_TRADE")
            out["trade_outcome"] = outcomes
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

    def _ensure_trainable(self, train: pd.DataFrame, label: str = "actual") -> None:
        if len(train) < self.config.minimum_training_rows:
            raise ValueError(
                f"Need at least {self.config.minimum_training_rows} training rows; got {len(train)}."
            )
        if train[label].nunique() < 2:
            raise ValueError("Training data needs at least two outcome classes.")

    def _add_trade_path_outcomes(self, data: pd.DataFrame) -> None:
        horizon = max(self.config.horizons)
        atr = data["atr_14_pct"] / 100 * data["close"]
        buy_target = data["close"] + self.config.target_atr_multiple * atr
        buy_stop = data["close"] - self.config.stop_atr_multiple * atr
        sell_target = data["close"] - self.config.target_atr_multiple * atr
        sell_stop = data["close"] + self.config.stop_atr_multiple * atr

        def first_step(condition_by_step: list[pd.Series]) -> pd.Series:
            result = pd.Series(np.nan, index=data.index)
            for step, condition in enumerate(condition_by_step, start=1):
                result = result.mask(result.isna() & condition.fillna(False), step)
            return result

        future_highs = [data["high"].shift(-step) for step in range(1, horizon + 1)]
        future_lows = [data["low"].shift(-step) for step in range(1, horizon + 1)]
        buy_target_step = first_step([value >= buy_target for value in future_highs])
        buy_stop_step = first_step([value <= buy_stop for value in future_lows])
        sell_target_step = first_step([value <= sell_target for value in future_lows])
        sell_stop_step = first_step([value >= sell_stop for value in future_highs])
        data["buy_trade_outcome"] = self._resolve_path(buy_target_step, buy_stop_step)
        data["sell_trade_outcome"] = self._resolve_path(sell_target_step, sell_stop_step)

    @staticmethod
    def _resolve_path(target_step: pd.Series, stop_step: pd.Series) -> pd.Series:
        outcome = pd.Series("NEITHER", index=target_step.index, dtype="object")
        outcome.loc[target_step.notna() & stop_step.isna()] = "TARGET"
        outcome.loc[target_step.isna() & stop_step.notna()] = "STOP"
        outcome.loc[target_step < stop_step] = "TARGET"
        outcome.loc[stop_step < target_step] = "STOP"
        outcome.loc[target_step.notna() & (target_step == stop_step)] = "AMBIGUOUS"
        return outcome

    @staticmethod
    def _evidence_direction(row: pd.Series) -> str:
        bullish = sum([
            row["ema_21_distance"] > 0,
            row["ema_21_slope"] > 0,
            row["rsi_14"] > 50 and row["rsi_change_3"] > 0,
            row["volume_ratio_20"] >= 1,
            row["breakout_high_20"] == 1,
        ])
        bearish = sum([
            row["ema_21_distance"] < 0,
            row["ema_21_slope"] < 0,
            row["rsi_14"] < 50 and row["rsi_change_3"] < 0,
            row["volume_ratio_20"] >= 1,
            row["breakdown_low_20"] == 1,
        ])
        return "UP" if bullish - bearish >= 2 else "DOWN" if bearish - bullish >= 2 else "FLAT"

    def _decision_reason(self, row: pd.Series) -> str:
        if row["prediction"] == "FLAT":
            return "FLAT is most probable"
        if row["confidence"] < self.config.minimum_confidence:
            return "Ensemble confidence below threshold"
        if row["horizon_agreement"] < self.config.minimum_horizon_agreement:
            return "Insufficient agreement across prediction horizons"
        if row["prediction"] != row["evidence"]:
            return "Model direction is not confirmed by technical evidence"
        return "Probability, horizon agreement, and technical evidence passed"

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
