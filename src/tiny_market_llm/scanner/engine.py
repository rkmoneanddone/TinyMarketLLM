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


CORE_FEATURE_COLUMNS = [
    "return_1", "return_3", "return_5", "body_pct", "upper_wick_pct",
    "lower_wick_pct", "range_pct", "ema_21_distance", "ema_50_distance",
    "ema_21_slope", "ema_50_slope", "rsi_14", "rsi_change_3",
    "atr_14_pct", "volume_ratio_20", "distance_high_20",
    "distance_low_20", "breakout_high_20", "breakdown_low_20",
]

# Deliberately small model input: raw candle behaviour plus three technical
# families (EMA trend, RSI momentum, and relative volume). ATR remains part of
# trade sizing/outcome evaluation, but is not an additional prediction vote.
COMPACT_FEATURE_COLUMNS = [
    "return_1", "return_3", "return_5", "body_pct", "close_location",
    "ema_21_distance", "ema_50_distance", "ema_21_slope",
    "rsi_14", "rsi_change_3", "volume_ratio_20",
]

CHART_FEATURE_COLUMNS = CORE_FEATURE_COLUMNS + [
    "distance_high_60", "distance_low_60", "breakout_high_60",
    "breakdown_low_60", "range_compression_10_50",
    "volume_expansion_5_20", "close_location",
    "breakout_strength_20_atr", "ema_21_distance_atr",
    "higher_low_strength_5_atr",
]


@dataclass(frozen=True)
class ScannerConfig:
    horizon: int = 1
    horizons: tuple[int, ...] = (1, 3, 5)
    trade_evaluation_horizon: int = 5
    flat_threshold_pct: float = 0.5
    minimum_confidence: float = 0.58
    minimum_training_rows: int = 120
    minimum_horizon_agreement: int = 2
    target_atr_multiple: float = 1.0
    stop_atr_multiple: float = 0.75
    minimum_resolved_trades: int = 20
    minimum_trade_symbols: int = 2
    feature_set: str = "core"
    eligible_setups: tuple[str, ...] = ("BREAKOUT_VOLUME", "BREAKOUT_RETEST")


class TinyMarketScanner:
    """Transparent UP/DOWN/FLAT classifier plus deterministic trade gate.

    Every feature uses the current or an earlier candle. Labels use only future
    closes and are removed from the row used for a live prediction.
    """

    def __init__(self, config: ScannerConfig | None = None):
        self.config = config or ScannerConfig()
        feature_sets = {
            "compact": COMPACT_FEATURE_COLUMNS,
            "core": CORE_FEATURE_COLUMNS,
            "chart": CHART_FEATURE_COLUMNS,
        }
        if self.config.feature_set not in feature_sets:
            raise ValueError(f"Unknown feature_set: {self.config.feature_set}")
        self.feature_columns = feature_sets[self.config.feature_set]

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

        prior_high_60 = high.shift(1).rolling(60).max()
        prior_low_60 = low.shift(1).rolling(60).min()
        atr_value = d["atr_14_pct"] / 100 * safe_close
        d["distance_high_60"] = (close - prior_high_60) / prior_high_60 * 100
        d["distance_low_60"] = (close - prior_low_60) / prior_low_60 * 100
        d["breakout_high_60"] = (close > prior_high_60).astype(float)
        d["breakdown_low_60"] = (close < prior_low_60).astype(float)
        d["range_compression_10_50"] = (
            d["range_pct"].rolling(10).mean()
            / d["range_pct"].rolling(50).mean().replace(0, np.nan)
        )
        d["volume_expansion_5_20"] = (
            d["volume"].rolling(5).mean()
            / d["volume"].rolling(20).mean().replace(0, np.nan)
        )
        candle_range = (high - low).replace(0, np.nan)
        d["close_location"] = (close - low) / candle_range
        d["breakout_strength_20_atr"] = (close - prior_high) / atr_value.replace(0, np.nan)
        d["ema_21_distance_atr"] = (close - ema21) / atr_value.replace(0, np.nan)
        recent_low = low.rolling(5).min()
        previous_low = low.shift(5).rolling(5).min()
        d["higher_low_strength_5_atr"] = (recent_low - previous_low) / atr_value.replace(0, np.nan)
        self._add_setup_signals(d, ema21, ema50, prior_high, prior_low, atr_value)
        d.replace([np.inf, -np.inf], np.nan, inplace=True)

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
        accuracy_ci_low, accuracy_ci_high = self._wilson_interval(
            int((result["prediction"] == result["actual"]).sum()), len(result)
        )
        target_rate_ci_low, target_rate_ci_high = self._wilson_interval(
            int((resolved_trades["trade_outcome"] == "TARGET").sum()),
            len(resolved_trades),
        )
        trade_symbols = int(directional["symbol"].nunique())
        predictive_edge = (
            accuracy_ci_low > majority_baseline
            and balanced_accuracy > (1 / 3)
        )
        trade_edge = (
            len(resolved_trades) >= self.config.minimum_resolved_trades
            and trade_symbols >= self.config.minimum_trade_symbols
            and target_before_stop_rate is not None
            and target_rate_ci_low is not None
            and target_rate_ci_low > 0.5
        )
        symbol_metrics = {}
        for symbol, group in result.groupby("symbol"):
            symbol_metrics[symbol] = {
                "rows": int(len(group)),
                "accuracy": float((group["prediction"] == group["actual"]).mean()),
                "majority_baseline_accuracy": float(group["actual"].value_counts(normalize=True).max()),
                "trades": int(group["decision"].isin(["BUY", "SELL"]).sum()),
            }
        summary = {
            "decision_horizon_candles": self.config.horizon,
            "trade_evaluation_horizon_candles": self.config.trade_evaluation_horizon,
            "train_start": str(train["timestamp"].min()),
            "train_end": str(cutoff),
            "test_start": str(test["timestamp"].min()),
            "test_end": str(test["timestamp"].max()),
            "training_rows": int(len(train)),
            "unseen_rows": int(len(result)),
            "accuracy": accuracy,
            "accuracy_ci_95_low": accuracy_ci_low,
            "accuracy_ci_95_high": accuracy_ci_high,
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
            "target_rate_ci_95_low": target_rate_ci_low,
            "target_rate_ci_95_high": target_rate_ci_high,
            "trade_symbols": trade_symbols,
            "ambiguous_trades": int((directional["trade_outcome"] == "AMBIGUOUS").sum()),
            "symbol_metrics": symbol_metrics,
            "horizon_metrics": self._horizon_metrics(result),
            "setup_metrics": self._setup_metrics(result),
        }
        return result, summary

    def daily_walk_forward_test(
        self,
        frames: dict[str, pd.DataFrame],
        test_start: str | pd.Timestamp,
        test_end: str | pd.Timestamp,
        training_years: int = 3,
    ) -> tuple[pd.DataFrame, dict]:
        """Retrain before every test day using only information then available.

        Features are prepared once for speed. Each fold purges training labels
        whose outcome timestamp reaches into the prediction day, preventing
        future leakage while closely matching a daily production run.
        """
        if training_years < 1:
            raise ValueError("training_years must be at least 1.")
        prepared = self._combine(frames)
        start = self._coerce_cutoff(test_start, prepared["timestamp"])
        end = self._coerce_cutoff(test_end, prepared["timestamp"])
        if start > end:
            raise ValueError("test_start must not be after test_end.")

        test_dates = (
            prepared.loc[
                prepared["timestamp"].between(start, end) & prepared["actual"].notna(),
                "timestamp",
            ]
            .drop_duplicates()
            .sort_values()
        )
        if test_dates.empty:
            raise ValueError("No labelled rows exist inside the walk-forward window.")

        predictions: list[pd.DataFrame] = []
        folds: list[dict] = []
        skipped: list[dict] = []
        for fold_number, prediction_time in enumerate(test_dates, start=1):
            label_cutoff = prediction_time - pd.Timedelta(nanoseconds=1)
            rolling_start = prediction_time - pd.DateOffset(years=training_years)
            train = prepared[
                (prepared["timestamp"] >= rolling_start)
                & (prepared["timestamp"] < prediction_time)
                & prepared["actual"].notna()
            ]
            test = prepared[prepared["timestamp"] == prediction_time]
            try:
                models = self._fit_horizon_models(train, label_cutoff=label_cutoff)
            except ValueError as error:
                skipped.append({"timestamp": str(prediction_time), "reason": str(error)})
                continue
            fold_rows = self._predictions(models, test)
            fold_rows.insert(0, "fold", fold_number)
            predictions.append(fold_rows)
            folds.append({
                "fold": fold_number,
                "prediction_time": str(prediction_time),
                "training_start": str(train["timestamp"].min()),
                "training_end": str(train["timestamp"].max()),
                "training_rows": int(len(train)),
                "test_rows": int(len(test)),
            })

        if not predictions:
            raise ValueError("No walk-forward fold had enough trainable history.")
        result = pd.concat(predictions, ignore_index=True)
        summary = self._quality_summary(result)
        summary.update({
            "mode": "daily_walk_forward",
            "decision_horizon_candles": self.config.horizon,
            "trade_evaluation_horizon_candles": self.config.trade_evaluation_horizon,
            "test_start": str(result["timestamp"].min()),
            "test_end": str(result["timestamp"].max()),
            "training_years": training_years,
            "fold_count": len(folds),
            "skipped_fold_count": len(skipped),
            "folds": folds,
            "skipped_folds": skipped,
        })
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
            models[horizon] = self._new_model().fit(subset[self.feature_columns], subset[label])
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
        probability_by_horizon = {}
        votes = []
        labels = ("UP", "DOWN", "FLAT")
        for horizon, model in models.items():
            raw = model.predict_proba(rows[self.feature_columns])
            classes = list(model.classes_)
            aligned = np.column_stack([
                raw[:, classes.index(label)] if label in classes else np.zeros(len(rows))
                for label in labels
            ])
            probability_by_horizon[horizon] = aligned
            votes.append(np.array(labels)[aligned.argmax(axis=1)])
        if self.config.horizon not in probability_by_horizon:
            raise ValueError("The configured decision horizon has no fitted model.")
        probabilities = probability_by_horizon[self.config.horizon]
        out = rows[["timestamp", "symbol", "close", "setup", "setup_direction"]].copy()
        for index, label in enumerate(labels):
            out[f"probability_{label.lower()}"] = probabilities[:, index]
        predicted_index = probabilities.argmax(axis=1)
        out["prediction"] = [labels[index] for index in predicted_index]
        out["confidence"] = probabilities.max(axis=1)
        for horizon, horizon_probabilities in probability_by_horizon.items():
            horizon_prediction = np.array(labels)[horizon_probabilities.argmax(axis=1)]
            out[f"prediction_{horizon}"] = horizon_prediction
            out[f"confidence_{horizon}"] = horizon_probabilities.max(axis=1)
        vote_matrix = np.column_stack(votes)
        out["horizon_agreement"] = [int((row == prediction).sum()) for row, prediction in zip(vote_matrix, out["prediction"])]
        out["horizon_votes"] = ["|".join(f"{h}:{vote}" for h, vote in zip(models, row)) for row in vote_matrix]
        evidence = rows.apply(self._evidence_direction, axis=1)
        out["evidence"] = evidence.values
        eligible_setup = out["setup"].apply(self._has_eligible_setup)
        out["decision"] = "WAIT"
        qualified = (
            (out["confidence"] >= self.config.minimum_confidence)
            & (out["horizon_agreement"] >= self.config.minimum_horizon_agreement)
            & (out["prediction"] == out["evidence"])
            & (out["prediction"] == out["setup_direction"])
            & eligible_setup
        )
        out.loc[qualified & (out["prediction"] == "UP"), "decision"] = "BUY"
        out.loc[qualified & (out["prediction"] == "DOWN"), "decision"] = "SELL"
        out["score"] = (out["confidence"] * 100).round(1)
        out["reason"] = out.apply(self._decision_reason, axis=1)
        if include_actual:
            out["actual"] = rows["actual"].values
            out["future_move_pct"] = rows["future_move_pct"].values
            out["correct"] = out["prediction"] == out["actual"]
            for horizon in models:
                out[f"actual_{horizon}"] = rows[f"actual_{horizon}"].values
                out[f"future_move_{horizon}_pct"] = rows[f"future_move_{horizon}_pct"].values
                out[f"correct_{horizon}"] = out[f"prediction_{horizon}"] == out[f"actual_{horizon}"]
            outcomes = []
            for decision, buy_outcome, sell_outcome in zip(
                out["decision"], rows["buy_trade_outcome"], rows["sell_trade_outcome"]
            ):
                outcomes.append(buy_outcome if decision == "BUY" else sell_outcome if decision == "SELL" else "NO_TRADE")
            out["trade_outcome"] = outcomes
            out["setup_outcome"] = np.where(
                out["setup_direction"] == "UP",
                rows["buy_trade_outcome"],
                np.where(out["setup_direction"] == "DOWN", rows["sell_trade_outcome"], "NO_SETUP"),
            )
        return out

    def _combine(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        items = []
        for symbol, frame in frames.items():
            prepared = self.prepare(frame)
            prepared["symbol"] = symbol.upper()
            items.append(prepared)
        if not items:
            raise ValueError("At least one stock dataset is required.")
        return pd.concat(items, ignore_index=True).dropna(subset=self.feature_columns)

    def _quality_summary(self, result: pd.DataFrame) -> dict:
        correct = result["prediction"] == result["actual"]
        accuracy = float(correct.mean())
        balanced_accuracy = float(balanced_accuracy_score(result["actual"], result["prediction"]))
        majority_baseline = float(result["actual"].value_counts(normalize=True).max())
        directional = result[result["decision"].isin(["BUY", "SELL"])]
        resolved = directional[directional["trade_outcome"].isin(["TARGET", "STOP"])]
        directional_accuracy = float((directional["prediction"] == directional["actual"]).mean()) if len(directional) else None
        target_rate = float((resolved["trade_outcome"] == "TARGET").mean()) if len(resolved) else None
        accuracy_low, accuracy_high = self._wilson_interval(int(correct.sum()), len(result))
        target_low, target_high = self._wilson_interval(int((resolved["trade_outcome"] == "TARGET").sum()), len(resolved))
        trade_symbols = int(directional["symbol"].nunique())
        predictive_edge = accuracy_low > majority_baseline and balanced_accuracy > (1 / 3)
        trade_edge = (
            len(resolved) >= self.config.minimum_resolved_trades
            and trade_symbols >= self.config.minimum_trade_symbols
            and target_low is not None and target_low > 0.5
        )
        return {
            "unseen_rows": int(len(result)), "accuracy": accuracy,
            "accuracy_ci_95_low": accuracy_low, "accuracy_ci_95_high": accuracy_high,
            "balanced_accuracy": balanced_accuracy,
            "majority_baseline_accuracy": majority_baseline,
            "beats_majority_baseline": accuracy > majority_baseline,
            "predictive_edge": predictive_edge, "trade_edge": trade_edge,
            "model_status": "CANDIDATE" if predictive_edge and trade_edge else "REJECTED",
            "directional_rows": int(len(directional)), "directional_accuracy": directional_accuracy,
            "resolved_trades": int(len(resolved)), "target_before_stop_rate": target_rate,
            "target_rate_ci_95_low": target_low, "target_rate_ci_95_high": target_high,
            "trade_symbols": trade_symbols,
            "ambiguous_trades": int((directional["trade_outcome"] == "AMBIGUOUS").sum()),
            "symbol_metrics": {
                symbol: {
                    "rows": int(len(group)),
                    "accuracy": float((group["prediction"] == group["actual"]).mean()),
                    "majority_baseline_accuracy": float(group["actual"].value_counts(normalize=True).max()),
                    "trades": int(group["decision"].isin(["BUY", "SELL"]).sum()),
                }
                for symbol, group in result.groupby("symbol")
            },
            "horizon_metrics": self._horizon_metrics(result),
            "setup_metrics": self._setup_metrics(result),
        }

    def _horizon_metrics(self, result: pd.DataFrame) -> dict[str, dict]:
        metrics = {}
        for horizon in self.config.horizons:
            actual = f"actual_{horizon}"
            prediction = f"prediction_{horizon}"
            if actual not in result or prediction not in result:
                continue
            eligible = result[result[actual].notna()]
            if eligible.empty:
                continue
            correct = eligible[prediction] == eligible[actual]
            low, high = self._wilson_interval(int(correct.sum()), len(eligible))
            metrics[str(horizon)] = {
                "rows": int(len(eligible)),
                "accuracy": float(correct.mean()),
                "accuracy_ci_95_low": low,
                "accuracy_ci_95_high": high,
                "balanced_accuracy": float(balanced_accuracy_score(eligible[actual], eligible[prediction])),
                "majority_baseline_accuracy": float(eligible[actual].value_counts(normalize=True).max()),
            }
        return metrics

    def _ensure_trainable(self, train: pd.DataFrame, label: str = "actual") -> None:
        if len(train) < self.config.minimum_training_rows:
            raise ValueError(
                f"Need at least {self.config.minimum_training_rows} training rows; got {len(train)}."
            )
        if train[label].nunique() < 2:
            raise ValueError("Training data needs at least two outcome classes.")

    @staticmethod
    def _add_setup_signals(
        data: pd.DataFrame,
        ema21: pd.Series,
        ema50: pd.Series,
        prior_high: pd.Series,
        prior_low: pd.Series,
        atr: pd.Series,
    ) -> None:
        close, open_, high, low = data["close"], data["open"], data["high"], data["low"]
        bull_cross = (ema21 > ema50) & (ema21.shift(1) <= ema50.shift(1))
        bear_cross = (ema21 < ema50) & (ema21.shift(1) >= ema50.shift(1))
        bull_pullback = (
            (ema21 > ema50) & (data["ema_21_slope"] > 0)
            & (low <= ema21 * 1.005) & (close > ema21) & (close > open_)
        )
        bear_pullback = (
            (ema21 < ema50) & (data["ema_21_slope"] < 0)
            & (high >= ema21 * 0.995) & (close < ema21) & (close < open_)
        )
        bull_breakout = (
            (close > prior_high) & (data["volume_ratio_20"] >= 1.2)
            & (data["close_location"] >= 0.65)
        )
        bear_breakout = (
            (close < prior_low) & (data["volume_ratio_20"] >= 1.2)
            & (data["close_location"] <= 0.35)
        )
        last_bull_level = prior_high.where(close > prior_high).shift(1).ffill(limit=5)
        last_bear_level = prior_low.where(close < prior_low).shift(1).ffill(limit=5)
        bull_retest = (
            last_bull_level.notna() & (low <= last_bull_level * 1.005)
            & (close > last_bull_level) & (close > open_)
        )
        bear_retest = (
            last_bear_level.notna() & (high >= last_bear_level * 0.995)
            & (close < last_bear_level) & (close < open_)
        )
        body = (close - open_).abs()
        lower_wick = np.minimum(open_, close) - low
        upper_wick = high - np.maximum(open_, close)
        support_rejection = (
            (low <= prior_low + 0.25 * atr) & (close > prior_low)
            & (lower_wick >= body * 1.2) & (close > open_)
        )
        resistance_rejection = (
            (high >= prior_high - 0.25 * atr) & (close < prior_high)
            & (upper_wick >= body * 1.2) & (close < open_)
        )
        definitions = [
            ("EMA_BULL_CROSS", "UP", bull_cross),
            ("EMA_BEAR_CROSS", "DOWN", bear_cross),
            ("BREAKOUT_VOLUME", "UP", bull_breakout),
            ("BREAKDOWN_VOLUME", "DOWN", bear_breakout),
            ("BREAKOUT_RETEST", "UP", bull_retest),
            ("BREAKDOWN_RETEST", "DOWN", bear_retest),
            ("EMA_BULL_PULLBACK", "UP", bull_pullback),
            ("EMA_BEAR_PULLBACK", "DOWN", bear_pullback),
            ("SUPPORT_REJECTION", "UP", support_rejection),
            ("RESISTANCE_REJECTION", "DOWN", resistance_rejection),
        ]
        setup, direction = [], []
        for index in data.index:
            matches = [(name, side) for name, side, mask in definitions if bool(mask.loc[index])]
            setup.append("|".join(name for name, _ in matches) if matches else "NONE")
            sides = {side for _, side in matches}
            direction.append(next(iter(sides)) if len(sides) == 1 else "CONFLICT" if sides else "FLAT")
        data["setup"] = setup
        data["setup_direction"] = direction

    def _setup_metrics(self, result: pd.DataFrame) -> dict[str, dict]:
        if "setup_outcome" not in result:
            return {}
        metrics = {}
        setup_names = sorted({
            name for value in result["setup"] if value != "NONE" for name in value.split("|")
        })
        for name in setup_names:
            group = result[result["setup"].str.split("|").apply(lambda values: name in values)]
            resolved = group[group["setup_outcome"].isin(["TARGET", "STOP"])]
            targets = int((resolved["setup_outcome"] == "TARGET").sum())
            low, high = self._wilson_interval(targets, len(resolved))
            metrics[name] = {
                "occurrences": int(len(group)),
                "symbols": int(group["symbol"].nunique()),
                "resolved": int(len(resolved)),
                "targets": targets,
                "stops": int((resolved["setup_outcome"] == "STOP").sum()),
                "target_before_stop_rate": float(targets / len(resolved)) if len(resolved) else None,
                "target_rate_ci_95_low": low,
                "target_rate_ci_95_high": high,
            }
        return metrics

    def _add_trade_path_outcomes(self, data: pd.DataFrame) -> None:
        horizon = self.config.trade_evaluation_horizon
        if horizon < 1:
            raise ValueError("trade_evaluation_horizon must be at least 1.")
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
        if row["setup"] == "NONE":
            return "No recognised trade setup"
        if row["setup_direction"] == "CONFLICT":
            return "Conflicting trade setups"
        if not self._has_eligible_setup(row["setup"]):
            return "Setup is measured but not approved for trading"
        if row["prediction"] == "FLAT":
            return "FLAT is most probable"
        if row["confidence"] < self.config.minimum_confidence:
            return "Ensemble confidence below threshold"
        if row["horizon_agreement"] < self.config.minimum_horizon_agreement:
            return "Insufficient agreement across prediction horizons"
        if row["prediction"] != row["evidence"]:
            return "Model direction is not confirmed by technical evidence"
        if row["prediction"] != row["setup_direction"]:
            return "Model direction does not confirm the detected setup"
        return "Setup, probability, horizon agreement, and technical evidence passed"

    def _has_eligible_setup(self, setup: str) -> bool:
        return bool(set(setup.split("|")) & set(self.config.eligible_setups))

    @staticmethod
    def _wilson_interval(successes: int, observations: int) -> tuple[float | None, float | None]:
        """95% Wilson score interval for a binomial rate."""
        if observations <= 0:
            return None, None
        z = 1.959963984540054
        rate = successes / observations
        denominator = 1 + z * z / observations
        centre = (rate + z * z / (2 * observations)) / denominator
        margin = (
            z
            * np.sqrt(
                rate * (1 - rate) / observations
                + z * z / (4 * observations * observations)
            )
            / denominator
        )
        return float(centre - margin), float(centre + margin)

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
