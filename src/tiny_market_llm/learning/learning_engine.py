from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import DatasetSplit
from .dataset_splitter import DatasetSplitter
from .state_builder import StateBuilder
from .pattern_learner import PatternLearner
from .evidence_analyzer import EvidenceAnalyzer
from .predictor import Predictor
from .evaluator import Evaluator
from .model_repository import ModelRepository


class LearningEngine:
    """Thin V2 coordinator. Business logic lives in supporting modules."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        minimum_samples: int = 20,
        promotion_threshold: float = 0.80,
    ):
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.minimum_samples = minimum_samples
        self.promotion_threshold = promotion_threshold

        self.splitter = DatasetSplitter(
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )

        self.state_builder = StateBuilder()

        self.pattern_learner = PatternLearner(
            minimum_samples=minimum_samples,
            promotion_threshold=promotion_threshold,
        )

        self.evidence_analyzer = EvidenceAnalyzer(
            minimum_samples=minimum_samples,
        )

        self.predictor = Predictor()

        self.evaluator = Evaluator(
            promotion_threshold=promotion_threshold,
        )

        self.repository = ModelRepository()

    def split(
        self,
        dataset: pd.DataFrame,
    ) -> DatasetSplit:
        self.state_builder.validate(dataset)
        return self.splitter.split(dataset)

    def train(
        self,
        train_data: pd.DataFrame,
    ) -> dict:
        states = self.state_builder.prepare(train_data)
        return self.pattern_learner.train(states)

    def analyze_supporting_evidence(
        self,
        train_data: pd.DataFrame,
        supporting_minimum_samples: int = 20,
    ) -> dict:
        states = self.state_builder.prepare(train_data)

        return self.evidence_analyzer.analyze(
            states,
            supporting_minimum_samples,
        )

    def train_with_evidence(
        self,
        train_data: pd.DataFrame,
        minimum_evidence_samples: int = 20,
    ) -> dict:
        states = self.state_builder.prepare(train_data)

        return self.evidence_analyzer.train_with_evidence(
            states,
            minimum_evidence_samples,
        )

    def predict(
        self,
        model,
        row,
        timeframe="1D",
        horizon_candles=1,
    ):
        return self.predictor.predict(
            model,
            row,
            timeframe=timeframe,
            horizon_candles=horizon_candles,
        )

    def predict_with_evidence(
        self,
        model,
        row,
        timeframe="1D",
        horizon_candles=1,
    ):
        return self.predictor.predict_with_evidence(
            model,
            row,
            timeframe=timeframe,
            horizon_candles=horizon_candles,
        )

    def evaluate(
        self,
        model,
        data: pd.DataFrame,
    ) -> dict:
        states = self.state_builder.prepare(data)

        return self.evaluator.evaluate(
            model,
            states,
            mode="core",
        )

    def save_candidate(
        self,
        model,
        path: Path,
    ) -> Path:
        return self.repository.save(
            model,
            path,
        )

    def save(
        self,
        model,
        path: Path,
    ) -> Path:
        return self.repository.save(
            model,
            path,
        )
