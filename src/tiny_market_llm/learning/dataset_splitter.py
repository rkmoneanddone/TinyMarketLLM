from __future__ import annotations

import pandas as pd
from .schemas import DatasetSplit

class DatasetSplitter:
    def __init__(self, train_ratio: float = 0.70, validation_ratio: float = 0.15):
        if train_ratio <= 0:
            raise ValueError("Train ratio must be > 0.")
        if validation_ratio <= 0:
            raise ValueError("Validation ratio must be > 0.")
        if train_ratio + validation_ratio >= 1:
            raise ValueError("Train + validation ratios must be < 1.")
        self.train_ratio=train_ratio
        self.validation_ratio=validation_ratio

    def split(self, dataset: pd.DataFrame) -> DatasetSplit:
        data = dataset.sort_values("decision_timestamp").reset_index(drop=True)
        total=len(data)
        train_end=int(total*self.train_ratio)
        validation_end=train_end+int(total*self.validation_ratio)
        train=data.iloc[:train_end].copy()
        validation=data.iloc[train_end:validation_end].copy()
        test=data.iloc[validation_end:].copy()
        if train.empty or validation.empty or test.empty:
            raise ValueError("Dataset split produced an empty partition.")
        return DatasetSplit(train=train, validation=validation, test=test)
