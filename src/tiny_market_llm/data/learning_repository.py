from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

LEARNING_DIR = (
    PROJECT_ROOT
    / "data"
    / "learning"
)


class LearningRepository:
    """
    Central repository for learning datasets.

    Dataset naming convention:

        SYMBOL_TIMEFRAME_training_VERSION.parquet

    Example:

        GRASIM_1D_training_v2.parquet
        RELIANCE_1D_training_v2.parquet
        TCS_15M_training_v2.parquet
    """

    def __init__(
        self,
        base_dir: Path | None = None,
    ):
        self.base_dir = (
            base_dir
            if base_dir is not None
            else LEARNING_DIR
        )

    def path(
        self,
        symbol: str,
        timeframe: str,
        version: str = "v2",
    ) -> Path:

        symbol = symbol.upper().strip()
        timeframe = timeframe.upper().strip()
        version = version.lower().strip()

        if not symbol:
            raise ValueError(
                "Symbol cannot be empty."
            )

        if not timeframe:
            raise ValueError(
                "Timeframe cannot be empty."
            )

        if not version:
            raise ValueError(
                "Version cannot be empty."
            )

        return (
            self.base_dir
            / f"{symbol}_{timeframe}_training_{version}.parquet"
        )

    def exists(
        self,
        symbol: str,
        timeframe: str,
        version: str = "v2",
    ) -> bool:

        return self.path(
            symbol,
            timeframe,
            version,
        ).exists()

    def load(
        self,
        symbol: str,
        timeframe: str,
        version: str = "v2",
    ) -> pd.DataFrame:

        dataset_path = self.path(
            symbol,
            timeframe,
            version,
        )

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Learning dataset not found: "
                f"{dataset_path}"
            )

        dataset = pd.read_parquet(
            dataset_path
        )

        if dataset.empty:
            raise ValueError(
                f"Learning dataset is empty: "
                f"{dataset_path}"
            )

        return dataset

    def save(
        self,
        dataset: pd.DataFrame,
        symbol: str,
        timeframe: str,
        version: str = "v2",
    ) -> Path:

        if dataset.empty:
            raise ValueError(
                "Cannot save empty learning dataset."
            )

        self.base_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset_path = self.path(
            symbol,
            timeframe,
            version,
        )

        dataset.to_parquet(
            dataset_path,
            index=False,
        )

        return dataset_path