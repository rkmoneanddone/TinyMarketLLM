from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MARKET_ROOT = PROJECT_ROOT / "data" / "market"

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


class MarketRepository:
    """
    Persistent storage for normalized market OHLC data.

    This module knows nothing about:
    - Dhan
    - indicators
    - predictions
    - backtesting
    - model training
    """

    def __init__(self, root: Path = MARKET_ROOT):
        self.root = Path(root)

    def dataset_path(self, symbol: str, timeframe: str) -> Path:
        return self.root / symbol.upper() / timeframe / "ohlc.parquet"

    def save(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
    ) -> Path:
        self._validate(data)

        path = self.dataset_path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            data = pd.concat(
                [existing, data],
                ignore_index=True,
            )

        data = self._normalize(data)
        data = self._deduplicate(data)

        data.to_parquet(path, index=False)

        return path

    def load(
        self,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:
        path = self.dataset_path(symbol, timeframe)

        if not path.exists():
            raise FileNotFoundError(
                f"Market dataset not found: {path}"
            )

        data = pd.read_parquet(path)

        self._validate(data)

        return self._normalize(data)

    def exists(
        self,
        symbol: str,
        timeframe: str,
    ) -> bool:
        return self.dataset_path(symbol, timeframe).exists()

    def latest_timestamp(
        self,
        symbol: str,
        timeframe: str,
    ):
        data = self.load(symbol, timeframe)

        if data.empty:
            return None

        return data["timestamp"].max()

    def candle_count(
        self,
        symbol: str,
        timeframe: str,
    ) -> int:
        return len(self.load(symbol, timeframe))

    def _validate(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Market data must be a pandas DataFrame.")

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required market columns: {missing}"
            )

    def _normalize(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        data = data.copy()

        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            utc=True,
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for column in numeric_columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        data = data.dropna(
            subset=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
            ]
        )

        data = data.sort_values("timestamp")

        return data.reset_index(drop=True)

    def _deduplicate(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        return (
            data
            .drop_duplicates(
                subset=["timestamp"],
                keep="last",
            )
            .sort_values("timestamp")
            .reset_index(drop=True)
        )