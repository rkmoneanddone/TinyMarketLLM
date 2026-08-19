from __future__ import annotations

import pandas as pd


def dhan_timestamp_to_exchange_timestamp(value) -> pd.Timestamp:
    """
    Convert Dhan historical-chart timestamp to the actual
    NSE trading-session date.

    Dhan's historical daily equity response represents the
    session timestamp one calendar day before the actual
    NSE trading date.

    Example:

        Dhan:  2026-07-19 18:30 UTC
        Candle: 2026-07-20 trading session
    """

    timestamp = pd.to_datetime(
        value,
        unit="s",
        utc=True,
    )

    return timestamp + pd.Timedelta(days=1)


def normalize_timestamp_column(
    data: pd.DataFrame,
    column: str = "timestamp",
) -> pd.DataFrame:

    if column not in data.columns:
        raise ValueError(
            f"Missing timestamp column: {column}"
        )

    result = data.copy()

    result[column] = (
        pd.to_datetime(
            result[column],
            unit="s",
            utc=True,
        )
        + pd.Timedelta(days=1)
    )

    return result