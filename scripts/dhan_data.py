from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from dhanhq import DhanContext, dhanhq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.tiny_market_llm.market.timestamp_normalizer import (
    normalize_timestamp_column,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_FILE = PROJECT_ROOT / ".env"
STOCK_UNIVERSE = (
    PROJECT_ROOT
    / "data"
    / "dhan"
    / "stock-list"
    / "stock_universe.json"
)

MARKET_DIR = PROJECT_ROOT / "data" / "market"


def load_environment():
    load_dotenv(ENV_FILE)

    client_id = os.getenv("DHAN_CLIENT_ID")
    access_token = os.getenv("DHAN_API_TOKEN")

    if not client_id:
        raise RuntimeError("DHAN_CLIENT_ID is missing from .env")

    if not access_token:
        raise RuntimeError("DHAN_API_TOKEN is missing from .env")

    return client_id, access_token


def load_stock_universe():
    if not STOCK_UNIVERSE.exists():
        raise FileNotFoundError(
            f"Stock universe not found: {STOCK_UNIVERSE}"
        )

    data = json.loads(
        STOCK_UNIVERSE.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            "stock_universe.json must contain a JSON object"
        )

    return data


def find_stock(universe: dict, symbol: str):
    symbol = symbol.upper().strip()

    for company_name, stock in universe.items():
        if (
            str(stock.get("underlying_symbol", ""))
            .upper()
            .strip()
            == symbol
        ):
            return company_name, stock

    raise ValueError(
        f"Stock '{symbol}' was not found in stock_universe.json"
    )


def fetch_daily(
    dhan,
    security_id: int,
    exchange_segment: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:

    print("[DHAN] Requesting historical daily data...")

    response = dhan.historical_daily_data(
        security_id=str(security_id),
        exchange_segment=exchange_segment,
        instrument_type="EQUITY",
        from_date=from_date,
        to_date=to_date,
    )

    if not isinstance(response, dict):
        raise RuntimeError(
            f"Unexpected Dhan response type: {type(response)}"
        )

    if response.get("status") != "success":
        remarks = response.get("remarks", "")
        raise RuntimeError(
            f"Dhan historical API failed: {remarks}"
        )

    data = response.get("data")

    if not isinstance(data, dict):
        raise RuntimeError(
            "Dhan response does not contain a valid 'data' object."
        )

    required_fields = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        field
        for field in required_fields
        if field not in data
    ]

    if missing:
        raise RuntimeError(
            f"Dhan response data missing fields: {missing}"
        )

    lengths = {
        field: len(data[field])
        for field in required_fields
    }

    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            f"Dhan response arrays have different lengths: {lengths}"
        )

    df = pd.DataFrame(
        {
            "timestamp": data["timestamp"],
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
        }
    )

    df["open"] = pd.to_numeric(
        df["open"],
        errors="coerce",
    )

    df["high"] = pd.to_numeric(
        df["high"],
        errors="coerce",
    )

    df["low"] = pd.to_numeric(
        df["low"],
        errors="coerce",
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    df["volume"] = pd.to_numeric(
        df["volume"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    df = normalize_timestamp_column(
    	df,
    	"timestamp",
    )

    df = df.sort_values("timestamp")

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    df = df.reset_index(drop=True)

    return df

def validate_ohlc(df: pd.DataFrame):
    if df.empty:
        raise RuntimeError("Dhan returned zero OHLC candles.")

    invalid_high = df["high"] < df[["open", "close", "low"]].max(axis=1)
    invalid_low = df["low"] > df[["open", "close", "high"]].min(axis=1)

    if invalid_high.any():
        raise RuntimeError(
            "OHLC validation failed: High is below another OHLC value."
        )

    if invalid_low.any():
        raise RuntimeError(
            "OHLC validation failed: Low is above another OHLC value."
        )

    if df["timestamp"].duplicated().any():
        raise RuntimeError(
            "OHLC validation failed: duplicate timestamps remain."
        )


def save_daily(symbol: str, df: pd.DataFrame) -> Path:
    destination = MARKET_DIR / symbol / "1D"
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = destination / "ohlc.parquet"

    df.to_parquet(
        output_file,
        index=False,
    )

    return output_file


def main():
    print()
    print("========================================")
    print(" TinyMarketLLM - Dhan Data Engine")
    print("========================================")
    print()

    client_id, access_token = load_environment()

    universe = load_stock_universe()

    company_name, stock = find_stock(
        universe,
        "GRASIM",
    )

    security_id = stock["security_id"]
    exchange_segment = stock.get(
        "exchange_segment",
        "NSE_EQ",
    )

    print(f"[STOCK] Company       : {company_name}")
    print(f"[STOCK] Symbol        : {stock['underlying_symbol']}")
    print(f"[STOCK] Security ID   : {security_id}")
    print(f"[STOCK] Exchange      : {exchange_segment}")
    print()

    context = DhanContext(
        client_id,
        access_token,
    )

    dhan = dhanhq(context)

    to_date = date.today()
    from_date = to_date - timedelta(days=3650)

    print(
        f"[DATA] Date range    : "
        f"{from_date} -> {to_date}"
    )
    print("[DATA] Timeframe     : 1D")
    print()

    df = fetch_daily(
        dhan=dhan,
        security_id=security_id,
        exchange_segment=exchange_segment,
        from_date=from_date.isoformat(),
        to_date=to_date.isoformat(),
    )

    validate_ohlc(df)

    output_file = save_daily(
        stock["underlying_symbol"],
        df,
    )

    print()
    print("[OK] Dhan data received")
    print(f"[OK] Candles        : {len(df)}")
    print(f"[OK] First candle   : {df['timestamp'].iloc[0]}")
    print(f"[OK] Last candle    : {df['timestamp'].iloc[-1]}")
    print(f"[OK] Saved          : {output_file}")
    print()

    print("========== LAST 10 CANDLES ==========")
    print(
        df.tail(10).to_string(
            index=False
        )
    )

    print()
    print("[SUCCESS] GRASIM 1D data pipeline completed.")


if __name__ == "__main__":
    main()