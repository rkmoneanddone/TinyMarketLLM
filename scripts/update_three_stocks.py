from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from dhanhq import DhanContext, dhanhq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tiny_market_llm.market.timestamp_normalizer import normalize_timestamp_column


SYMBOLS = ("GRASIM", "RELIANCE", "TCS")
UNIVERSE_PATH = ROOT / "data" / "dhan" / "stock-list" / "stock_universe.json"


def stock_records() -> dict[str, dict]:
    universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    records = {}
    for symbol in SYMBOLS:
        for company, item in universe.items():
            if str(item.get("underlying_symbol", "")).upper() == symbol:
                records[symbol] = {"company": company, **item}
                break
        if symbol not in records:
            raise ValueError(f"{symbol} is missing from the existing stock universe.")
    return records


def fetch(dhan, stock: dict, from_date: date, to_date: date) -> pd.DataFrame:
    response = dhan.historical_daily_data(
        security_id=str(stock["security_id"]),
        exchange_segment=stock.get("exchange_segment", "NSE_EQ"),
        instrument_type="EQUITY",
        from_date=from_date.isoformat(),
        to_date=to_date.isoformat(),
    )
    if not isinstance(response, dict) or response.get("status") != "success":
        raise RuntimeError(f"Dhan historical API failed: {response.get('remarks', response)}")
    payload = response.get("data", {})
    columns = ("timestamp", "open", "high", "low", "close", "volume")
    if any(column not in payload for column in columns):
        raise RuntimeError("Dhan response is missing required OHLC fields.")
    frame = pd.DataFrame({column: payload[column] for column in columns})
    frame = normalize_timestamp_column(frame, "timestamp")
    return frame


def merge_and_save(symbol: str, fresh: pd.DataFrame) -> Path:
    path = ROOT / "data" / "market" / symbol / "1D" / "ohlc.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, fresh], ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    combined.to_parquet(path, index=False)
    return path


def main() -> None:
    load_dotenv(ROOT / ".env")
    client_id = os.getenv("DHAN_CLIENT_ID")
    token = os.getenv("DHAN_API_TOKEN")
    if not client_id or not token:
        raise RuntimeError("Set DHAN_CLIENT_ID and DHAN_API_TOKEN in .env.")

    dhan = dhanhq(DhanContext(client_id, token))
    for symbol, stock in stock_records().items():
        path = ROOT / "data" / "market" / symbol / "1D" / "ohlc.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            last = pd.to_datetime(existing["timestamp"], utc=True).max().date()
            start = last + timedelta(days=1)
        else:
            start = date.today() - timedelta(days=3650)

        if start > date.today():
            print(f"[SKIP] {symbol}: local data is already current")
            continue

        frame = fetch(dhan, stock, start, date.today())
        if frame.empty:
            print(f"[SKIP] {symbol}: Dhan returned no new candles")
            continue
        saved = merge_and_save(symbol, frame)
        print(f"[OK] {symbol}: {len(frame)} received -> {saved}")


if __name__ == "__main__":
    main()
