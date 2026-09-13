from __future__ import annotations

import argparse
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


UNIVERSE_PATH = ROOT / "data" / "dhan" / "stock-list" / "stock_universe.json"
REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
SUPPORTED_INTERVALS = {1, 5, 15, 25, 60}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download local-only Dhan intraday OHLC data.")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--interval", type=int, default=60, choices=sorted(SUPPORTED_INTERVALS))
    parser.add_argument("--from-date", type=date.fromisoformat, required=True)
    parser.add_argument("--to-date", type=date.fromisoformat, default=date.today())
    return parser.parse_args()


def load_stocks(symbols: list[str]) -> dict[str, dict]:
    universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    indexed = {
        str(item.get("underlying_symbol", "")).strip().upper(): item
        for item in universe.values()
    }
    requested = [symbol.strip().upper() for symbol in symbols]
    missing = [symbol for symbol in requested if symbol not in indexed]
    if missing:
        raise ValueError(f"Symbols missing from stock universe: {', '.join(missing)}")
    return {symbol: indexed[symbol] for symbol in requested}


def response_frame(response: dict) -> pd.DataFrame:
    if not isinstance(response, dict):
        raise RuntimeError(f"Unexpected Dhan response type: {type(response).__name__}")
    if response.get("status") not in (None, "success"):
        raise RuntimeError(f"Dhan intraday API failed: {response.get('remarks', response)}")
    payload = response.get("data", response)
    missing = [column for column in REQUIRED_COLUMNS if column not in payload]
    if missing:
        raise RuntimeError(f"Dhan response missing fields: {missing}")
    lengths = {column: len(payload[column]) for column in REQUIRED_COLUMNS}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"Dhan response arrays have different lengths: {lengths}")
    frame = pd.DataFrame({column: payload[column] for column in REQUIRED_COLUMNS})
    if frame.empty:
        return frame
    for column in REQUIRED_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=REQUIRED_COLUMNS)
    return normalize_timestamp_column(frame, "timestamp")


def validate(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    if frame["timestamp"].duplicated().any():
        raise RuntimeError("Duplicate timestamps remain after merge.")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise RuntimeError("Invalid OHLC: high is below another price.")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise RuntimeError("Invalid OHLC: low is above another price.")


def fetch_symbol(dhan: dhanhq, stock: dict, start: date, end: date, interval: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=84), end)
        response = dhan.intraday_minute_data(
            security_id=str(stock["security_id"]),
            exchange_segment=stock.get("exchange_segment", "NSE_EQ"),
            instrument_type="EQUITY",
            from_date=cursor.isoformat(),
            to_date=chunk_end.isoformat(),
            interval=interval,
        )
        frame = response_frame(response)
        if not frame.empty:
            frames.append(frame)
        cursor = chunk_end + timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


def save(symbol: str, interval: int, fresh: pd.DataFrame) -> Path:
    timeframe = f"{interval}m" if interval < 60 else "1H"
    path = ROOT / "data" / "market" / symbol / timeframe / "ohlc.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, fresh], ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    combined = combined.reset_index(drop=True)
    validate(combined)
    temporary = path.with_suffix(".parquet.tmp")
    combined.to_parquet(temporary, index=False)
    os.replace(temporary, path)
    return path


def main() -> None:
    args = parse_args()
    if args.from_date > args.to_date:
        raise ValueError("--from-date must not be later than --to-date")

    load_dotenv(ROOT / ".env")
    client_id = os.getenv("DHAN_CLIENT_ID")
    token = os.getenv("DHAN_API_TOKEN")
    if not client_id or not token:
        raise RuntimeError("Set DHAN_CLIENT_ID and DHAN_API_TOKEN in the project .env file.")

    client = dhanhq(DhanContext(client_id, token))
    for symbol, stock in load_stocks(args.symbols).items():
        print(f"[FETCH] {symbol} {args.interval}m: {args.from_date} -> {args.to_date}")
        frame = fetch_symbol(client, stock, args.from_date, args.to_date, args.interval)
        if frame.empty:
            print(f"[EMPTY] {symbol}: Dhan returned no candles")
            continue
        output = save(symbol, args.interval, frame)
        print(
            f"[OK] {symbol}: {len(frame)} received | "
            f"{frame['timestamp'].iloc[0]} -> {frame['timestamp'].iloc[-1]} | {output}"
        )


if __name__ == "__main__":
    main()
