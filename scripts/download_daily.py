from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from dhanhq import DhanContext, dhanhq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tiny_market_llm.market.timestamp_normalizer import normalize_timestamp_column

UNIVERSE = ROOT / "data" / "dhan" / "stock-list" / "stock_universe.json"
REQUIRED = ("timestamp", "open", "high", "low", "close", "volume")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Dhan Daily OHLC for named NSE stocks.")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--from-date", type=date.fromisoformat, required=True)
    parser.add_argument("--to-date", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client_id, token = os.getenv("DHAN_CLIENT_ID"), os.getenv("DHAN_API_TOKEN")
    if not client_id or not token:
        raise RuntimeError("Set DHAN_CLIENT_ID and DHAN_API_TOKEN in .env")
    universe = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    indexed = {str(v.get("underlying_symbol", "")).upper(): v for v in universe.values()}
    client = dhanhq(DhanContext(client_id, token))

    for raw_symbol in args.symbols:
        symbol = raw_symbol.strip().upper()
        if symbol not in indexed:
            raise ValueError(f"{symbol} is missing from stock_universe.json")
        stock = indexed[symbol]
        print(f"[FETCH] {symbol} 1D: {args.from_date} -> {args.to_date}")
        response = None
        for attempt in range(7):
            response = client.historical_daily_data(
                security_id=str(stock["security_id"]),
                exchange_segment=stock.get("exchange_segment", "NSE_EQ"),
                instrument_type="EQUITY",
                from_date=args.from_date.isoformat(), to_date=args.to_date.isoformat(),
            )
            remarks = response.get("remarks", {}) if isinstance(response, dict) else {}
            if not isinstance(remarks, dict) or remarks.get("error_code") != "DH-904":
                break
            wait = min(2 ** attempt, 30)
            print(f"[WAIT] Dhan rate limit; retrying in {wait}s")
            time.sleep(wait)
        if not isinstance(response, dict) or response.get("status") != "success":
            raise RuntimeError(f"Dhan Daily API failed for {symbol}: {response}")
        payload = response.get("data", {})
        if any(column not in payload for column in REQUIRED):
            raise RuntimeError(f"Dhan response for {symbol} lacks OHLC fields")
        fresh = pd.DataFrame({column: payload[column] for column in REQUIRED})
        fresh = normalize_timestamp_column(fresh, "timestamp")
        path = ROOT / "data" / "market" / symbol / "1D" / "ohlc.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        combined = pd.concat([old, fresh], ignore_index=True).sort_values("timestamp")
        combined = combined.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        temporary = path.with_suffix(".parquet.tmp")
        combined.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        print(f"[OK] {symbol}: {len(fresh)} received -> {path}")
        time.sleep(1.1)


if __name__ == "__main__":
    main()
