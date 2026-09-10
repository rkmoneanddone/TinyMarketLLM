from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from dhanhq import DhanContext, dhanhq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dhan_data import (
    fetch_daily,
    validate_ohlc,
    load_stock_universe,
)


SYMBOL = "GRASIM"

FROM_DATE = "2024-12-17"
TO_DATE = "2025-03-03"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "market"
    / SYMBOL
    / "historical_tests"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "GRASIM_2024-12-17_to_2025-02-28_1D.parquet"
)


def load_credentials():
    env_file = PROJECT_ROOT / ".env"

    load_dotenv(env_file)

    client_id = os.getenv("DHAN_CLIENT_ID")
    access_token = os.getenv("DHAN_API_TOKEN")

    if not client_id:
        raise RuntimeError(
            "DHAN_CLIENT_ID is missing from .env"
        )

    if not access_token:
        raise RuntimeError(
            "DHAN_API_TOKEN is missing from .env"
        )

    return client_id, access_token


def main():

    print()
    print("=" * 60)
    print(" TinyMarketLLM - Historical Dhan Window")
    print("=" * 60)
    print()

    print("[1] Loading credentials...")

    client_id, access_token = load_credentials()

    print("[OK] Credentials loaded.")

    print()
    print("[2] Loading stock universe...")

    universe = load_stock_universe()

    company_name = None
    stock = None

    for name, item in universe.items():

        if (
            str(item.get("underlying_symbol", ""))
            .upper()
            .strip()
            == SYMBOL
        ):
            company_name = name
            stock = item
            break

    if stock is None:
        raise ValueError(
            f"{SYMBOL} not found in stock_universe.json"
        )

    security_id = stock["security_id"]

    exchange_segment = stock.get(
        "exchange_segment",
        "NSE_EQ",
    )

    print(
        f"[OK] Company       : {company_name}"
    )

    print(
        f"[OK] Symbol        : {SYMBOL}"
    )

    print(
        f"[OK] Security ID   : {security_id}"
    )

    print(
        f"[OK] Exchange      : {exchange_segment}"
    )

    print()
    print("[3] Requesting Dhan historical data...")

    print(
        f"[DATA] From        : {FROM_DATE}"
    )

    print(
        f"[DATA] To          : {TO_DATE}"
    )

    context = DhanContext(
        client_id,
        access_token,
    )

    dhan = dhanhq(context)

    df = fetch_daily(
        dhan=dhan,
        security_id=security_id,
        exchange_segment=exchange_segment,
        from_date=FROM_DATE,
        to_date=TO_DATE,
    )

    validate_ohlc(df)

    print()
    print(
        f"[OK] Candles received: {len(df)}"
    )

    # --------------------------------------------------
    # Verify the important dates explicitly.
    # --------------------------------------------------

    df["date"] = (
        df["timestamp"]
        .dt.strftime("%Y-%m-%d")
    )

    required_dates = [
        "2025-02-25",
        "2025-02-27",
        "2025-02-28",
    ]

    print()
    print("[4] Verifying required candles...")

    for required_date in required_dates:

        rows = df[
            df["date"] == required_date
        ]

        if rows.empty:
            print(
                f"[MISSING] {required_date}"
            )
        else:
            print(
                f"[OK]      {required_date}"
            )

            print(
                rows[
                    [
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                ].to_string(
                    index=False
                )
            )

    # Remove helper column before saving.
    df = df.drop(
        columns=["date"]
    )

    # --------------------------------------------------
    # Save isolated historical test data.
    # Do NOT overwrite the main market dataset.
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("[5] Historical dataset saved.")

    print(
        f"[OK] File: {OUTPUT_FILE}"
    )

    print()
    print("========== WINDOW ==========")

    print(
        "First:",
        df["timestamp"].iloc[0],
    )

    print(
        "Last :",
        df["timestamp"].iloc[-1],
    )

    print(
        "Rows :",
        len(df),
    )

    print()
    print("[SUCCESS] Historical window ready.")


if __name__ == "__main__":
    main()
