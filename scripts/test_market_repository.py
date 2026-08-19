from pathlib import Path
import sys

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.tiny_market_llm.data.market_repository import MarketRepository


def main():
    repository = MarketRepository()

    symbol = "GRASIM"
    timeframe = "1D"

    print("=" * 40)
    print(" TinyMarketLLM - Market Repository Test")
    print("=" * 40)

    print()
    print("[TEST] Loading existing dataset...")

    data = repository.load(symbol, timeframe)

    print(f"[OK] Symbol       : {symbol}")
    print(f"[OK] Timeframe    : {timeframe}")
    print(f"[OK] Candles      : {len(data)}")
    print(f"[OK] First        : {data['timestamp'].min()}")
    print(f"[OK] Last         : {data['timestamp'].max()}")
    print(f"[OK] Dataset path : {repository.dataset_path(symbol, timeframe)}")

    print()
    print("[TEST] Required columns...")

    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    actual = set(data.columns)

    missing = required - actual

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    print("[OK] All required columns present.")

    print()
    print("[TEST] Duplicate timestamps...")

    duplicates = data["timestamp"].duplicated().sum()

    print(f"[OK] Duplicate candles: {duplicates}")

    if duplicates:
        raise RuntimeError(
            "Duplicate timestamps detected."
        )

    print()
    print("[TEST] Timestamp ordering...")

    ordered = data["timestamp"].is_monotonic_increasing

    print(f"[OK] Ordered: {ordered}")

    if not ordered:
        raise RuntimeError(
            "Timestamp ordering is invalid."
        )

    print()
    print("[TEST] Latest timestamp...")

    print(
        f"[OK] Latest: "
        f"{repository.latest_timestamp(symbol, timeframe)}"
    )

    print()
    print("[TEST] Candle count...")

    print(
        f"[OK] Count: "
        f"{repository.candle_count(symbol, timeframe)}"
    )

    print()
    print("========== RESULT ==========")
    print("PASS")


if __name__ == "__main__":
    main()