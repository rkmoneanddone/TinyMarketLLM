from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

import pandas as pd

from src.tiny_market_llm.validation import (
    TechnicalValidator,
)


def main():

    print("=" * 50)
    print(" TinyMarketLLM - Technical Validator Test")
    print("=" * 50)
    print()

    path = (
        PROJECT_ROOT
        / "data"
        / "market"
        / "GRASIM"
        / "1D"
        / "ohlc.parquet"
    )

    print("[1] Loading Dhan OHLC...")

    data = pd.read_parquet(path)

    print(
        f"[OK] Records: {len(data)}"
    )

    print()
    print("[2] Calculating technical state...")

    validator = TechnicalValidator(
        rsi_period=14,
    )

    result = validator.evaluate(
        data
    )

    print("[OK] Technical state calculated.")

    print()
    print("========== RESULT ==========")
    print()

    print(
        "Valid             :",
        result.valid,
    )

    print(
        "Direction         :",
        result.direction,
    )

    print(
        "Price             :",
        round(result.price, 2),
    )

    print()
    print("EMA")
    print(
        "21 EMA            :",
        round(result.ema_21, 2),
    )

    print(
        "50 EMA            :",
        round(result.ema_50, 2),
    )

    print(
        "200 EMA           :",
        round(result.ema_200, 2),
    )

    print(
        "EMA Alignment     :",
        result.ema_alignment,
    )

    print(
        "Alignment Type    :",
        result.ema_alignment_type,
    )

    print()
    print("RSI")
    print(
        "RSI               :",
        round(result.rsi, 2),
    )

    print(
        "RSI Direction     :",
        result.rsi_direction,
    )

    print()
    print("FIBONACCI")

    print(
        "Swing Low         :",
        result.swing_low,
    )

    print(
        "Swing High        :",
        result.swing_high,
    )

    print(
        "Fib 1.618         :",
        (
            round(result.fib_1618, 2)
            if result.fib_1618 is not None
            else None
        ),
    )

    print()
    print("Reasons:")

    for reason in result.reasons:
        print(
            " -",
            reason,
        )

    print()
    print("PASS")


if __name__ == "__main__":
    main()
