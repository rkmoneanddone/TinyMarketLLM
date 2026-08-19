from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from src.tiny_market_llm.data.learning_repository import (
    LearningRepository,
)


def main():

    print("=" * 46)
    print(" TinyMarketLLM - Learning Repository Test")
    print("=" * 46)
    print()

    repository = LearningRepository()

    symbol = "GRASIM"
    timeframe = "1D"
    version = "v2"

    print("[1] Resolving dataset path...")

    path = repository.path(
        symbol=symbol,
        timeframe=timeframe,
        version=version,
    )

    print(f"[OK] Path: {path}")

    print()
    print("[2] Checking dataset...")

    if not repository.exists(
        symbol,
        timeframe,
        version,
    ):
        raise RuntimeError(
            f"Dataset does not exist: {path}"
        )

    print("[OK] Dataset exists.")

    print()
    print("[3] Loading dataset...")

    dataset = repository.load(
        symbol,
        timeframe,
        version,
    )

    print(
        f"[OK] Records: {len(dataset)}"
    )

    print(
        f"[OK] Columns: {len(dataset.columns)}"
    )

    print()
    print("========== RESULT ==========")
    print("PASS")


if __name__ == "__main__":
    main()