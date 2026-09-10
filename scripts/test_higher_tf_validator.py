from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from src.tiny_market_llm.validation import HigherTFValidator


def main():

    print("=" * 46)
    print(" TinyMarketLLM - Higher TF Validator Test")
    print("=" * 46)
    print()

    validator = HigherTFValidator()

    result = validator.validate(
        current_price=1220.0,
        direction="UP",
        structure="HL",

        ma_21=1215.0,
        ma_50=1195.0,
        ma_200=1140.0,

        previous_high=1245.0,
        previous_low=1120.0,

        swing_high=1235.0,
        swing_low=1180.0,
    )

    print("========== RESULT ==========")
    print()

    print("Valid                  :", result.valid)
    print("Direction              :", result.direction)
    print("MA Alignment           :", result.ma_alignment)
    print("MA Alignment Type      :", result.ma_alignment_type)
    print("Structure Valid        :", result.structure_valid)
    print("Previous Level         :", result.previous_level)
    print(
        "Previous Level Distance:",
        result.previous_level_distance_pct,
    )
    print("Fib 1.618              :", result.fib_1618)
    print("Target                 :", result.target)
    print("Target Type            :", result.target_type)
    print("Fib Target             :", result.fib_target)

    print()
    print("Reasons:")

    for reason in result.reasons:
        print(" -", reason)

    print()
    print("PASS")


if __name__ == "__main__":
    main()
