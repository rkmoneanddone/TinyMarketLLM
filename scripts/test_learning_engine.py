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

from src.tiny_market_llm.learning.learning_engine import (
    LearningEngine,
)


def main():

    print("=" * 46)
    print(" TinyMarketLLM - Learning Engine Test")
    print("=" * 46)
    print()

    symbol = "GRASIM"
    timeframe = "1D"
    version = "v2"

    print("[1] Loading learning dataset...")

    repository = LearningRepository()

    dataset = repository.load(
        symbol=symbol,
        timeframe=timeframe,
        version=version,
    )

    print(
        f"[OK] Records: {len(dataset)}"
    )

    print()
    print("[2] Creating chronological split...")

    engine = LearningEngine(
        train_ratio=0.70,
        validation_ratio=0.15,
        minimum_samples=20,
        promotion_threshold=0.80,
    )

    split = engine.split(
        dataset
    )

    print(
        f"[OK] Train      : {len(split.train)}"
    )

    print(
        f"[OK] Validation : {len(split.validation)}"
    )

    print(
        f"[OK] Test       : {len(split.test)}"
    )

    print()
    print("[3] Checking chronological separation...")

    if not (
        split.train["decision_timestamp"].max()
        <
        split.validation["decision_timestamp"].min()
    ):
        raise RuntimeError(
            "Train/validation overlap detected."
        )

    if not (
        split.validation["decision_timestamp"].max()
        <
        split.test["decision_timestamp"].min()
    ):
        raise RuntimeError(
            "Validation/test overlap detected."
        )

    print(
        "[OK] No chronological overlap."
    )

    print()
    print("[4] Training candidate model...")

    model = engine.train(
        split.train
    )

    patterns = model.get(
        "patterns",
        [],
    )

    print(
        f"[OK] Patterns learned: {len(patterns)}"
    )

    print()
    print(
        "[5] Evaluating on unseen validation data..."
    )

    validation_result = engine.evaluate(
        model,
        split.validation,
    )

    print(
        f"Evaluated : "
        f"{validation_result['evaluated']}"
    )

    print(
        f"Correct   : "
        f"{validation_result['correct']}"
    )

    print(
        f"Accuracy  : "
        f"{validation_result['accuracy']:.2%}"
    )

    gate = (
        validation_result["accuracy"]
        >= engine.promotion_threshold
    )

    print(
        f"80% Gate  : "
        f"{'PASS' if gate else 'FAIL'}"
    )

    print()
    print(
        "[6] Evaluating on untouched test data..."
    )

    test_result = engine.evaluate(
        model,
        split.test,
    )

    print(
        f"Evaluated : "
        f"{test_result['evaluated']}"
    )

    print(
        f"Correct   : "
        f"{test_result['correct']}"
    )

    print(
        f"Accuracy  : "
        f"{test_result['accuracy']:.2%}"
    )

    print()
    print("[7] Sample learned patterns...")

    for pattern in patterns[:10]:

        print(
            pattern
        )

    print()
    print("[8] Saving candidate model...")

    output = Path(
        "data"
    ) / "models" / "candidate_grasim_v2.json"

    engine.save(
        model,
        output,
    )

    print(
        f"[OK] Candidate saved: {output}"
    )

    print()
    print("========== RESULT ==========")
    print("PASS")


if __name__ == "__main__":
    main()