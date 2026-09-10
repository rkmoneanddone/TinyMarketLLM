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

    print()
    print(
        "[4A] Analyzing supporting evidence..."
    )

    evidence = (
        engine.analyze_supporting_evidence(
            split.train,
            supporting_minimum_samples=20,
        )
    )

    print(
        f"[OK] Core states analyzed: "
        f"{len(evidence)}"
    )

    for state_key, state_data in evidence.items():

        print()
        print("=" * 70)
        print(
            f"CORE STATE: {state_key}"
        )
        print(
            f"SAMPLES: {state_data['samples']}"
        )

        print(
            "BASE: "
            f"UP={state_data['base_probability']['UP']:.2%} "
            f"DOWN={state_data['base_probability']['DOWN']:.2%} "
            f"FLAT={state_data['base_probability']['FLAT']:.2%}"
        )

        for feature, values in (
            state_data["features"].items()
        ):

            print()
            print(
                f"  FEATURE: {feature}"
            )

            for value, stats in values.items():

                probability = (
                    stats[
                        "direction_probability"
                    ]
                )

                delta = stats["delta"]

                print(
                    f"    {value}: "
                    f"samples={stats['samples']} "
                    f"UP={probability['UP']:.2%} "
                    f"DOWN={probability['DOWN']:.2%} "
                    f"FLAT={probability['FLAT']:.2%} "
                    f"ΔUP={delta['UP']:+.2%} "
                    f"ΔDOWN={delta['DOWN']:+.2%}"
                )

        print()
        print(
            f"Core state: {state_key}"
        )

        print(
            f"Samples: "
            f"{state_data['samples']}"
        )

        print(
            "Base probability: "
            f"{state_data['base_probability']}"
        )

        for feature, values in (
            state_data["features"].items()
        ):

            print(
                f"  {feature}: "
                f"{len(values)} observed values"
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
        "[4B] Training evidence model..."
    )

    evidence_model = (
        engine.train_with_evidence(
            split.train,
            minimum_evidence_samples=20,
        )
    )

    print(
        "[OK] Evidence patterns: "
        f"{len(evidence_model['patterns'])}"
    )

    print()
    print(
        "[4C] Comparing core vs evidence..."
    )

    core_correct = 0
    core_evaluated = 0

    evidence_correct = 0
    evidence_evaluated = 0

    for _, row in split.validation.iterrows():

        actual = row["label_direction"]

        # Existing/core model
        core_prediction = engine.predict(
            model,
            row,
        )

        if core_prediction is not None:
            core_evaluated += 1

            if (
                core_prediction.direction
                == actual
            ):
                core_correct += 1

        # Evidence model
        evidence_prediction = (
            engine.predict_with_evidence(
                evidence_model,
                row,
            )
        )

        if evidence_prediction.confidence > 0:

            evidence_evaluated += 1

            if (
                evidence_prediction.direction
                == actual
            ):
                evidence_correct += 1

    core_accuracy = (
        core_correct / core_evaluated
        if core_evaluated
        else 0.0
    )

    evidence_accuracy = (
        evidence_correct / evidence_evaluated
        if evidence_evaluated
        else 0.0
    )

    print(
        f"Core accuracy     : "
        f"{core_accuracy:.2%}"
    )

    print(
        f"Evidence accuracy : "
        f"{evidence_accuracy:.2%}"
    )

    print(
        f"Improvement       : "
        f"{evidence_accuracy - core_accuracy:+.2%}"
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