
"""
SentinelAI - Grouped Split Validation

Checks:
1. Every training fingerprint follows the training assignment rule.
2. Every testing fingerprint follows the testing assignment rule.
3. No missing or invalid target labels.
4. No NaN or infinite feature values.
5. Records and class distributions across both splits.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "dataset" / "grouped_v1" / "train"
TEST_PATH = PROJECT_ROOT / "dataset" / "grouped_v1" / "test"

MODEL_PATH = PROJECT_ROOT / "models" / "sentinelai_random_forest.joblib"

REPORT_PATH = PROJECT_ROOT / "reports" / "grouped_split_validation.json"

TARGET_COLUMN = "target"

TEST_MODULO = 5
TEST_REMAINDER = 0


def validate_partition(directory, split_name, feature_columns):

    files = sorted(directory.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet partitions found: {directory}"
        )

    results = {
        "split": split_name,
        "partitions": len(files),
        "records": 0,
        "assignment_violations": 0,
        "invalid_targets": 0,
        "missing_feature_values": 0,
        "infinite_feature_values": 0,
        "label_counts": {"0": 0, "1": 0}
    }

    expected_test = split_name == "test"

    for index, file_path in enumerate(files, start=1):

        df = pd.read_parquet(
            file_path,
            columns=feature_columns + [TARGET_COLUMN]
        )

        features = df[feature_columns]

        hashes = pd.util.hash_pandas_object(
            features,
            index=False
        ).to_numpy(dtype=np.uint64)

        assigned_to_test = (
            hashes % np.uint64(TEST_MODULO)
        ) == np.uint64(TEST_REMAINDER)

        if expected_test:
            violations = ~assigned_to_test
        else:
            violations = assigned_to_test

        results["assignment_violations"] += int(
            np.count_nonzero(violations)
        )

        targets = df[TARGET_COLUMN]

        results["invalid_targets"] += int(
            (~targets.isin([0, 1])).sum()
        )

        results["missing_feature_values"] += int(
            features.isna().sum().sum()
        )

        numeric_features = features.select_dtypes(
            include=[np.number]
        )

        results["infinite_feature_values"] += int(
            np.isinf(numeric_features.to_numpy()).sum()
        )

        for label, count in targets.value_counts().items():

            label_key = str(int(label))

            if label_key in results["label_counts"]:
                results["label_counts"][label_key] += int(count)

        results["records"] += len(df)

        print(
            f"[{index}/{len(files)}] {file_path.name} | "
            f"Records: {len(df):,} | "
            f"Assignment violations: "
            f"{int(np.count_nonzero(violations)):,}"
        )

        del df, features, hashes, assigned_to_test

    return results


def main():

    print("=" * 60)
    print("SENTINELAI - GROUPED SPLIT VALIDATION")
    print("=" * 60)

    bundle = joblib.load(MODEL_PATH)
    feature_columns = bundle["feature_columns"]

    print(f"Features checked: {len(feature_columns)}")

    train_results = validate_partition(
        TRAIN_PATH,
        "train",
        feature_columns
    )

    test_results = validate_partition(
        TEST_PATH,
        "test",
        feature_columns
    )

    total_records = (
        train_results["records"] + test_results["records"]
    )

    total_violations = (
        train_results["assignment_violations"]
        + test_results["assignment_violations"]
    )

    total_invalid_targets = (
        train_results["invalid_targets"]
        + test_results["invalid_targets"]
    )

    total_missing = (
        train_results["missing_feature_values"]
        + test_results["missing_feature_values"]
    )

    total_infinite = (
        train_results["infinite_feature_values"]
        + test_results["infinite_feature_values"]
    )

    report = {
        "validation": "Duplicate-Aware Grouped Split",
        "fingerprint_features": len(feature_columns),
        "assignment_rule": (
            f"fingerprint % {TEST_MODULO} == {TEST_REMAINDER}"
        ),
        "total_records": total_records,
        "train": train_results,
        "test": test_results,
        "total_assignment_violations": total_violations,
        "total_invalid_targets": total_invalid_targets,
        "total_missing_feature_values": total_missing,
        "total_infinite_feature_values": total_infinite,
        "validation_passed": (
            total_records > 0
            and total_violations == 0
            and total_invalid_targets == 0
            and total_missing == 0
            and total_infinite == 0
        ),
        "interpretation": (
            "Zero assignment violations confirm that every matching "
            "64-bit feature fingerprint follows the same split rule. "
            "This prevents matching fingerprints from crossing splits. "
            "Theoretical hash collisions remain a limitation."
        )
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"Total records: {total_records:,}")
    print(f"Training records: {train_results['records']:,}")
    print(f"Testing records: {test_results['records']:,}")
    print(f"Assignment violations: {total_violations:,}")
    print(f"Invalid targets: {total_invalid_targets:,}")
    print(f"Missing feature values: {total_missing:,}")
    print(f"Infinite feature values: {total_infinite:,}")

    print("\nTraining labels:", train_results["label_counts"])
    print("Testing labels:", test_results["label_counts"])

    print(
        "\nValidation passed:",
        report["validation_passed"]
    )

    print(f"\nReport saved: {REPORT_PATH}")

    if not report["validation_passed"]:
        raise RuntimeError(
            "Grouped split validation failed. "
            "Do not retrain until the issues are investigated."
        )


if __name__ == "__main__":
    main()
