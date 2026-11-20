
"""
SentinelAI - Duplicate-Aware Grouped Dataset Split

Purpose:
1. Read existing prepared train and test Parquet partitions.
2. Calculate a fingerprint using all 70 model features.
3. Assign identical fingerprints to the same split.
4. Create a new approximately 80/20 grouped dataset.
5. Save class distribution and split statistics.

Original prepared data is never modified.

Memory strategy:
- Reads one Parquet partition at a time.
- Does not maintain a global fingerprint dictionary.
- Uses deterministic hash-based assignment.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_TRAIN = PROJECT_ROOT / "dataset" / "prepared" / "train"
SOURCE_TEST = PROJECT_ROOT / "dataset" / "prepared" / "test"

OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "grouped_v1"
OUTPUT_TRAIN = OUTPUT_ROOT / "train"
OUTPUT_TEST = OUTPUT_ROOT / "test"

REPORTS_PATH = PROJECT_ROOT / "reports"
OUTPUT_REPORT = REPORTS_PATH / "grouped_split_summary.json"

MODEL_PATH = PROJECT_ROOT / "models" / "sentinelai_random_forest.joblib"

TARGET_COLUMN = "target"

# Approximately 20% of fingerprints go to test.
# All records sharing a fingerprint get the same assignment.
TEST_MODULO = 5
TEST_REMAINDER = 0


# --------------------------------------------------
# LOAD FEATURE CONFIGURATION
# --------------------------------------------------

def load_feature_columns():

    import joblib

    print("\nLoading model feature configuration...")

    bundle = joblib.load(MODEL_PATH)
    feature_columns = bundle["feature_columns"]

    print(f"Features used: {len(feature_columns)}")

    return feature_columns


# --------------------------------------------------
# PREPARE OUTPUT DIRECTORIES
# --------------------------------------------------

def prepare_output_directories():

    OUTPUT_TRAIN.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEST.mkdir(parents=True, exist_ok=True)
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    existing_files = (
        list(OUTPUT_TRAIN.glob("*.parquet"))
        + list(OUTPUT_TEST.glob("*.parquet"))
    )

    if existing_files:
        raise FileExistsError(
            "\nGrouped dataset output already contains Parquet files.\n"
            f"Location: {OUTPUT_ROOT}\n"
            "To avoid mixing old and new results, move or remove "
            "the generated grouped_v1 folder before rerunning."
        )


# --------------------------------------------------
# PROCESS ONE SOURCE PARTITION
# --------------------------------------------------

def process_partition(
    file_path,
    source_name,
    feature_columns,
    statistics
):

    df = pd.read_parquet(file_path)

    required_columns = feature_columns + [TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {file_path.name}: {missing_columns}"
        )

    # Keep only model features and target.
    df = df[required_columns].copy()

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Missing target values found in {file_path.name}"
        )

    # Calculate deterministic 64-bit feature fingerprints.
    fingerprints = pd.util.hash_pandas_object(
        df[feature_columns],
        index=False
    ).to_numpy(dtype=np.uint64)

    # Deterministic group assignment.
    test_mask = (
        fingerprints % np.uint64(TEST_MODULO)
    ) == np.uint64(TEST_REMAINDER)

    train_mask = ~test_mask

    train_df = df.loc[train_mask]
    test_df = df.loc[test_mask]

    # Save partition files separately.
    output_name = f"{source_name}_{file_path.stem}.parquet"

    if len(train_df):
        train_df.to_parquet(
            OUTPUT_TRAIN / output_name,
            index=False
        )

    if len(test_df):
        test_df.to_parquet(
            OUTPUT_TEST / output_name,
            index=False
        )

    # Update statistics.
    statistics["total_records"] += len(df)
    statistics["train_records"] += len(train_df)
    statistics["test_records"] += len(test_df)

    for label, count in df[TARGET_COLUMN].value_counts().items():
        statistics["overall_label_counts"][str(int(label))] += int(count)

    for label, count in train_df[TARGET_COLUMN].value_counts().items():
        statistics["train_label_counts"][str(int(label))] += int(count)

    for label, count in test_df[TARGET_COLUMN].value_counts().items():
        statistics["test_label_counts"][str(int(label))] += int(count)

    return len(df), len(train_df), len(test_df)


# --------------------------------------------------
# MAIN PROCESS
# --------------------------------------------------

def main():

    print("=" * 65)
    print("SENTINELAI - DUPLICATE-AWARE GROUPED DATASET SPLIT")
    print("=" * 65)

    feature_columns = load_feature_columns()

    prepare_output_directories()

    statistics = {
        "method": "Deterministic 64-bit feature fingerprint assignment",
        "fingerprint_features": len(feature_columns),
        "test_assignment_rule": (
            f"fingerprint % {TEST_MODULO} == {TEST_REMAINDER}"
        ),
        "total_records": 0,
        "train_records": 0,
        "test_records": 0,
        "overall_label_counts": {"0": 0, "1": 0},
        "train_label_counts": {"0": 0, "1": 0},
        "test_label_counts": {"0": 0, "1": 0},
        "source_partitions_processed": 0
    }

    sources = [
        ("original_train", SOURCE_TRAIN),
        ("original_test", SOURCE_TEST)
    ]

    for source_name, source_directory in sources:

        files = sorted(source_directory.glob("*.parquet"))

        if not files:
            raise FileNotFoundError(
                f"No Parquet files found in {source_directory}"
            )

        print(f"\nProcessing source: {source_name}")
        print(f"Partitions found: {len(files)}")

        for index, file_path in enumerate(files, start=1):

            total, train_count, test_count = process_partition(
                file_path,
                source_name,
                feature_columns,
                statistics
            )

            statistics["source_partitions_processed"] += 1

            print(
                f"[{index}/{len(files)}] {file_path.name} | "
                f"Total: {total:,} | "
                f"Train: {train_count:,} | "
                f"Test: {test_count:,}"
            )

    # --------------------------------------------------
    # CALCULATE FINAL METRICS
    # --------------------------------------------------

    total_records = statistics["total_records"]

    train_records = statistics["train_records"]
    test_records = statistics["test_records"]

    statistics["train_percentage"] = round(
        train_records / total_records * 100, 4
    )

    statistics["test_percentage"] = round(
        test_records / total_records * 100, 4
    )

    statistics["output_train_directory"] = str(OUTPUT_TRAIN)
    statistics["output_test_directory"] = str(OUTPUT_TEST)

    statistics["interpretation"] = (
        "Identical feature fingerprints are deterministically assigned "
        "to the same split. This prevents matching fingerprints from "
        "being distributed across training and testing. The split is "
        "approximately 80/20, not guaranteed to be exactly stratified. "
        "Fingerprint collisions are theoretically possible."
    )

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as file:
        json.dump(statistics, file, indent=4)

    # --------------------------------------------------
    # DISPLAY SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("GROUPED SPLIT COMPLETED")
    print("=" * 65)

    print(f"Total records: {total_records:,}")
    print(f"Training records: {train_records:,}")
    print(f"Testing records: {test_records:,}")

    print(
        f"Training percentage: "
        f"{statistics['train_percentage']}%"
    )

    print(
        f"Testing percentage: "
        f"{statistics['test_percentage']}%"
    )

    print("\nOverall labels:", statistics["overall_label_counts"])
    print("Training labels:", statistics["train_label_counts"])
    print("Testing labels:", statistics["test_label_counts"])

    print(f"\nReport saved: {OUTPUT_REPORT}")

    print("\nNew grouped dataset:")
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
