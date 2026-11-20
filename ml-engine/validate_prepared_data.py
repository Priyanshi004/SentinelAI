
"""
SentinelAI - Prepared Dataset Validation

Validates Parquet partitions without loading the complete
training or testing dataset into RAM.
"""

from pathlib import Path
from collections import Counter

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPARED_PATH = PROJECT_ROOT / "dataset" / "prepared"

EXPECTED_FEATURES = 70
BATCH_SIZE = 50_000


def validate_split(split_name):
    folder = PREPARED_PATH / split_name
    files = sorted(folder.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {folder}"
        )

    first_file = pq.ParquetFile(files[0])
    schema = first_file.schema_arrow
    columns = schema.names

    target_candidates = [
        column for column in columns
        if column.lower() in {"target", "label", "class"}
    ]

    if len(target_candidates) != 1:
        raise ValueError(
            f"Could not identify one target column. "
            f"Columns: {columns}"
        )

    target_column = target_candidates[0]
    feature_count = len(columns) - 1

    total_rows = 0
    missing_targets = 0
    class_counts = Counter()

    for file in files:
        parquet_file = pq.ParquetFile(file)

        if parquet_file.schema_arrow.names != columns:
            raise ValueError(
                f"Schema mismatch in {file.name}"
            )

        total_rows += parquet_file.metadata.num_rows

        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=[target_column]
        ):
            target_array = batch.column(0)

            missing_targets += target_array.null_count

            values = target_array.to_numpy(
                zero_copy_only=False
            )

            unique_values, counts = np.unique(
                values,
                return_counts=True
            )

            for value, count in zip(unique_values, counts):
                class_counts[int(value)] += int(count)

    print(f"\n{split_name.upper()} DATASET")
    print("-" * 45)
    print("Parquet files:", len(files))
    print("Records:", f"{total_rows:,}")
    print("Features:", feature_count)
    print("Target column:", target_column)
    print("Missing targets:", missing_targets)

    print("\nClass distribution:")

    for label, count in sorted(class_counts.items()):
        percentage = count / total_rows * 100
        name = "BENIGN" if label == 0 else "ATTACK"
        print(
            f"{name} ({label}): "
            f"{count:,} ({percentage:.4f}%)"
        )

    if feature_count != EXPECTED_FEATURES:
        print(
            f"\nWARNING: Expected {EXPECTED_FEATURES} "
            f"features but found {feature_count}."
        )

    if missing_targets:
        raise ValueError(
            f"Missing target values found in {split_name}"
        )

    if set(class_counts.keys()) != {0, 1}:
        raise ValueError(
            f"Unexpected target values in {split_name}: "
            f"{list(class_counts.keys())}"
        )

    return total_rows, class_counts


def main():
    print("=" * 55)
    print("SENTINELAI - PREPARED DATA VALIDATION")
    print("=" * 55)

    train_rows, train_counts = validate_split("train")
    test_rows, test_counts = validate_split("test")

    print("\n" + "=" * 55)
    print("COMBINED VALIDATION")
    print("=" * 55)

    print("Total prepared records:", f"{train_rows + test_rows:,}")
    print("Training records:", f"{train_rows:,}")
    print("Testing records:", f"{test_rows:,}")

    print("\nValidation completed successfully.")


if __name__ == "__main__":
    main()
