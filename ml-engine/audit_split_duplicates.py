
"""
SentinelAI - Train/Test Duplicate Overlap Audit

Audits whether matching feature fingerprints occur in
both training and testing Parquet partitions.

Uses pandas 64-bit row fingerprints to keep memory usage low.
This is a highly practical duplicate-screening method, not
a cryptographic proof of exact row equality.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "dataset" / "prepared" / "train"
TEST_PATH = PROJECT_ROOT / "dataset" / "prepared" / "test"

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "duplicate_overlap_audit.json"
)

TARGET_COLUMN = "target"


# --------------------------------------------------
# 2. LOAD TRAINING FINGERPRINTS
# --------------------------------------------------

def collect_training_fingerprints():

    files = sorted(TRAIN_PATH.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No training Parquet files found in {TRAIN_PATH}"
        )

    hash_parts = []
    target_parts = []

    feature_columns = None

    print("\nCollecting training fingerprints...")

    total_rows = 0

    for file in files:

        df = pd.read_parquet(file)

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column missing in {file.name}"
            )

        current_features = [
            column for column in df.columns
            if column != TARGET_COLUMN
        ]

        if feature_columns is None:
            feature_columns = current_features

        elif current_features != feature_columns:
            raise ValueError(
                f"Feature mismatch in {file.name}"
            )

        fingerprints = pd.util.hash_pandas_object(
            df[feature_columns],
            index=False
        ).to_numpy(dtype=np.uint64)

        targets = df[TARGET_COLUMN].to_numpy(dtype=np.int8)

        hash_parts.append(fingerprints)
        target_parts.append(targets)

        total_rows += len(df)

        print(
            f"Training rows scanned: {total_rows:,}",
            end="\r"
        )

        del df

    train_hashes = np.concatenate(hash_parts)
    train_targets = np.concatenate(target_parts)

    print("\nTraining fingerprint collection completed.")

    return train_hashes, train_targets, feature_columns


# --------------------------------------------------
# 3. BUILD UNIQUE TRAINING FINGERPRINT INDEX
# --------------------------------------------------

def build_training_index(train_hashes, train_targets):

    print("\nBuilding training fingerprint index...")

    sort_order = np.argsort(
        train_hashes,
        kind="mergesort"
    )

    sorted_hashes = train_hashes[sort_order]
    sorted_targets = train_targets[sort_order]

    unique_hashes, first_indices = np.unique(
        sorted_hashes,
        return_index=True
    )

    minimum_targets = np.minimum.reduceat(
        sorted_targets,
        first_indices
    )

    maximum_targets = np.maximum.reduceat(
        sorted_targets,
        first_indices
    )

    repeated_rows = (
        len(train_hashes) - len(unique_hashes)
    )

    conflicting_groups = int(
        np.count_nonzero(
            minimum_targets != maximum_targets
        )
    )

    print("Training records:", f"{len(train_hashes):,}")
    print("Unique fingerprints:", f"{len(unique_hashes):,}")
    print("Repeated training rows:", f"{repeated_rows:,}")
    print("Conflicting-label groups:", f"{conflicting_groups:,}")

    return (
        unique_hashes,
        minimum_targets,
        maximum_targets,
        repeated_rows,
        conflicting_groups
    )


# --------------------------------------------------
# 4. AUDIT TESTING DATA
# --------------------------------------------------

def audit_test_data(
    feature_columns,
    unique_hashes,
    minimum_targets,
    maximum_targets
):

    files = sorted(TEST_PATH.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No testing Parquet files found in {TEST_PATH}"
        )

    total_test_rows = 0
    overlapping_rows = 0
    same_label_overlaps = 0
    opposite_only_overlaps = 0
    ambiguous_label_overlaps = 0

    print("\nAuditing test fingerprints...")

    for file in files:

        df = pd.read_parquet(file)

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column missing in {file.name}"
            )

        if [
            column for column in df.columns
            if column != TARGET_COLUMN
        ] != feature_columns:
            raise ValueError(
                f"Feature mismatch in {file.name}"
            )

        test_hashes = pd.util.hash_pandas_object(
            df[feature_columns],
            index=False
        ).to_numpy(dtype=np.uint64)

        test_targets = df[TARGET_COLUMN].to_numpy(
            dtype=np.int8
        )

        positions = np.searchsorted(
            unique_hashes,
            test_hashes
        )

        valid = positions < len(unique_hashes)

        matched_positions = np.flatnonzero(valid)

        valid[matched_positions] = (
            unique_hashes[positions[matched_positions]]
            == test_hashes[matched_positions]
        )

        matched_count = int(np.count_nonzero(valid))

        overlapping_rows += matched_count

        if matched_count:

            matched_positions = positions[valid]

            matched_test_targets = test_targets[valid]

            train_min = minimum_targets[matched_positions]
            train_max = maximum_targets[matched_positions]

            # At least one training record has the same label.
            same_label = (
                (matched_test_targets >= train_min)
                & (matched_test_targets <= train_max)
            )

            # Training has only the opposite label.
            opposite_only = (
                (matched_test_targets == 0)
                & (train_min == 1)
            ) | (
                (matched_test_targets == 1)
                & (train_max == 0)
            )

            # Same feature fingerprint appears with both labels.
            ambiguous = train_min != train_max

            same_label_overlaps += int(
                np.count_nonzero(same_label)
            )

            opposite_only_overlaps += int(
                np.count_nonzero(opposite_only)
            )

            ambiguous_label_overlaps += int(
                np.count_nonzero(ambiguous)
            )

        total_test_rows += len(df)

        print(
            f"Test rows scanned: {total_test_rows:,}",
            end="\r"
        )

        del df

    overlap_percentage = (
        overlapping_rows / total_test_rows * 100
        if total_test_rows
        else 0
    )

    return {
        "testing_records": total_test_rows,
        "test_records_with_training_feature_match": overlapping_rows,
        "test_feature_overlap_percentage": round(
            overlap_percentage, 6
        ),
        "overlaps_with_same_training_label": same_label_overlaps,
        "overlaps_with_opposite_label_only": opposite_only_overlaps,
        "overlaps_with_ambiguous_training_labels": ambiguous_label_overlaps
    }


# --------------------------------------------------
# 5. MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("SENTINELAI - DUPLICATE OVERLAP AUDIT")
    print("=" * 60)

    train_hashes, train_targets, feature_columns = (
        collect_training_fingerprints()
    )

    (
        unique_hashes,
        minimum_targets,
        maximum_targets,
        repeated_rows,
        conflicting_groups
    ) = build_training_index(
        train_hashes,
        train_targets
    )

    test_results = audit_test_data(
        feature_columns,
        unique_hashes,
        minimum_targets,
        maximum_targets
    )

    report = {
        "audit": "Train/Test Feature Fingerprint Overlap",
        "fingerprint_method": "pandas hash_pandas_object (64-bit)",
        "features_checked": len(feature_columns),
        "training_records": len(train_hashes),
        "unique_training_fingerprints": len(unique_hashes),
        "repeated_training_rows": repeated_rows,
        "training_conflicting_label_groups": conflicting_groups,
        **test_results,
        "interpretation": (
            "Feature fingerprint overlap is a duplicate-screening "
            "signal. It is not a cryptographic proof of exact equality."
        )
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n\n" + "=" * 60)
    print("DUPLICATE OVERLAP AUDIT COMPLETED")
    print("=" * 60)

    print(
    "Test feature overlap:",
    f"{test_results['test_records_with_training_feature_match']:,}"
)

    print(
        "Overlap percentage:",
        f"{report['test_feature_overlap_percentage']}%"
    )

    print("Report saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
