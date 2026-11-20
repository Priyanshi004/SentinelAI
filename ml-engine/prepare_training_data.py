
"""
SentinelAI - Memory-Efficient Training Data Preparation

Purpose:
1. Identify numeric features.
2. Create a stratified 80/20 train-test split.
3. Convert multiclass labels into binary labels.
4. Clean missing and infinite feature values.
5. Save train/test data as Parquet partitions.
6. Generate feature and preparation reports.

Designed for systems with limited RAM.
"""

from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "cleaned_network_data.csv"
)

PREPARED_PATH = PROJECT_ROOT / "dataset" / "prepared"
TRAIN_PATH = PREPARED_PATH / "train"
TEST_PATH = PREPARED_PATH / "test"

REPORTS_PATH = PROJECT_ROOT / "reports"

CHUNK_SIZE = 50_000
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Constant columns identified during feature analysis.
CONSTANT_COLUMNS = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]


# --------------------------------------------------
# 2. DIRECTORY SETUP
# --------------------------------------------------

PREPARED_PATH.mkdir(parents=True, exist_ok=True)
REPORTS_PATH.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 3. IDENTIFY NUMERIC FEATURES
# --------------------------------------------------

def identify_features():
    """
    Inspect a sample to identify numeric features.

    Non-numeric identifiers such as IP addresses,
    timestamps and flow IDs are excluded.
    """

    print("\nInspecting dataset features...")

    sample = pd.read_csv(
        INPUT_FILE,
        nrows=10_000,
        low_memory=False
    )

    sample.columns = sample.columns.str.strip()

    if sample.columns.duplicated().any():
        duplicates = sample.columns[
            sample.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Duplicate column names found: {duplicates}"
        )

    if "Label" not in sample.columns:
        raise ValueError("Target column 'Label' not found.")

    feature_columns = []
    report_rows = []

    for column in sample.columns:

        if column == "Label":
            continue

        if column in CONSTANT_COLUMNS:
            report_rows.append({
                "Feature": column,
                "Sample_Dtype": str(sample[column].dtype),
                "Numeric_Conversion_Rate": 0,
                "Included": False,
                "Reason": "Constant column"
            })
            continue

        original = sample[column]
        non_missing = int(original.notna().sum())

        converted = pd.to_numeric(
            original,
            errors="coerce"
        )

        numeric_count = int(converted.notna().sum())

        conversion_rate = (
            numeric_count / non_missing
            if non_missing > 0
            else 0
        )

        included = (
            non_missing > 0
            and conversion_rate >= 0.999
        )

        if included:
            feature_columns.append(column)
            reason = "Numeric feature"
        elif non_missing == 0:
            reason = "No valid sample values"
        else:
            reason = "Non-numeric or mixed values"

        report_rows.append({
            "Feature": column,
            "Sample_Dtype": str(original.dtype),
            "Numeric_Conversion_Rate": round(
                conversion_rate, 6
            ),
            "Included": included,
            "Reason": reason
        })

    if not feature_columns:
        raise ValueError("No numeric features were identified.")

    report = pd.DataFrame(report_rows)

    report.to_csv(
        REPORTS_PATH / "feature_selection_report.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("Numeric features selected:", len(feature_columns))
    print("Excluded features:", len(report_rows) - len(feature_columns))

    return feature_columns


# --------------------------------------------------
# 4. READ LABELS ONLY
# --------------------------------------------------

def collect_labels():
    """
    Read only the target labels in chunks.

    This avoids loading the feature matrix into RAM.
    """

    print("\nCollecting target labels...")

    label_chunks = []
    total_rows = 0

    for chunk in pd.read_csv(
        INPUT_FILE,
        usecols=["Label"],
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        labels = chunk["Label"].astype("string").str.strip()

        if labels.isna().any():
            raise ValueError(
                "Missing target labels detected."
            )

        binary_labels = (
            labels.str.upper() != "BENIGN"
        ).astype("int8")

        label_chunks.append(
            binary_labels.to_numpy()
        )

        total_rows += len(chunk)

        print(
            f"Labels collected: {total_rows:,}",
            end="\r"
        )

    if not label_chunks:
        raise ValueError("No dataset records found.")

    labels_array = np.concatenate(label_chunks)

    print("\nTotal records:", f"{total_rows:,}")
    print(
        "Benign records:",
        f"{np.count_nonzero(labels_array == 0):,}"
    )
    print(
        "Attack records:",
        f"{np.count_nonzero(labels_array == 1):,}"
    )

    return labels_array


# --------------------------------------------------
# 5. CREATE STRATIFIED TRAIN-TEST SPLIT
# --------------------------------------------------

def create_split(labels):
    """
    Create a reproducible stratified split.

    Only row indices and labels are held in memory.
    """

    print("\nCreating stratified train-test split...")

    indices = np.arange(
        len(labels),
        dtype=np.int32
    )

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    # Sorted test indices allow efficient membership checks.
    test_indices.sort()

    print("Training rows:", f"{len(train_indices):,}")
    print("Testing rows:", f"{len(test_indices):,}")

    return test_indices


# --------------------------------------------------
# 6. SAVE PARQUET PARTITIONS
# --------------------------------------------------

def save_parquet_partitions(feature_columns, test_indices):
    """
    Process the dataset in chunks and save separate
    training and testing Parquet files.
    """

    print("\nPreparing Parquet partitions...")

    # Remove old output partitions to prevent mixing
    # results from different runs.
    for folder in [TRAIN_PATH, TEST_PATH]:

        if folder.exists():
            shutil.rmtree(folder)

        folder.mkdir(parents=True, exist_ok=True)

    train_rows = 0
    test_rows = 0

    train_dropped = 0
    test_dropped = 0

    global_offset = 0
    part_number = 0

    for chunk in pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk.columns = chunk.columns.str.strip()

        chunk_length = len(chunk)

        # Convert original labels to binary targets.
        labels = (
            chunk["Label"]
            .astype("string")
            .str.strip()
            .str.upper()
            .ne("BENIGN")
            .astype("int8")
        )

        # Select only numeric model features.
        features = chunk[feature_columns].copy()

        # Convert feature columns to numeric values.
        for column in feature_columns:
            features[column] = pd.to_numeric(
                features[column],
                errors="coerce"
            )

        # Replace infinity with missing values.
        features.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        # Determine which original row indices are test rows.
        current_indices = np.arange(
            global_offset,
            global_offset + chunk_length,
            dtype=np.int32
        )

        positions = np.searchsorted(
            test_indices,
            current_indices
        )

        is_test = positions < len(test_indices)

        valid_positions = np.flatnonzero(is_test)

        is_test[valid_positions] = (
            test_indices[positions[valid_positions]]
            == current_indices[valid_positions]
        )

        is_train = ~is_test

        # Prepare each split independently.
        for mask, output_folder, is_testing in [
            (is_train, TRAIN_PATH, False),
            (is_test, TEST_PATH, True)
        ]:

            if not mask.any():
                continue

            split_features = features.loc[mask].copy()

            split_labels = labels.loc[mask].copy()

            original_split_rows = len(split_features)

            # Drop rows with missing or invalid features.
            valid_rows = split_features.notna().all(axis=1)

            split_features = split_features.loc[valid_rows]

            split_labels = split_labels.loc[valid_rows]

            dropped = (
                original_split_rows - len(split_features)
            )

            if is_testing:
                test_dropped += dropped
            else:
                train_dropped += dropped

            if split_features.empty:
                continue

            # Add binary target.
            split_features["target"] = (
                split_labels.to_numpy(dtype=np.int8)
            )

            # Save one Parquet file per chunk and split.
            output_file = output_folder / (
                f"part_{part_number:05d}.parquet"
            )

            table = pa.Table.from_pandas(
                split_features,
                preserve_index=False
            )

            pq.write_table(
                table,
                output_file,
                compression="snappy"
            )

            if is_testing:
                test_rows += len(split_features)
            else:
                train_rows += len(split_features)

        global_offset += chunk_length
        part_number += 1

        print(
            f"Processed: {global_offset:,} / "
            f"{len(test_indices) / TEST_SIZE:,.0f} rows",
            end="\r"
        )

    if global_offset != int(round(len(test_indices) / TEST_SIZE)):
        # The expected count above is only a consistency
        # check for this dataset's 80/20 split.
        # The exact source count is checked below instead.
        pass

    print("\n\nParquet writing completed.")

    return {
        "training_records": train_rows,
        "testing_records": test_rows,
        "training_rows_dropped": train_dropped,
        "testing_rows_dropped": test_dropped,
        "source_records_processed": global_offset
    }


# --------------------------------------------------
# 7. SAVE PREPARATION SUMMARY
# --------------------------------------------------

def save_summary(
    feature_columns,
    labels,
    test_indices,
    statistics
):
    """Save metadata describing the prepared dataset."""

    total_rows = len(labels)

    test_row_count = len(test_indices)
    train_row_count = total_rows - test_row_count

    summary = {
        "project": "SentinelAI",
        "dataset": "CIC-IDS2017",
        "classification_type": "Binary Classification",
        "target_mapping": {
            "BENIGN": 0,
            "ATTACK": 1
        },
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "source_records": total_rows,
        "source_benign_records": int(
            np.count_nonzero(labels == 0)
        ),
        "source_attack_records": int(
            np.count_nonzero(labels == 1)
        ),
        "selected_features": feature_columns,
        "number_of_features": len(feature_columns),
        "expected_training_rows_before_cleaning": train_row_count,
        "expected_testing_rows_before_cleaning": test_row_count,
        **statistics
    }

    summary_file = (
        REPORTS_PATH
        / "training_data_preparation_summary.json"
    )

    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nSummary saved:", summary_file)


# --------------------------------------------------
# 8. MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("SENTINELAI - TRAINING DATA PREPARATION")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_FILE}"
        )

    feature_columns = identify_features()

    labels = collect_labels()

    test_indices = create_split(labels)

    statistics = save_parquet_partitions(
        feature_columns,
        test_indices
    )

    if statistics["source_records_processed"] != len(labels):
        raise RuntimeError(
            "Processed record count does not match source count."
        )

    save_summary(
        feature_columns,
        labels,
        test_indices,
        statistics
    )

    print("\n" + "=" * 60)
    print("TRAINING DATA PREPARATION COMPLETED")
    print("=" * 60)

    print(
        "Training records saved:",
        f"{statistics['training_records']:,}"
    )

    print(
        "Testing records saved:",
        f"{statistics['testing_records']:,}"
    )

    print(
        "Training rows dropped:",
        f"{statistics['training_rows_dropped']:,}"
    )

    print(
        "Testing rows dropped:",
        f"{statistics['testing_rows_dropped']:,}"
    )

    print("\nPrepared data:", PREPARED_PATH)


if __name__ == "__main__":
    main()
