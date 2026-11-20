
"""
SentinelAI - Conflicting Feature Fingerprint Analysis

Purpose:
1. Identify feature fingerprints associated with both BENIGN and ATTACK.
2. Count conflicting fingerprint groups and their records.
3. Measure the percentage of affected training records.
4. Save a detailed JSON report and CSV report.

Memory-conscious:
- Reads Parquet files in batches.
- Stores only fingerprint-level statistics.
- Does not load the complete feature matrix into memory.

Note:
Pandas 64-bit fingerprints are screening identifiers.
A hash collision is theoretically possible, so results
must not be described as cryptographic proof of equality.
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

REPORTS_PATH = PROJECT_ROOT / "reports"

TARGET_COLUMN = "target"

OUTPUT_JSON = REPORTS_PATH / "conflicting_fingerprint_analysis.json"
OUTPUT_CSV = REPORTS_PATH / "conflicting_fingerprints.csv"


# --------------------------------------------------
# 2. COLLECT FINGERPRINT STATISTICS
# --------------------------------------------------

def collect_fingerprint_statistics():

    files = sorted(TRAIN_PATH.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No training Parquet files found: {TRAIN_PATH}"
        )

    fingerprint_stats = {}

    total_rows = 0

    feature_columns = None

    print("\nCollecting training fingerprint statistics...")

    for file in files:

        df = pd.read_parquet(file)

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column missing in {file.name}"
            )

        current_features = [
            column
            for column in df.columns
            if column != TARGET_COLUMN
        ]

        if feature_columns is None:
            feature_columns = current_features

        elif current_features != feature_columns:
            raise ValueError(
                f"Feature mismatch in {file.name}"
            )

        hashes = pd.util.hash_pandas_object(
            df[feature_columns],
            index=False
        ).to_numpy(dtype=np.uint64)

        targets = df[TARGET_COLUMN].to_numpy(
            dtype=np.int8
        )

        unique_hashes, inverse = np.unique(
            hashes,
            return_inverse=True
        )

        benign_counts = np.bincount(
            inverse,
            weights=(targets == 0).astype(np.int32),
            minlength=len(unique_hashes)
        ).astype(np.int64)

        attack_counts = np.bincount(
            inverse,
            weights=(targets == 1).astype(np.int32),
            minlength=len(unique_hashes)
        ).astype(np.int64)

        for index, fingerprint in enumerate(unique_hashes):

            key = int(fingerprint)

            if key not in fingerprint_stats:
                fingerprint_stats[key] = [0, 0]

            fingerprint_stats[key][0] += int(
                benign_counts[index]
            )

            fingerprint_stats[key][1] += int(
                attack_counts[index]
            )

        total_rows += len(df)

        print(
            f"Rows processed: {total_rows:,}",
            end="\r"
        )

        del df, hashes, targets
        del unique_hashes, inverse
        del benign_counts, attack_counts

    print("\nFingerprint collection completed.")

    return fingerprint_stats, total_rows, feature_columns


# --------------------------------------------------
# 3. ANALYZE CONFLICTS
# --------------------------------------------------

def analyze_conflicts(fingerprint_stats, total_rows):

    print("\nAnalyzing conflicting fingerprints...")

    conflicting_groups = []

    conflicting_records = 0
    benign_conflicting_records = 0
    attack_conflicting_records = 0

    for fingerprint, counts in fingerprint_stats.items():

        benign_count, attack_count = counts

        if benign_count > 0 and attack_count > 0:

            group_records = benign_count + attack_count

            conflicting_groups.append({
                "fingerprint": str(fingerprint),
                "benign_records": benign_count,
                "attack_records": attack_count,
                "total_records": group_records
            })

            conflicting_records += group_records
            benign_conflicting_records += benign_count
            attack_conflicting_records += attack_count

    conflicting_groups.sort(
        key=lambda item: item["total_records"],
        reverse=True
    )

    conflict_percentage = (
        conflicting_records / total_rows * 100
        if total_rows
        else 0
    )

    report = {
        "analysis": "Conflicting Feature Fingerprints",
        "fingerprint_method": "pandas hash_pandas_object (64-bit)",
        "training_records": total_rows,
        "unique_fingerprints": len(fingerprint_stats),
        "conflicting_fingerprint_groups": len(conflicting_groups),
        "records_in_conflicting_groups": conflicting_records,
        "benign_records_in_conflicting_groups": (
            benign_conflicting_records
        ),
        "attack_records_in_conflicting_groups": (
            attack_conflicting_records
        ),
        "conflicting_record_percentage": round(
            conflict_percentage, 6
        ),
        "interpretation": (
            "Groups sharing a 64-bit feature fingerprint but having "
            "both binary labels require investigation. Hash matches "
            "are screening signals, not cryptographic proof of "
            "exact feature equality."
        )
    }

    return report, conflicting_groups


# --------------------------------------------------
# 4. SAVE REPORTS
# --------------------------------------------------

def save_reports(report, conflicting_groups):

    REPORTS_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    pd.DataFrame(conflicting_groups).to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nReports saved successfully.")

    print("JSON:", OUTPUT_JSON)
    print("CSV:", OUTPUT_CSV)


# --------------------------------------------------
# 5. MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("SENTINELAI - CONFLICTING FINGERPRINT ANALYSIS")
    print("=" * 60)

    fingerprint_stats, total_rows, feature_columns = (
        collect_fingerprint_statistics()
    )

    report, conflicting_groups = analyze_conflicts(
        fingerprint_stats,
        total_rows
    )

    report["features_checked"] = len(feature_columns)

    save_reports(
        report,
        conflicting_groups
    )

    print("\n" + "=" * 60)
    print("CONFLICT ANALYSIS COMPLETED")
    print("=" * 60)

    print(
        "Training records:",
        f"{total_rows:,}"
    )

    print(
        "Unique fingerprints:",
        f"{len(fingerprint_stats):,}"
    )

    print(
        "Conflicting groups:",
        f"{report['conflicting_fingerprint_groups']:,}"
    )

    print(
        "Records in conflicting groups:",
        f"{report['records_in_conflicting_groups']:,}"
    )

    print(
        "Conflicting record percentage:",
        f"{report['conflicting_record_percentage']}%"
    )


if __name__ == "__main__":
    main()
