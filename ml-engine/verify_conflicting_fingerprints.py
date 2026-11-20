
"""
SentinelAI - Exact Verification of Conflicting Feature Fingerprints

Purpose:
1. Load candidate conflicting fingerprints.
2. Retrieve matching rows from training Parquet partitions.
3. Compare the actual 70 feature values.
4. Identify genuine conflicting-label feature groups.
5. Identify possible 64-bit hash collisions.
6. Save verification reports.

Memory strategy:
- Reads one Parquet partition at a time.
- Retains only rows matching candidate fingerprints.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "dataset" / "prepared" / "train"
MODELS_PATH = PROJECT_ROOT / "models"
REPORTS_PATH = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_PATH / "sentinelai_random_forest.joblib"
CANDIDATE_PATH = REPORTS_PATH / "conflicting_fingerprints.csv"

OUTPUT_JSON = REPORTS_PATH / "conflicting_fingerprint_verification.json"
OUTPUT_CSV = REPORTS_PATH / "verified_conflicting_fingerprints.csv"
COLLISION_CSV = REPORTS_PATH / "fingerprint_hash_collisions.csv"

TARGET_COLUMN = "target"


# --------------------------------------------------
# LOAD CONFIGURATION
# --------------------------------------------------

def load_configuration():
    print("\nLoading model configuration...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"Candidate fingerprint report not found: {CANDIDATE_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)
    feature_columns = bundle["feature_columns"]

    candidates = pd.read_csv(
        CANDIDATE_PATH,
        dtype={"fingerprint": "string"}
    )

    if "fingerprint" not in candidates.columns:
        raise ValueError("Candidate CSV must contain a fingerprint column.")

    candidate_values = candidates["fingerprint"].dropna().tolist()

    candidate_hash_set = {int(value) for value in candidate_values}

    if not candidate_hash_set:
        raise ValueError("No candidate fingerprints were found.")

    candidate_hashes = np.fromiter(
        candidate_hash_set,
        dtype=np.uint64,
        count=len(candidate_hash_set)
    )

    print(f"Features loaded: {len(feature_columns)}")
    print(f"Candidate fingerprint groups: {len(candidate_hash_set)}")

    return feature_columns, candidate_hashes, candidate_hash_set


# --------------------------------------------------
# COLLECT ONLY CANDIDATE RECORDS
# --------------------------------------------------

def collect_candidate_records(feature_columns, candidate_hashes):
    print("\nScanning training Parquet partitions...")

    files = sorted(TRAIN_PATH.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No Parquet files found in {TRAIN_PATH}")

    matched_frames = []
    total_rows_scanned = 0
    matched_rows = 0

    required_columns = feature_columns + [TARGET_COLUMN]

    for index, file in enumerate(files, start=1):
        df = pd.read_parquet(
            file,
            columns=required_columns
        )

        total_rows_scanned += len(df)

        hashes = pd.util.hash_pandas_object(
            df[feature_columns],
            index=False
        ).to_numpy(dtype=np.uint64)

        mask = np.isin(hashes, candidate_hashes)

        if np.any(mask):
            matched = df.loc[mask].copy()

            matched["fingerprint"] = hashes[mask]

            matched_frames.append(matched)
            matched_rows += len(matched)

        print(
            f"[{index}/{len(files)}] "
            f"{file.name} | "
            f"Rows scanned: {total_rows_scanned:,} | "
            f"Matched: {matched_rows:,}"
        )

        del df, hashes, mask

    if not matched_frames:
        raise ValueError(
            "No records matched the candidate fingerprints. "
            "Check the feature columns and fingerprint-generation method."
        )

    candidate_records = pd.concat(
        matched_frames,
        ignore_index=True
    )

    print("\nCollection completed.")
    print(f"Total training rows scanned: {total_rows_scanned:,}")
    print(f"Matching records collected: {len(candidate_records):,}")

    return candidate_records, total_rows_scanned


# --------------------------------------------------
# VERIFY EXACT FEATURE GROUPS
# --------------------------------------------------

def verify_exact_feature_groups(candidate_records, feature_columns):
    print("\nComparing actual feature values...")

    candidate_records["benign_flag"] = (
        candidate_records[TARGET_COLUMN] == 0
    ).astype(np.int8)

    candidate_records["attack_flag"] = (
        candidate_records[TARGET_COLUMN] == 1
    ).astype(np.int8)

    # Group by the fingerprint AND all actual feature values.
    # Different feature rows with the same hash remain separate.
    group_columns = ["fingerprint"] + feature_columns

    exact_groups = (
        candidate_records
        .groupby(
            group_columns,
            dropna=False,
            sort=False,
            observed=True
        )
        .agg(
            total_records=(TARGET_COLUMN, "size"),
            distinct_labels=(TARGET_COLUMN, "nunique"),
            benign_records=("benign_flag", "sum"),
            attack_records=("attack_flag", "sum")
        )
        .reset_index()
    )

    # A genuine conflict requires identical feature values
    # with both BENIGN and ATTACK labels.
    verified_conflicts = exact_groups[
        exact_groups["distinct_labels"] > 1
    ].copy()

    # A fingerprint mapping to multiple exact feature groups
    # is a possible 64-bit hash collision.
    groups_per_fingerprint = (
        exact_groups
        .groupby("fingerprint", sort=False)
        .size()
    )

    collision_fingerprints = groups_per_fingerprint[
        groups_per_fingerprint > 1
    ]

    collision_report = (
        groups_per_fingerprint[
            groups_per_fingerprint > 1
        ]
        .rename("distinct_feature_groups")
        .reset_index()
    )

    collision_report["records_with_fingerprint"] = (
        collision_report["fingerprint"].map(
            candidate_records.groupby("fingerprint").size()
        )
    )

    return exact_groups, verified_conflicts, collision_report


# --------------------------------------------------
# SAVE REPORTS
# --------------------------------------------------

def save_reports(
    candidate_records,
    exact_groups,
    verified_conflicts,
    collision_report,
    total_rows_scanned,
    candidate_hash_set
):
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    verified_conflicts[
        [
            "fingerprint",
            "total_records",
            "benign_records",
            "attack_records"
        ]
    ].to_csv(OUTPUT_CSV, index=False)

    collision_report.to_csv(
        COLLISION_CSV,
        index=False
    )

    verified_records = int(
        verified_conflicts["total_records"].sum()
    )

    benign_records = int(
        verified_conflicts["benign_records"].sum()
    )

    attack_records = int(
        verified_conflicts["attack_records"].sum()
    )

    report = {
        "analysis": "Exact Conflicting Feature Fingerprint Verification",
        "fingerprint_method": "pandas hash_pandas_object (64-bit)",
        "verification_method": (
            "Exact grouping using all 70 original feature values"
        ),
        "training_records_scanned": total_rows_scanned,
        "candidate_fingerprint_count": len(candidate_hash_set),
        "matching_records_collected": len(candidate_records),
        "exact_feature_groups_found": len(exact_groups),
        "verified_conflicting_feature_groups": len(verified_conflicts),
        "records_in_verified_conflicting_groups": verified_records,
        "benign_records_in_verified_conflicts": benign_records,
        "attack_records_in_verified_conflicts": attack_records,
        "possible_hash_collision_fingerprints": len(collision_report),
        "interpretation": (
            "Verified conflicts represent identical feature values "
            "associated with both binary labels. Possible hash collisions "
            "represent one 64-bit fingerprint mapping to multiple distinct "
            "feature groups. Results apply to the candidate fingerprints "
            "screened by the earlier analysis."
        )
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nReports saved successfully!")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Verified conflicts CSV: {OUTPUT_CSV}")
    print(f"Possible collisions CSV: {COLLISION_CSV}")

    return report


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("=" * 65)
    print("SENTINELAI - EXACT FINGERPRINT VERIFICATION")
    print("=" * 65)

    (
        feature_columns,
        candidate_hashes,
        candidate_hash_set
    ) = load_configuration()

    (
        candidate_records,
        total_rows_scanned
    ) = collect_candidate_records(
        feature_columns,
        candidate_hashes
    )

    (
        exact_groups,
        verified_conflicts,
        collision_report
    ) = verify_exact_feature_groups(
        candidate_records,
        feature_columns
    )

    report = save_reports(
        candidate_records,
        exact_groups,
        verified_conflicts,
        collision_report,
        total_rows_scanned,
        candidate_hash_set
    )

    print("\n" + "=" * 65)
    print("VERIFICATION SUMMARY")
    print("=" * 65)

    print(
        "Candidate fingerprints:",
        f"{report['candidate_fingerprint_count']:,}"
    )

    print(
        "Matching records:",
        f"{report['matching_records_collected']:,}"
    )

    print(
        "Exact feature groups:",
        f"{report['exact_feature_groups_found']:,}"
    )

    print(
        "Verified conflicting groups:",
        f"{report['verified_conflicting_feature_groups']:,}"
    )

    print(
        "Records in verified conflicts:",
        f"{report['records_in_verified_conflicting_groups']:,}"
    )

    print(
        "Possible hash collisions:",
        f"{report['possible_hash_collision_fingerprints']:,}"
    )

    print("\nVerification completed successfully.")


if __name__ == "__main__":
    main()
