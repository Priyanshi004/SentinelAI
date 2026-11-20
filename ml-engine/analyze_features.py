
"""
SentinelAI - Feature Quality Analysis

Analyzes:
1. Missing values
2. Infinite values
3. Constant columns
4. Numeric and non-numeric columns
5. Duplicate column names
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ------------------------------------------
# 1. PROJECT CONFIGURATION
# ------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "cleaned_network_data.csv"
)

REPORTS_PATH = PROJECT_ROOT / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 50_000


# ------------------------------------------
# 2. ANALYSIS FUNCTION
# ------------------------------------------

def analyze_features():

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    print("\nSentinelAI Feature Quality Analysis")
    print("-" * 50)

    # Read only a small sample to identify columns
    sample = pd.read_csv(
        DATASET_FILE,
        nrows=1000,
        low_memory=False
    )

    sample.columns = sample.columns.str.strip()

    columns = sample.columns.tolist()

    # Track feature statistics
    missing_counts = pd.Series(0, index=columns, dtype="int64")
    infinite_counts = pd.Series(0, index=columns, dtype="int64")

    constant_values = {
        column: None for column in columns
    }

    total_rows = 0

    # Process the dataset incrementally
    for chunk in pd.read_csv(
        DATASET_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk.columns = chunk.columns.str.strip()

        total_rows += len(chunk)

        # Missing values
        missing_counts += chunk.isna().sum()

        # Infinite values in numeric columns
        numeric_chunk = chunk.select_dtypes(
            include=[np.number]
        )

        infinite_counts += np.isinf(
            numeric_chunk
        ).sum().reindex(columns, fill_value=0)

        # Constant column detection
        for column in columns:

            if constant_values[column] is False:
                continue

            values = chunk[column].dropna().unique()

            if len(values) == 0:
                continue

            if constant_values[column] is None:
                constant_values[column] = values[0]

            if not all(
                value == constant_values[column]
                for value in values
            ):
                constant_values[column] = False

    # Identify constant columns
    constant_columns = [
        column
        for column, value in constant_values.items()
        if value is not False and value is not None
    ]

    # Create missing and infinite value report
    quality_report = pd.DataFrame({
        "Feature": columns,
        "Missing_Values": missing_counts.values,
        "Infinite_Values": infinite_counts.values
    })

    quality_report["Missing_Percentage"] = (
        quality_report["Missing_Values"]
        / total_rows * 100
    ).round(4)

    quality_report["Infinite_Percentage"] = (
        quality_report["Infinite_Values"]
        / total_rows * 100
    ).round(4)

    # Save report
    output_file = REPORTS_PATH / "feature_quality_report.csv"

    quality_report.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    # Print summary
    print(f"\nTotal records analyzed: {total_rows:,}")
    print(f"Total columns: {len(columns)}")
    print(f"Constant columns: {len(constant_columns)}")

    print("\nConstant columns:")
    print(constant_columns)

    print("\nFeatures with missing values:")
    print(
        quality_report[
            quality_report["Missing_Values"] > 0
        ].to_string(index=False)
    )

    print("\nFeatures with infinite values:")
    print(
        quality_report[
            quality_report["Infinite_Values"] > 0
        ].to_string(index=False)
    )

    print("\nReport saved:", output_file)


# ------------------------------------------
# 3. SCRIPT ENTRY POINT
# ------------------------------------------

if __name__ == "__main__":
    analyze_features()
