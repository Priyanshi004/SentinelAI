
"""
SentinelAI - Dataset Preprocessing Module

Purpose:
1. Locate raw CIC-IDS2017 CSV files.
2. Read network traffic data in memory-efficient chunks.
3. Clean column names and labels.
4. Handle infinite and missing values.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ------------------------------------------
# 1. PROJECT PATH CONFIGURATION
# ------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "dataset" / "processed"
REPORTS_PATH = PROJECT_ROOT / "reports"

CHUNK_SIZE = 50_000


# ------------------------------------------
# 2. CREATE REQUIRED DIRECTORIES
# ------------------------------------------

PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
REPORTS_PATH.mkdir(parents=True, exist_ok=True)


# ------------------------------------------
# 3. FIND RAW DATASET FILES
# ------------------------------------------

def get_dataset_files():
    """Return all CSV files in the raw dataset folder."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {RAW_DATA_PATH}"
        )

    files = sorted(RAW_DATA_PATH.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            "No CSV files found in dataset/raw."
        )

    return files


# ------------------------------------------
# 4. CLEAN NETWORK TRAFFIC DATA
# ------------------------------------------

def clean_chunk(df):
    """Clean one chunk of network traffic records."""

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Remove leading and trailing spaces from labels
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype("string").str.strip()

    # Replace infinite values with missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove rows without a valid target label
    if "Label" in df.columns:
        df.dropna(subset=["Label"], inplace=True)

    return df

# ------------------------------------------
# 5. PROCESS DATASET IN CHUNKS
# ------------------------------------------

def process_dataset():
    """Read, clean and save the dataset in chunks."""

    dataset_files = get_dataset_files()

    total_processed = 0
    total_skipped = 0

    output_file = PROCESSED_DATA_PATH / "cleaned_network_data.csv"

    # Start with a fresh output file for this run
    if output_file.exists():
        output_file.unlink()

    print("\nSentinelAI Dataset Preprocessing Started")
    print("-" * 50)

    for file in dataset_files:

        print(f"\nProcessing file: {file.name}")

        file_processed = 0

        try:
            for chunk in pd.read_csv(
                file,
                chunksize=CHUNK_SIZE,
                low_memory=False
            ):

                original_rows = len(chunk)

                # Clean the current chunk
                chunk = clean_chunk(chunk)

                # Count rows removed due to missing labels
                skipped_rows = original_rows - len(chunk)
                total_skipped += skipped_rows

                # Skip empty chunks
                if chunk.empty:
                    continue

                # Save incrementally
                chunk.to_csv(
                    output_file,
                    mode="a",
                    index=False,
                    header=not output_file.exists()
                    or output_file.stat().st_size == 0
                )

                processed_rows = len(chunk)

                file_processed += processed_rows
                total_processed += processed_rows

            print(f"Records processed: {file_processed:,}")

        except Exception as error:
            print(f"Error processing {file.name}: {error}")
            raise

    print("\n" + "-" * 50)
    print("DATASET PREPROCESSING COMPLETED")
    print("-" * 50)

    print(f"Total processed records: {total_processed:,}")
    print(f"Rows skipped: {total_skipped:,}")
    print(f"Output file: {output_file}")


# ------------------------------------------
# 6. SCRIPT ENTRY POINT
# ------------------------------------------

if __name__ == "__main__":
    process_dataset()
