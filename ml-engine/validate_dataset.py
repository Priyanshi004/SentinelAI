
from pathlib import Path
from collections import Counter
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "cleaned_network_data.csv"
)

CHUNK_SIZE = 50_000

total_rows = 0
label_counts = Counter()
missing_labels = 0

print("SentinelAI Dataset Validation")
print("-" * 45)

for chunk in pd.read_csv(
    DATASET_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    chunk.columns = chunk.columns.str.strip()

    total_rows += len(chunk)

    labels = chunk["Label"].astype("string").str.strip()

    label_counts.update(labels.value_counts().to_dict())
    missing_labels += labels.isna().sum()

print("\nTotal records:", f"{total_rows:,}")
print("Number of attack categories:", len(label_counts))
print("Missing labels:", missing_labels)

print("\nAttack distribution:")

for label, count in label_counts.most_common():
    percentage = count / total_rows * 100
    print(f"{label}: {count:,} ({percentage:.4f}%)")

print("\nValidation completed.")
