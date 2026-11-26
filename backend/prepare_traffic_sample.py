import json
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
test_dir = project_root / "dataset" / "prepared" / "test"
output_file = Path(__file__).parent / "app" / "data" / "traffic_sample.json"

samples = {0: [], 1: []}
required_per_class = 100

for parquet_file in sorted(test_dir.glob("*.parquet")):
    df = pd.read_parquet(parquet_file)

    for target in (0, 1):
        remaining = required_per_class - len(samples[target])

        if remaining > 0:
            matching = df[df["target"] == target].head(remaining)
            samples[target].extend(matching.to_dict(orient="records"))

    print(
        f"Checked {parquet_file.name}: "
        f"BENIGN={len(samples[0])}, ATTACK={len(samples[1])}"
    )

    if all(len(samples[target]) >= required_per_class for target in (0, 1)):
        break

if any(len(samples[target]) < required_per_class for target in (0, 1)):
    raise RuntimeError("Could not collect 100 records from each class.")

records = samples[0] + samples[1]
output_file.write_text(
    json.dumps(records, allow_nan=False),
    encoding="utf-8"
)

print("\nBalanced sample created successfully.")
print("Total records:", len(records))
print("BENIGN:", len(samples[0]))
print("ATTACK:", len(samples[1]))
print("Output:", output_file)
print("File size:", round(output_file.stat().st_size / 1024, 2), "KB")

