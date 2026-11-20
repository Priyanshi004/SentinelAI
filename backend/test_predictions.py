
from pathlib import Path

import pandas as pd

from app.services.model_service import model_service


ROOT_DIR = Path(__file__).resolve().parents[1]

TEST_DATA_DIR = (
    ROOT_DIR
    / "dataset"
    / "grouped_v1"
    / "test"
)


def main():
    print("\nLoading SentinelAI model...")

    model_service.load_model()

    feature_columns = model_service.get_feature_columns()

    print("\nSearching for real BENIGN and ATTACK records...")

    benign_record = None
    attack_record = None

    parquet_files = sorted(TEST_DATA_DIR.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in {TEST_DATA_DIR}"
        )

    for file_path in parquet_files:
        df = pd.read_parquet(file_path)

        if benign_record is None:
            benign_rows = df[df["target"] == 0]

            if not benign_rows.empty:
                benign_record = benign_rows.iloc[0]

        if attack_record is None:
            attack_rows = df[df["target"] == 1]

            if not attack_rows.empty:
                attack_record = attack_rows.iloc[0]

        if benign_record is not None and attack_record is not None:
            break

    if benign_record is None or attack_record is None:
        raise ValueError(
            "Could not find both BENIGN and ATTACK records."
        )

    print("\nTesting BENIGN network flow...")

    benign_features = {
        feature: float(benign_record[feature])
        for feature in feature_columns
    }

    benign_result = model_service.predict(benign_features)

    print("Actual class: BENIGN")
    print(f"Predicted class: {benign_result['prediction']}")
    print(
        f"Attack probability: "
        f"{benign_result['attack_probability']}"
    )

    print("\nTesting ATTACK network flow...")

    attack_features = {
        feature: float(attack_record[feature])
        for feature in feature_columns
    }

    attack_result = model_service.predict(attack_features)

    print("Actual class: ATTACK")
    print(f"Predicted class: {attack_result['prediction']}")
    print(
        f"Attack probability: "
        f"{attack_result['attack_probability']}"
    )

    print("\nREAL DATA PREDICTION TEST COMPLETED")

    print(
        "\nBENIGN prediction correct:",
        benign_result["prediction"] == "BENIGN"
    )

    print(
        "ATTACK prediction correct:",
        attack_result["prediction"] == "ATTACK"
    )


if __name__ == "__main__":
    main()
