
"""
SentinelAI - Full Test Dataset Evaluation

Evaluates the saved Random Forest model against
the complete prepared test dataset.

Memory-efficient batch processing for 8 GB RAM.
"""

from pathlib import Path
import json
import gc

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = PROJECT_ROOT / "dataset" / "prepared" / "test"

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "sentinelai_random_forest.joblib"
)

REPORT_PATH = PROJECT_ROOT / "reports"

REPORT_PATH.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "target"

BATCH_SIZE = 10_000

RANDOM_STATE = 42


# ==================================================
# 2. LOAD MODEL
# ==================================================

def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    print("\nLoading saved Random Forest model...")

    bundle = joblib.load(MODEL_FILE)

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    print("Model:", bundle["model_name"])
    print("Features:", len(feature_columns))

    return model, feature_columns


# ==================================================
# 3. VALIDATE FEATURES
# ==================================================

def prepare_batch(batch_df, feature_columns):

    if TARGET_COLUMN not in batch_df.columns:
        raise ValueError(
            "Target column missing from test data."
        )

    missing_columns = [
        column
        for column in feature_columns
        if column not in batch_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing features: {missing_columns}"
        )

    X = batch_df[feature_columns]

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if X.isna().any().any():
        raise ValueError(
            "Missing or infinite feature values found."
        )

    X = X.astype("float32")

    y = batch_df[TARGET_COLUMN].astype("int8")

    if not y.isin([0, 1]).all():
        raise ValueError(
            "Unexpected target values found."
        )

    return X, y


# ==================================================
# 4. EVALUATE COMPLETE TEST DATASET
# ==================================================

def evaluate_full_dataset(model, feature_columns):

    files = sorted(TEST_PATH.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {TEST_PATH}"
        )

    all_actual = []
    all_predictions = []
    all_probabilities = []

    total_records = 0

    print("\nStarting full test-set evaluation...")
    print("Parquet files:", len(files))
    print("Batch size:", BATCH_SIZE)

    for file_index, file in enumerate(files, start=1):

        print(
            f"\nProcessing file {file_index}/{len(files)}: "
            f"{file.name}"
        )

        parquet_file = pq.ParquetFile(file)

        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE
        ):

            batch_df = batch.to_pandas()

            X_batch, y_batch = prepare_batch(
                batch_df,
                feature_columns,
            )

            predictions = model.predict(X_batch)

            probabilities = model.predict_proba(
                X_batch
            )[:, 1]

            all_actual.extend(
                y_batch.to_numpy().tolist()
            )

            all_predictions.extend(
                predictions.tolist()
            )

            all_probabilities.extend(
                probabilities.tolist()
            )

            total_records += len(y_batch)

            if total_records % 50_000 < len(y_batch):
                print(
                    f"Evaluated records: {total_records:,}"
                )

            del batch_df
            del X_batch
            del y_batch
            del predictions
            del probabilities

            gc.collect()

    print("\nAll test records processed.")

    y_true = np.asarray(
        all_actual,
        dtype=np.int8,
    )

    y_pred = np.asarray(
        all_predictions,
        dtype=np.int8,
    )

    y_prob = np.asarray(
        all_probabilities,
        dtype=np.float32,
    )

    return y_true, y_pred, y_prob


# ==================================================
# 5. GENERATE METRICS AND REPORTS
# ==================================================

def generate_reports(y_true, y_pred, y_prob):

    print("\nGenerating evaluation metrics...")

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    metrics = {
        "model": "RandomForestClassifier",
        "evaluation_type": "Full test dataset",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "false_positive_rate": float(false_positive_rate),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "testing_records": int(len(y_true)),
        "random_state": RANDOM_STATE,
    }

    print("\n" + "=" * 55)
    print("SENTINELAI - FULL TEST DATASET RESULTS")
    print("=" * 55)

    for name, value in metrics.items():

        if isinstance(value, (float, np.floating)):
            print(f"{name}: {value:.6f}")

        elif isinstance(value, (int, np.integer)):
            print(f"{name}: {value:,}")

        else:
            print(f"{name}: {value}")

    # Classification report.
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0,
    )

    print("\nCLASSIFICATION REPORT")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["BENIGN", "ATTACK"],
            zero_division=0,
        )
    )

    # Save metrics.
    with open(
        REPORT_PATH / "full_test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # Save classification report.
    with open(
        REPORT_PATH / "full_test_classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    # Save confusion matrix.
    plt.figure(figsize=(7, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["BENIGN", "ATTACK"],
        yticklabels=["BENIGN", "ATTACK"],
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.title(
        "SentinelAI - Full Test Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_PATH / "full_test_confusion_matrix.png",
        dpi=150,
    )

    plt.close()

    print("\nFull evaluation reports saved successfully.")

    return metrics


# ==================================================
# 6. MAIN
# ==================================================

def main():

    print("=" * 55)
    print("SENTINELAI - FULL TEST DATASET EVALUATION")
    print("=" * 55)

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    model, feature_columns = load_model()

    y_true, y_pred, y_prob = evaluate_full_dataset(
        model,
        feature_columns,
    )

    metrics = generate_reports(
        y_true,
        y_pred,
        y_prob,
    )

    print("\n" + "=" * 55)
    print("FULL TEST EVALUATION COMPLETED")
    print("=" * 55)

    print(
        f"Records evaluated: {metrics['testing_records']:,}"
    )

    print(
        f"Accuracy: {metrics['accuracy']:.6f}"
    )

    print(
        f"F1 Score: {metrics['f1_score']:.6f}"
    )

    print("\nGenerated reports:")
    print("1. reports/full_test_metrics.json")
    print("2. reports/full_test_classification_report.json")
    print("3. reports/full_test_confusion_matrix.png")

    del y_true, y_pred, y_prob, model
    gc.collect()


if __name__ == "__main__":
    main()
