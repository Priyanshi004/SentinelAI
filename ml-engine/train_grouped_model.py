
"""
SentinelAI - Duplicate-Aware Grouped Random Forest Training

Training:
- Uses dataset/grouped_v1/train
- Stratified reservoir sampling: 80,000 BENIGN + 20,000 ATTACK
- Trains a separate Random Forest model

Evaluation:
- Uses ALL records from dataset/grouped_v1/test
- Calculates accuracy, precision, recall, F1, ROC-AUC and FPR
- Saves metrics, classification report and confusion matrix

The original baseline model is not overwritten.
"""

from pathlib import Path
import json
import time
import gc

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "dataset" / "grouped_v1" / "train"
TEST_DIR = ROOT / "dataset" / "grouped_v1" / "test"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

BASELINE_MODEL = MODEL_DIR / "sentinelai_random_forest.joblib"
GROUPED_MODEL = MODEL_DIR / "sentinelai_random_forest_grouped.joblib"

METRICS_FILE = REPORT_DIR / "grouped_model_metrics.json"
CLASSIFICATION_FILE = REPORT_DIR / "grouped_model_classification_report.json"
CONFUSION_FILE = REPORT_DIR / "grouped_model_confusion_matrix.png"

SAMPLE_LIMITS = {
    0: 80000,   # BENIGN
    1: 20000,   # ATTACK
}

RANDOM_STATE = 42
BATCH_SIZE = 10000


# --------------------------------------------------
# 2. LOAD FEATURE INFORMATION
# --------------------------------------------------

def load_feature_columns():

    if not BASELINE_MODEL.exists():
        raise FileNotFoundError(
            f"Baseline model not found: {BASELINE_MODEL}"
        )

    bundle = joblib.load(BASELINE_MODEL)

    feature_columns = bundle["feature_columns"]

    print(f"Loaded {len(feature_columns)} feature columns.")

    return feature_columns


# --------------------------------------------------
# 3. IDENTIFY TARGET COLUMN
# --------------------------------------------------

def identify_target_column(parquet_file):

    columns = pq.read_schema(parquet_file).names

    candidates = ["target", "Target", "label", "Label"]

    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(
        f"Could not identify target column. Available: {columns}"
    )


# --------------------------------------------------
# 4. STRATIFIED RESERVOIR SAMPLING
# --------------------------------------------------

def create_training_sample(feature_columns, target_column):

    print("\nStarting stratified training sample creation...")

    rng = np.random.default_rng(RANDOM_STATE)

    reservoirs = {
        0: np.empty((0, len(feature_columns)), dtype=np.float32),
        1: np.empty((0, len(feature_columns)), dtype=np.float32),
    }

    priorities = {
        0: np.empty(0, dtype=np.float64),
        1: np.empty(0, dtype=np.float64),
    }

    train_files = sorted(TRAIN_DIR.glob("*.parquet"))

    if not train_files:
        raise FileNotFoundError(
            f"No Parquet files found in {TRAIN_DIR}"
        )

    records_seen = {0: 0, 1: 0}

    required_columns = feature_columns + [target_column]

    for file_number, file_path in enumerate(train_files, start=1):

        df = pd.read_parquet(
            file_path,
            columns=required_columns
        )

        labels = df[target_column].to_numpy(dtype=np.int8)

        features = df[feature_columns].to_numpy(
            dtype=np.float32,
            copy=True
        )

        for class_label in (0, 1):

            mask = labels == class_label

            if not mask.any():
                continue

            class_features = features[mask]
            class_priorities = rng.random(len(class_features))

            records_seen[class_label] += len(class_features)

            combined_features = np.concatenate(
                [reservoirs[class_label], class_features],
                axis=0
            )

            combined_priorities = np.concatenate(
                [priorities[class_label], class_priorities]
            )

            limit = SAMPLE_LIMITS[class_label]

            if len(combined_priorities) > limit:

                selected = np.argpartition(
                    combined_priorities,
                    -limit
                )[-limit:]

                reservoirs[class_label] = combined_features[selected]
                priorities[class_label] = combined_priorities[selected]

            else:

                reservoirs[class_label] = combined_features
                priorities[class_label] = combined_priorities

        del df, features, labels
        gc.collect()

        if file_number % 20 == 0:
            print(
                f"Processed {file_number}/{len(train_files)} "
                f"training partitions..."
            )

    X_train = np.concatenate(
        [reservoirs[0], reservoirs[1]],
        axis=0
    )

    y_train = np.concatenate(
        [
            np.zeros(len(reservoirs[0]), dtype=np.int8),
            np.ones(len(reservoirs[1]), dtype=np.int8)
        ]
    )

    # Shuffle the combined training sample.
    permutation = rng.permutation(len(y_train))

    X_train = X_train[permutation]
    y_train = y_train[permutation]

    sample_summary = {
        "benign_sampled": int(len(reservoirs[0])),
        "attack_sampled": int(len(reservoirs[1])),
        "total_sampled": int(len(y_train)),
        "benign_available": int(records_seen[0]),
        "attack_available": int(records_seen[1]),
    }

    print("\nTraining sample created:")
    print(json.dumps(sample_summary, indent=4))

    return X_train, y_train, sample_summary


# --------------------------------------------------
# 5. TRAIN RANDOM FOREST
# --------------------------------------------------

def train_model(X_train, y_train):

    print("\nStarting grouped Random Forest training...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )

    start_time = time.time()

    model.fit(X_train, y_train)

    training_seconds = time.time() - start_time

    print("\nModel training completed.")
    print(f"Training time: {training_seconds:.2f} seconds")

    return model, training_seconds


# --------------------------------------------------
# 6. EVALUATE ON THE COMPLETE GROUPED TEST SET
# --------------------------------------------------

def evaluate_model(model, feature_columns, target_column):

    print("\nStarting full grouped test evaluation...")

    test_files = sorted(TEST_DIR.glob("*.parquet"))

    if not test_files:
        raise FileNotFoundError(
            f"No Parquet files found in {TEST_DIR}"
        )

    y_true_batches = []
    y_pred_batches = []
    y_probability_batches = []

    total_records = 0

    for file_number, file_path in enumerate(test_files, start=1):

        df = pd.read_parquet(
            file_path,
            columns=feature_columns + [target_column]
        )

        y_true = df[target_column].to_numpy(dtype=np.int8)

        for start in range(0, len(df), BATCH_SIZE):

            end = min(start + BATCH_SIZE, len(df))

            X_batch = df.iloc[start:end][feature_columns].to_numpy(
                dtype=np.float32
            )

            predictions = model.predict(X_batch)

            positive_index = list(model.classes_).index(1)

            probabilities = model.predict_proba(X_batch)[
                :, positive_index
            ]

            y_true_batches.append(y_true[start:end])
            y_pred_batches.append(predictions)
            y_probability_batches.append(probabilities)

            total_records += len(X_batch)

        del df
        gc.collect()

        if file_number % 20 == 0:
            print(
                f"Evaluated {file_number}/{len(test_files)} "
                f"test partitions..."
            )

    y_true_all = np.concatenate(y_true_batches)
    y_pred_all = np.concatenate(y_pred_batches)
    y_probability_all = np.concatenate(y_probability_batches)

    cm = confusion_matrix(
        y_true_all,
        y_pred_all,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": "Random Forest - Duplicate-Aware Grouped Split",
        "training_records": 100000,
        "testing_records": int(total_records),
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "precision": float(
            precision_score(y_true_all, y_pred_all, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_all, y_pred_all, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true_all, y_pred_all, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true_all, y_probability_all)
        ),
        "false_positive_rate": float(
            fp / (fp + tn) if (fp + tn) else 0
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    classification = classification_report(
        y_true_all,
        y_pred_all,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0,
    )

    print("\nGrouped model evaluation results:")
    print(json.dumps(metrics, indent=4))

    return metrics, classification, cm


# --------------------------------------------------
# 7. SAVE CONFUSION MATRIX
# --------------------------------------------------

def save_confusion_matrix(cm):

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["BENIGN", "ATTACK"]
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    plt.title("SentinelAI - Grouped Model Confusion Matrix")
    plt.tight_layout()

    plt.savefig(CONFUSION_FILE, dpi=300)
    plt.close(fig)

    print(f"\nConfusion matrix saved: {CONFUSION_FILE}")


# --------------------------------------------------
# 8. MAIN EXECUTION
# --------------------------------------------------

def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    feature_columns = load_feature_columns()

    first_train_file = sorted(TRAIN_DIR.glob("*.parquet"))[0]

    target_column = identify_target_column(first_train_file)

    print(f"Target column: {target_column}")

    X_train, y_train, sample_summary = create_training_sample(
        feature_columns,
        target_column
    )

    model, training_seconds = train_model(
        X_train,
        y_train
    )

    model_bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "training_type": "duplicate_aware_grouped_split",
        "training_sample": sample_summary,
    }

    joblib.dump(model_bundle, GROUPED_MODEL)

    print(f"\nGrouped model saved: {GROUPED_MODEL}")

    del X_train, y_train
    gc.collect()

    metrics, classification, cm = evaluate_model(
        model,
        feature_columns,
        target_column
    )

    metrics["training_seconds"] = float(training_seconds)
    metrics["training_sample"] = sample_summary
    metrics["split_type"] = "duplicate_aware_grouped_split"

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with open(CLASSIFICATION_FILE, "w", encoding="utf-8") as file:
        json.dump(classification, file, indent=4)

    save_confusion_matrix(cm)

    print("\n" + "=" * 55)
    print("SENTINELAI GROUPED MODEL PIPELINE COMPLETED")
    print("=" * 55)

    print(f"Model: {GROUPED_MODEL}")
    print(f"Metrics: {METRICS_FILE}")
    print(f"Classification: {CLASSIFICATION_FILE}")
    print(f"Confusion matrix: {CONFUSION_FILE}")


if __name__ == "__main__":
    main()
