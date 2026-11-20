
"""
SentinelAI - Random Forest Training Module

Binary classification:
0 = BENIGN
1 = ATTACK

Dataset: CIC-IDS2017
Model: Random Forest Classifier

Features:
- Class-stratified sampling
- Memory-efficient Parquet processing
- Model evaluation
- Classification reports
- Confusion matrix
- Feature importance
- Model serialization
"""

from pathlib import Path
import json
import gc

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "dataset" / "prepared" / "train"
TEST_PATH = PROJECT_ROOT / "dataset" / "prepared" / "test"

MODEL_PATH = PROJECT_ROOT / "models"
REPORT_PATH = PROJECT_ROOT / "reports"

MODEL_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Controlled sample sizes for an 8 GB RAM laptop.
TRAIN_SAMPLE_SIZE = 100_000
TEST_SAMPLE_SIZE = 100_000

BATCH_SIZE = 10_000

TARGET_COLUMN = "target"

CLASS_NAMES = {
    0: "BENIGN",
    1: "ATTACK",
}


# ==================================================
# 2. COUNT DATASET CLASSES
# ==================================================

def count_classes(folder):
    """
    Count records belonging to each target class
    without loading the entire dataset into RAM.
    """

    files = sorted(folder.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found: {folder}"
        )

    counts = {
        0: 0,
        1: 0,
    }

    print(f"\nCounting classes in {folder.name}...")

    for file in files:

        parquet_file = pq.ParquetFile(file)

        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=[TARGET_COLUMN],
        ):

            values = batch.column(0).to_numpy(
                zero_copy_only=False
            )

            unique, frequencies = np.unique(
                values,
                return_counts=True,
            )

            for label, frequency in zip(
                unique,
                frequencies,
            ):

                label = int(label)

                if label not in counts:
                    raise ValueError(
                        f"Unexpected target label: {label}"
                    )

                counts[label] += int(frequency)

    return counts


# ==================================================
# 3. CALCULATE STRATIFIED SAMPLE QUOTAS
# ==================================================

def calculate_quotas(class_counts, sample_size):
    """
    Allocate sample records proportionally
    between BENIGN and ATTACK classes.
    """

    total_records = sum(class_counts.values())

    if total_records == 0:
        raise ValueError("Dataset contains no records.")

    if sample_size > total_records:
        raise ValueError(
            f"Requested {sample_size:,} records, "
            f"but only {total_records:,} exist."
        )

    if sample_size < 2:
        raise ValueError(
            "Sample size must be at least 2."
        )

    if any(count == 0 for count in class_counts.values()):
        raise ValueError(
            "Both classes must be present."
        )

    exact_quotas = {
        label: sample_size * count / total_records
        for label, count in class_counts.items()
    }

    quotas = {
        label: int(np.floor(value))
        for label, value in exact_quotas.items()
    }

    remaining = sample_size - sum(quotas.values())

    labels_by_fraction = sorted(
        class_counts,
        key=lambda label: (
            exact_quotas[label] - quotas[label]
        ),
        reverse=True,
    )

    for label in labels_by_fraction[:remaining]:
        quotas[label] += 1

    # Ensure both classes have at least one sample.
    for label in class_counts:

        if quotas[label] == 0:

            other_label = 1 - label

            if quotas[other_label] <= 1:
                raise ValueError(
                    "Sample size is too small for both classes."
                )

            quotas[label] = 1
            quotas[other_label] -= 1

    for label, quota in quotas.items():

        if quota > class_counts[label]:
            raise ValueError(
                f"Insufficient records for class {label}."
            )

    return quotas


# ==================================================
# 4. LOAD CLASS-STRATIFIED PARQUET SAMPLE
# ==================================================

def load_parquet_sample(folder, sample_size, seed):
    """
    Generate a uniform class-stratified sample.

    Uses random priorities to retain the required
    number of records from each class.
    """

    files = sorted(folder.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found: {folder}"
        )

    rng = np.random.default_rng(seed)

    # First pass: count class records.
    class_counts = count_classes(folder)

    quotas = calculate_quotas(
        class_counts,
        sample_size,
    )

    print("\nAvailable class records:", class_counts)
    print("Stratified sample quotas:", quotas)

    # Second pass: collect samples.
    reservoirs = {
        0: None,
        1: None,
    }

    print("\nCollecting stratified samples...")

    for file in files:

        parquet_file = pq.ParquetFile(file)

        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
        ):

            chunk = batch.to_pandas()

            for label in (0, 1):

                candidates = chunk.loc[
                    chunk[TARGET_COLUMN] == label
                ].copy()

                if candidates.empty:
                    continue

                candidates["_sampling_key"] = rng.random(
                    len(candidates)
                )

                existing = reservoirs[label]

                if existing is None:

                    combined = candidates

                else:

                    combined = pd.concat(
                        [existing, candidates],
                        ignore_index=True,
                    )

                quota = quotas[label]

                if len(combined) > quota:

                    keys = combined[
                        "_sampling_key"
                    ].to_numpy()

                    selected_positions = np.argpartition(
                        keys,
                        quota - 1,
                    )[:quota]

                    combined = combined.iloc[
                        selected_positions
                    ]

                reservoirs[label] = combined

            del chunk

        print(f"Processed: {file.name}")

    # Combine class samples.
    samples = []

    for label in (0, 1):

        reservoir = reservoirs[label]

        if (
            reservoir is None
            or len(reservoir) != quotas[label]
        ):

            raise ValueError(
                f"Sampling failed for class {label}."
            )

        samples.append(
            reservoir.drop(
                columns=["_sampling_key"]
            )
        )

    result = pd.concat(
        samples,
        ignore_index=True,
    )

    result = result.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    print("\nFinal sample distribution:")

    print(
        result[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print("\nTotal sampled records:", len(result))

    return result


# ==================================================
# 5. PREPARE FEATURES AND TARGET
# ==================================================

def prepare_xy(data):
    """
    Separate input features and target.

    Converts features to float32 to reduce
    memory consumption.
    """

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    y = data[TARGET_COLUMN].astype("int8")

    X = data.drop(
        columns=[TARGET_COLUMN]
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if X.isna().any().any():
        raise ValueError(
            "Missing or infinite feature values found."
        )

    return X.astype("float32"), y


# ==================================================
# 6. TRAIN RANDOM FOREST
# ==================================================

def train_model(X_train, y_train):
    """
    Train the Random Forest classifier.
    """

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

    print("\n" + "=" * 50)
    print("RANDOM FOREST TRAINING")
    print("=" * 50)

    print("Training records:", f"{len(X_train):,}")
    print("Features:", X_train.shape[1])
    print("Estimators:", model.n_estimators)

    print("\nTraining in progress...")

    model.fit(
        X_train,
        y_train,
    )

    print("\nModel training completed.")

    return model


# ==================================================
# 7. EVALUATE RANDOM FOREST
# ==================================================

def evaluate_model(model, X_test, y_test, training_count):
    """
    Evaluate model predictions and generate reports.
    """

    print("\nEvaluating model...")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
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
        "training_records": int(training_count),
        "testing_records": int(len(y_test)),
        "number_of_features": int(X_test.shape[1]),
    }

    # Display metrics safely.
    print("\n" + "=" * 50)
    print("CLASSIFICATION METRICS")
    print("=" * 50)

    for name, value in metrics.items():

        if isinstance(value, (float, np.floating)):
            print(f"{name}: {value:.4f}")

        elif isinstance(value, (int, np.integer)):
            print(f"{name}: {value:,}")

        else:
            print(f"{name}: {value}")

    # Classification report.
    report_text = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    )

    print("\nCLASSIFICATION REPORT")
    print("-" * 50)
    print(report_text)

    report_data = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0,
    )

    # Save metrics.
    with open(
        REPORT_PATH / "model_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    with open(
        REPORT_PATH / "classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report_data,
            file,
            indent=4,
        )

    # ------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------

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
        "SentinelAI - Random Forest Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_PATH / "confusion_matrix.png",
        dpi=150,
    )

    plt.close()

    # ------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------

    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False,
    )

    importance.to_csv(
        REPORT_PATH / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(10, 8))

    sns.barplot(
        data=importance.head(15),
        x="Importance",
        y="Feature",
    )

    plt.title(
        "SentinelAI - Top 15 Important Features"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_PATH / "feature_importance.png",
        dpi=150,
    )

    plt.close()

    print("\nReports saved successfully.")

    return metrics


# ==================================================
# 8. SAVE TRAINED MODEL
# ==================================================

def save_model(model, feature_columns, training_count):
    """
    Save the model with its feature schema.
    """

    model_bundle = {
        "model": model,
        "feature_columns": list(feature_columns),
        "target_mapping": {
            "BENIGN": 0,
            "ATTACK": 1,
        },
        "model_name": "RandomForestClassifier",
        "random_state": RANDOM_STATE,
        "training_sample_size": int(training_count),
    }

    model_file = (
        MODEL_PATH
        / "sentinelai_random_forest.joblib"
    )

    joblib.dump(
        model_bundle,
        model_file,
        compress=3,
    )

    print("\nModel saved successfully:")
    print(model_file)

    return model_file


# ==================================================
# 9. MAIN TRAINING PIPELINE
# ==================================================

def main():

    print("=" * 55)
    print("SENTINELAI - RANDOM FOREST TRAINING")
    print("=" * 55)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Testing directory not found: {TEST_PATH}"
        )

    # Load training data.
    print("\nLoading training sample...")

    train_data = load_parquet_sample(
        TRAIN_PATH,
        TRAIN_SAMPLE_SIZE,
        RANDOM_STATE,
    )

    # Load testing data.
    print("\nLoading testing sample...")

    test_data = load_parquet_sample(
        TEST_PATH,
        TEST_SAMPLE_SIZE,
        RANDOM_STATE + 1,
    )

    # Prepare features.
    print("\nPreparing features...")

    X_train, y_train = prepare_xy(train_data)

    X_test, y_test = prepare_xy(test_data)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and testing feature columns do not match."
        )

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            "Both datasets must contain BENIGN and ATTACK records."
        )

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class distribution:")
    print(y_test.value_counts().sort_index())

    # Release original DataFrames.
    del train_data, test_data
    gc.collect()

    # Train model.
    model = train_model(
        X_train,
        y_train,
    )

    # Evaluate model.
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        len(y_train),
    )

    # Save model.
    save_model(
        model,
        X_train.columns,
        len(y_train),
    )

    print("\n" + "=" * 55)
    print("SENTINELAI TRAINING PIPELINE COMPLETED")
    print("=" * 55)

    print(
        f"\nAccuracy: {metrics['accuracy']:.4f}"
    )

    print(
        f"F1 Score: {metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC: {metrics['roc_auc']:.4f}"
    )

    print("\nGenerated files:")
    print("1. models/sentinelai_random_forest.joblib")
    print("2. reports/model_metrics.json")
    print("3. reports/classification_report.json")
    print("4. reports/confusion_matrix.png")
    print("5. reports/feature_importance.csv")
    print("6. reports/feature_importance.png")

    # Release memory.
    del X_train, X_test, y_train, y_test, model

    gc.collect()


# ==================================================
# 10. SCRIPT EXECUTION
# ==================================================

if __name__ == "__main__":
    main()
