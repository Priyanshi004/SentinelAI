
"""
SentinelAI - Machine Learning Model Service

Responsibilities:
1. Load the trained grouped Random Forest model.
2. Retrieve the expected feature columns.
3. Validate incoming network-flow features.
4. Generate BENIGN / ATTACK predictions.
5. Return prediction probabilities and model details.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np


# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[3]

# Trained model location
MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "sentinelai_random_forest_grouped.joblib"
)


class SentinelAIModelService:

    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.target_column = None

    def load_model(self) -> None:
        """Load the trained Random Forest model."""

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}"
            )

        bundle = joblib.load(MODEL_PATH)

        self.model = bundle["model"]
        self.feature_columns = bundle["feature_columns"]

        self.target_column = bundle.get(
            "target_column",
            "target"
        )

        if len(self.feature_columns) != 70:
            raise ValueError(
                f"Expected 70 features, got {len(self.feature_columns)}"
            )

        if not hasattr(self.model, "predict"):
            raise TypeError(
                "Loaded object is not a valid prediction model."
            )

        if not hasattr(self.model, "predict_proba"):
            raise TypeError(
                "Loaded model does not support probability predictions."
            )

        print("SentinelAI Random Forest loaded successfully.")
        print(f"Model path: {MODEL_PATH}")
        print(f"Features expected: {len(self.feature_columns)}")

    def is_loaded(self) -> bool:
        """Check whether the model is loaded."""

        return self.model is not None

    def get_feature_columns(self) -> list[str]:
        """Return a copy of the expected feature names."""

        return self.feature_columns.copy()

    def predict(
        self,
        input_features: dict[str, Any]
    ) -> dict:
        """Predict whether a network flow is BENIGN or an ATTACK."""

        if not self.is_loaded():
            raise RuntimeError(
                "ML model has not been loaded."
            )

        if not isinstance(input_features, dict):
            raise ValueError(
                "Input features must be provided as a dictionary."
            )

        # Check for missing features
        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in input_features
        ]

        # Check for unexpected features
        extra_features = [
            feature
            for feature in input_features
            if feature not in self.feature_columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}"
            )

        if extra_features:
            raise ValueError(
                f"Unexpected features: {extra_features}"
            )

        # Convert feature values to numeric values
        try:
            values = [
                float(input_features[feature])
                for feature in self.feature_columns
            ]

        except (TypeError, ValueError) as exc:
            raise ValueError(
                "All feature values must be numeric."
            ) from exc

        # Reject NaN and infinite values
        if not np.isfinite(values).all():
            raise ValueError(
                "Feature values must be finite numbers."
            )

        # Convert to NumPy array.
        # This matches the format used during model training.
        input_array = np.array(
            [values],
            dtype=np.float64
        )

        # Generate prediction
        prediction = int(
            self.model.predict(input_array)[0]
        )

        # Generate class probabilities
        probabilities = self.model.predict_proba(
            input_array
        )[0]

        # Find the probability corresponding to ATTACK (class 1)
        positive_index = list(
            self.model.classes_
        ).index(1)

        attack_probability = float(
            probabilities[positive_index]
        )

        benign_probability = 1.0 - attack_probability

        # Convert numeric prediction to readable label
        label = (
            "ATTACK"
            if prediction == 1
            else "BENIGN"
        )

        return {
            "prediction": label,
            "prediction_code": prediction,
            "attack_probability": round(
                attack_probability,
                6
            ),
            "benign_probability": round(
                benign_probability,
                6
            ),
            "confidence": round(
                max(
                    attack_probability,
                    benign_probability
                ),
                6
            ),
            "model": "SentinelAI Random Forest",
        }


# Shared model service instance
model_service = SentinelAIModelService()
