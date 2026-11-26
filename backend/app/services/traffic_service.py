"""
SentinelAI - Simulated Network Traffic Service

Uses CIC-IDS2017 test records to simulate incoming network flows.
Every flow is classified using the trained Random Forest model.

This is a dataset-driven simulation, not live packet capture.
"""

import json
import threading

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.model_service import model_service


# backend/app/data/traffic_sample.json
APP_DIR = Path(__file__).resolve().parents[1]
SAMPLE_PATH = APP_DIR / "data" / "traffic_sample.json"

MAX_RECENT_EVENTS = 50


class TrafficSimulationService:

    def __init__(self):
        self.samples: list[dict[str, Any]] = []
        self.sample_index = 0

        self.is_running = False

        self.total_flows = 0
        self.benign_flows = 0
        self.attack_flows = 0

        self.recent_events = deque(
            maxlen=MAX_RECENT_EVENTS
        )

        self.lock = threading.Lock()

    def load_samples(self) -> None:
        """Load the prepared CIC-IDS2017 sample records."""

        if self.samples:
            return

        if not SAMPLE_PATH.exists():
            raise FileNotFoundError(
                f"Traffic sample not found: {SAMPLE_PATH}"
            )

        with SAMPLE_PATH.open("r", encoding="utf-8") as file:
            records = json.load(file)

        if not isinstance(records, list) or not records:
            raise ValueError(
                "Traffic sample must contain a non-empty JSON list."
            )

        expected_features = set(
            model_service.get_feature_columns()
        )

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid traffic record at index {index}."
                )

            if "target" not in record:
                raise ValueError(
                    f"Missing target column in record {index}."
                )

            record_features = set(record) - {"target"}

            if record_features != expected_features:
                raise ValueError(
                    f"Feature mismatch in traffic record {index}."
                )

        self.samples = records

        print(
            f"Traffic sample loaded successfully: "
            f"{len(self.samples)} records."
        )

    def start(self) -> dict:
        """Start the traffic simulation."""

        if not model_service.is_loaded():
            raise RuntimeError(
                "ML model must be loaded before starting traffic simulation."
            )

        with self.lock:
            self.load_samples()
            self.is_running = True

            return self.get_status()

    def stop(self) -> dict:
        """Stop the traffic simulation."""

        with self.lock:
            self.is_running = False

            return self.get_status()

    def get_status(self) -> dict:
        """Return current simulation status and traffic statistics."""

        attack_rate = (
            self.attack_flows / self.total_flows * 100
            if self.total_flows
            else 0.0
        )

        return {
            "is_running": self.is_running,
            "total_flows": self.total_flows,
            "benign_flows": self.benign_flows,
            "attack_flows": self.attack_flows,
            "attack_rate": round(attack_rate, 2),
            "sample_records": len(self.samples),
            "recent_events": list(self.recent_events),
        }

    def process_next_flow(self) -> dict | None:
        """Process one simulated flow using the Random Forest model."""

        with self.lock:

            if not self.is_running:
                return None

            if not model_service.is_loaded():
                raise RuntimeError(
                    "ML model is not loaded."
                )

            self.load_samples()

            # Cycle through the sample records.
            record = self.samples[
                self.sample_index
            ]

            self.sample_index = (
                self.sample_index + 1
            ) % len(self.samples)

            # Exclude the dataset target from model input.
            features = {
                key: value
                for key, value in record.items()
                if key != "target"
            }

            # Generate an actual ML prediction.
            prediction = model_service.predict(features)

            self.total_flows += 1

            if prediction["prediction"] == "ATTACK":
                self.attack_flows += 1
            else:
                self.benign_flows += 1

            self.total_flows_id = self.total_flows

            event = {
                "flow_id": f"SIM-{self.total_flows_id:06d}",
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "destination_port": int(
                    record["Destination Port"]
                ),
                "flow_duration": float(
                    record["Flow Duration"]
                ),
                "total_packets": (
                    int(record["Total Fwd Packets"])
                    + int(record["Total Backward Packets"])
                ),
                "total_bytes": (
                    float(record["Total Length of Fwd Packets"])
                    + float(record["Total Length of Bwd Packets"])
                ),
                "prediction": prediction["prediction"],
                "attack_probability": prediction[
                    "attack_probability"
                ],
                "benign_probability": prediction[
                    "benign_probability"
                ],
                "confidence": prediction["confidence"],
                "model": prediction["model"],
                "dataset_label": (
                    "ATTACK"
                    if int(record["target"]) == 1
                    else "BENIGN"
                ),
                "source": "CIC-IDS2017 simulated sample",
            }

            self.recent_events.appendleft(event)

            return event


# Shared traffic simulation instance
traffic_service = TrafficSimulationService()
