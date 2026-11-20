# SentinelAI

AI-Powered Intelligent Cybersecurity Threat Detection and Network Monitoring System.

## Overview

SentinelAI is a machine-learning-based cybersecurity application that analyzes network-flow records and classifies traffic as BENIGN or ATTACK using a trained Random Forest model.

The project uses the CIC-IDS2017 intrusion detection dataset for model development and evaluation. A FastAPI backend exposes the trained model through a REST API.

## Key Features

- Network traffic preprocessing and feature analysis.
- Binary classification of BENIGN and ATTACK traffic.
- Random Forest model training and evaluation.
- Duplicate fingerprint analysis to identify potential train-test leakage.
- Grouped dataset splitting to keep matching feature fingerprints together.
- Full grouped test-set evaluation.
- FastAPI prediction endpoint.
- Input validation for 70 network-flow features.
- Attack and benign probability scores.
- Swagger API documentation.

## Technology Stack

**Machine Learning**
- Python
- Pandas
- NumPy
- Scikit-learn
- PyArrow
- Matplotlib
- Seaborn
- Joblib

**Backend**
- FastAPI
- Pydantic
- Uvicorn

**Dataset**
- CIC-IDS2017

## Model Performance

The grouped Random Forest model was evaluated on 569,393 test records.

| Metric | Result |
|---|---:|
| Accuracy | 99.8446% |
| Precision | 99.6264% |
| Recall | 99.5902% |
| F1 Score | 99.6083% |
| ROC-AUC | 99.9873% |
| False Positive Rate | 0.09246% |

These results are from a grouped split designed to prevent identical feature fingerprints from crossing the training and test sets.

**Evaluation limitation:** These results are based on CIC-IDS2017. They do not establish generalization to unseen networks, future attacks, or real-world production traffic.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Backend and model health |
| POST | `/api/predict` | Classify network traffic |

Interactive API documentation:

`http://127.0.0.1:8000/docs`

## Prediction Example

Request:

```json
{
  "features": {
    "Destination Port": 54863.0,
    "Flow Duration": 3.0
  }
}