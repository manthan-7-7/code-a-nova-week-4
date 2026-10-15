"""Tests for FastAPI endpoints: /predict, /predict/batch, /health, /metrics, /model/info."""

import pytest
from fastapi.testclient import TestClient

from shieldml.api.app import create_app
from shieldml.api.state import app_state
from shieldml.ingestion.data_generator import generate_synthetic_transactions
from shieldml.features.engineer import FeatureEngineer
from shieldml.features.pipeline import FraudFeaturePipeline
from shieldml.models.trainers import train_lightgbm
from shieldml.explainability.shap_explainer import FraudShapExplainer


@pytest.fixture(scope="module")
def test_client():
    """Sets up test client and seeds in-memory artifacts for fast, reliable unit testing."""
    # Seed app_state with lightweight fitted components
    df = generate_synthetic_transactions(n_transactions=300, random_state=42)
    engineer = FeatureEngineer()
    engineer.fit(df)
    feat_df = engineer.transform(df, is_training=True)

    pipeline = FraudFeaturePipeline()
    pipeline.fit(feat_df)
    X = pipeline.transform(feat_df)
    y = df["is_fraud"].values.astype(int)

    model = train_lightgbm(X, y, n_estimators=30, max_depth=3)
    explainer = FraudShapExplainer(model, pipeline.get_feature_names())

    app_state.engineer = engineer
    app_state.preprocessor = pipeline
    app_state.model = model
    app_state.explainer = explainer
    app_state.threshold = 0.40
    app_state.model_version = "1.0.0-test"
    app_state.is_ready = True
    app_state.metadata = {
        "model_name": "Test-LightGBM",
        "model_type": "LightGBM",
        "training_timestamp": "2024-01-01 00:00:00",
        "validation_metrics": {"f2_score": 0.88},
        "test_metrics": {"f2_score": 0.86, "roc_auc": 0.94},
    }

    app = create_app()
    return TestClient(app)


def test_health_endpoint(test_client):
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "memory_usage_mb" in data


def test_metrics_endpoint(test_client):
    resp = test_client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "p95_latency_ms" in data
    assert "sla_compliance_rate_pct" in data


def test_model_info_endpoint(test_client):
    resp = test_client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["optimal_threshold"] == 0.40
    assert len(data["feature_names"]) > 0


def test_predict_single_endpoint(test_client):
    payload = {
        "transaction_id": "tx_unit_001",
        "timestamp": "2024-01-15 12:00:00",
        "user_id": "usr_00010",
        "amount": 350.0,
        "merchant_id": "merch_0005",
        "merchant_category": "electronics",
        "device_id": "dev_00012",
        "channel": "web",
        "distance_from_home": 45.0,
        "tx_count_user_1h": 2,
    }

    resp = test_client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["transaction_id"] == "tx_unit_001"
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["action"] in ["APPROVE", "FLAG_CHALLENGE", "BLOCK"]
    assert "latency_ms" in data
    assert "explanation" in data
    assert data["explanation"] is not None
    assert len(data["explanation"]["top_risk_drivers"]) >= 0


def test_predict_validation_error(test_client):
    # Negative amount should be rejected by Pydantic validation
    payload = {
        "transaction_id": "tx_bad_002",
        "user_id": "usr_00010",
        "amount": -50.0,  # Invalid
        "merchant_id": "merch_0005",
    }
    resp = test_client.post("/predict", json=payload)
    assert resp.status_code == 422  # Unprocessable Entity


def test_predict_batch_endpoint(test_client):
    batch_payload = {
        "transactions": [
            {
                "transaction_id": f"tx_batch_{i}",
                "user_id": f"usr_{i}",
                "amount": float(50 + i * 10),
                "merchant_id": "merch_0001",
            }
            for i in range(5)
        ]
    }
    resp = test_client.post("/predict/batch", json=batch_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_processed"] == 5
    assert len(data["predictions"]) == 5
