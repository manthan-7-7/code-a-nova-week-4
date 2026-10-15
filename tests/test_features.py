"""Tests for feature engineering, velocities, cyclics, and preprocessor pipeline."""

import pytest
import numpy as np
import pandas as pd

from shieldml.ingestion.data_generator import generate_synthetic_transactions
from shieldml.features.engineer import FeatureEngineer
from shieldml.features.pipeline import FraudFeaturePipeline


@pytest.fixture
def sample_tx_df():
    return generate_synthetic_transactions(n_transactions=600, fraud_rate=0.03, random_state=42)


def test_feature_engineer_transformation(sample_tx_df):
    """Verifies that FeatureEngineer computes velocities, cyclics, and z-scores."""
    engineer = FeatureEngineer()
    engineer.fit(sample_tx_df)

    feat_df = engineer.transform(sample_tx_df, is_training=True)

    # Check cyclics
    assert "hour_sin" in feat_df.columns
    assert "hour_cos" in feat_df.columns
    # Check trigonometry identity: sin^2 + cos^2 = 1
    sin_sq_cos_sq = feat_df["hour_sin"]**2 + feat_df["hour_cos"]**2
    assert np.allclose(sin_sq_cos_sq, 1.0, atol=1e-4)

    # Check merchant zscore
    assert "merchant_amount_zscore" in feat_df.columns
    assert not feat_df["merchant_amount_zscore"].isna().any()

    # Check velocity columns
    for window in ["1h", "24h", "7d"]:
        assert f"tx_count_user_{window}" in feat_df.columns
        assert f"tx_amount_sum_user_{window}" in feat_df.columns
        assert (feat_df[f"tx_count_user_{window}"] >= 0).all()


def test_transform_single():
    """Verifies ultra-fast single transaction feature transformation for real-time inference."""
    engineer = FeatureEngineer()
    sample_df = generate_synthetic_transactions(n_transactions=200, random_state=42)
    engineer.fit(sample_df)

    single_tx = {
        "transaction_id": "tx_test_101",
        "timestamp": "2024-01-10 15:30:00",
        "user_id": "usr_00001",
        "amount": 150.0,
        "merchant_id": "merch_0001",
        "merchant_category": "retail",
        "device_id": "dev_00001",
        "channel": "mobile_app",
        "distance_from_home": 5.2,
    }

    out = engineer.transform_single(single_tx)
    assert isinstance(out, dict)
    assert "log_amount" in out
    assert "hour_sin" in out
    assert "merchant_amount_zscore" in out
    assert "tx_count_user_1h" in out


def test_fraud_feature_pipeline(sample_tx_df):
    """Verifies FraudFeaturePipeline produces standard numpy matrices with clean feature names."""
    engineer = FeatureEngineer()
    engineer.fit(sample_tx_df)
    feat_df = engineer.transform(sample_tx_df, is_training=True)

    pipeline = FraudFeaturePipeline()
    pipeline.fit(feat_df)

    X_mat = pipeline.transform(feat_df)
    assert isinstance(X_mat, np.ndarray)
    assert X_mat.shape[0] == len(sample_tx_df)
    assert X_mat.shape[1] == len(pipeline.get_feature_names())
    assert not np.isnan(X_mat).any()
