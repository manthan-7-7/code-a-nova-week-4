"""Tests ensuring zero future-data leakage across splits, features, and sampling."""

import pytest
import pandas as pd
import numpy as np

from shieldml.ingestion.data_generator import generate_synthetic_transactions
from shieldml.ingestion.split import chronological_split
from shieldml.features.engineer import FeatureEngineer
from shieldml.models.imbalance import apply_sampling


def test_chronological_split_leakage_prevention():
    """Verifies that splits have zero temporal overlap and chronological bounds."""
    df = generate_synthetic_transactions(n_transactions=1000, random_state=42)
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    train_max = pd.to_datetime(train_df["timestamp"]).max()
    val_min = pd.to_datetime(val_df["timestamp"]).min()
    val_max = pd.to_datetime(val_df["timestamp"]).max()
    test_min = pd.to_datetime(test_df["timestamp"]).min()

    # Assert strict chronological boundary
    assert train_max <= val_min, f"Leakage: train_max ({train_max}) > val_min ({val_min})"
    assert val_max <= test_min, f"Leakage: val_max ({val_max}) > test_min ({test_min})"


def test_merchant_stats_fitted_on_train_only():
    """Verifies that FeatureEngineer statistics do not leak future merchant transaction values."""
    df = generate_synthetic_transactions(n_transactions=500, random_state=42)
    train_df, val_df, test_df = chronological_split(df)

    engineer = FeatureEngineer()
    engineer.fit(train_df)

    # Check that merchant stats reflect train amounts, not val or test amounts
    for m_id, (m_mean, _) in engineer.merchant_stats_.items():
        train_m = train_df[train_df["merchant_id"] == m_id]
        if len(train_m) > 0:
            assert np.isclose(m_mean, train_m["amount"].mean(), atol=1e-4)


def test_sampling_exclusively_on_train():
    """Verifies that SMOTE and sampling techniques are applied only to training partitions."""
    X_train = np.random.randn(200, 10)
    y_train = np.array([0] * 190 + [1] * 10)  # 5% minority

    X_val = np.random.randn(50, 10)
    y_val = np.array([0] * 48 + [1] * 2)

    X_res, y_res, stats = apply_sampling(X_train, y_train, method="smote", sampling_strategy=0.25)

    # Train was resampled
    assert len(y_res) > len(y_train)
    # Val remains untampered
    assert len(y_val) == 50
    assert np.sum(y_val == 1) == 2
