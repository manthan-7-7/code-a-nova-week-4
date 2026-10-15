"""Tests for data ingestion, synthetic data generation, and profiling."""

import pytest
import pandas as pd
import numpy as np

from shieldml.ingestion.data_loader import DataLoader
from shieldml.ingestion.data_generator import generate_synthetic_transactions
from shieldml.ingestion.split import chronological_split


def test_synthetic_data_generation():
    """Verifies that synthetic data generation creates valid, sorted chronological datasets."""
    df = generate_synthetic_transactions(n_transactions=500, fraud_rate=0.04, random_state=42)
    assert len(df) == 500
    assert "transaction_id" in df.columns
    assert "is_fraud" in df.columns
    assert df["is_fraud"].sum() > 0

    # Ensure chronological order
    timestamps = pd.to_datetime(df["timestamp"])
    assert timestamps.is_monotonic_increasing


def test_data_loader_validation():
    """Tests schema validation, missing column checks, and negative amount clipping."""
    loader = DataLoader(target_col="is_fraud")

    # Valid data
    df_valid = pd.DataFrame({
        "transaction_id": ["tx_1", "tx_2"],
        "timestamp": ["2024-01-01 10:00:00", "2024-01-01 11:00:00"],
        "user_id": ["u1", "u2"],
        "amount": [50.0, -10.0],  # Includes negative to test clipping
        "merchant_id": ["m1", "m2"],
        "is_fraud": [0, 1],
    })

    cleaned = loader.validate_and_clean(df_valid)
    assert len(cleaned) == 2
    assert (cleaned["amount"] >= 0).all()
    assert cleaned.loc[1, "amount"] == 0.0  # Clipped

    # Missing mandatory column
    df_invalid = df_valid.drop(columns=["merchant_id"])
    with pytest.raises(ValueError, match="Missing required columns"):
        loader.validate_and_clean(df_invalid)


def test_dataset_profiling():
    """Tests statistical profiling of transaction datasets."""
    loader = DataLoader(target_col="is_fraud")
    df = generate_synthetic_transactions(n_transactions=300, fraud_rate=0.05, random_state=42)
    profile = loader.profile_dataset(df)

    assert profile["total_records"] == 300
    assert "class_distribution" in profile
    assert profile["class_distribution"]["fraud_count"] > 0
    assert "amount_stats" in profile
    assert profile["amount_stats"]["mean"] > 0
