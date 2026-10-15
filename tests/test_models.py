"""Tests for model training, soft voting ensembles, and probability calibration."""

import pytest
import numpy as np

from shieldml.models.baselines import train_logistic_regression
from shieldml.models.trainers import train_random_forest, train_lightgbm
from shieldml.models.ensemble import SoftVotingEnsemble, calibrate_model_probabilities


@pytest.fixture
def dummy_train_val_data():
    np.random.seed(42)
    X_train = np.random.randn(300, 8)
    y_train = (X_train[:, 0] + X_train[:, 1] > 1.2).astype(int)

    X_val = np.random.randn(100, 8)
    y_val = (X_val[:, 0] + X_val[:, 1] > 1.2).astype(int)
    return X_train, y_train, X_val, y_val


def test_baseline_logistic_regression(dummy_train_val_data):
    X_tr, y_tr, X_val, y_val = dummy_train_val_data
    lr = train_logistic_regression(X_tr, y_tr)
    probs = lr.predict_proba(X_val)

    assert probs.shape == (100, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_soft_voting_ensemble(dummy_train_val_data):
    X_tr, y_tr, X_val, y_val = dummy_train_val_data
    rf = train_random_forest(X_tr, y_tr, n_estimators=30, max_depth=4)
    lgb = train_lightgbm(X_tr, y_tr, n_estimators=30, max_depth=3)

    ensemble = SoftVotingEnsemble(
        estimators=[("lgb", lgb), ("rf", rf)],
        weights=[0.6, 0.4],
        threshold=0.45,
    )

    probs = ensemble.predict_proba(X_val)
    preds = ensemble.predict(X_val)

    assert probs.shape == (100, 2)
    assert len(preds) == 100
    assert set(preds).issubset({0, 1})


def test_probability_calibration(dummy_train_val_data):
    X_tr, y_tr, X_val, y_val = dummy_train_val_data
    rf = train_random_forest(X_tr, y_tr, n_estimators=20, max_depth=3)
    calibrated = calibrate_model_probabilities(rf, X_val, y_val, method="sigmoid")

    probs = calibrated.predict_proba(X_val)
    assert probs.shape == (100, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
