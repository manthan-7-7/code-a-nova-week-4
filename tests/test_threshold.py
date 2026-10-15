"""Tests for F2 threshold optimization and classification metrics."""

import pytest
import numpy as np

from shieldml.models.threshold import optimize_decision_threshold, compute_classification_metrics


def test_f2_threshold_optimization():
    """Verifies that F2 threshold optimizer successfully finds an optimal threshold."""
    # Synthesize probabilities with realistic calibration
    np.random.seed(42)
    y_true = np.array([0] * 900 + [1] * 100)
    # Probabilities for class 0 centered at 0.05, class 1 centered at 0.65
    y_prob = np.concatenate([
        np.random.beta(0.5, 9.5, size=900),
        np.random.beta(6.5, 3.5, size=100),
    ])

    best_thresh, summary = optimize_decision_threshold(y_true, y_prob, beta=2.0)
    assert 0.01 <= best_thresh <= 0.99
    assert "best_metrics" in summary
    assert summary["best_metrics"]["f2_score"] > 0.0


def test_classification_metrics_computation():
    """Verifies accuracy of ROC-AUC, PR-AUC, F2, Precision, Recall, and Confusion Matrix."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f2_score"] == 1.0

    cm = metrics["confusion_matrix"]
    assert cm["tn"] == 4
    assert cm["fp"] == 0
    assert cm["fn"] == 0
    assert cm["tp"] == 4
