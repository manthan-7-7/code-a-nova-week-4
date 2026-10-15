"""Threshold Optimization and Metric Calculation for Fraud Detection."""

from typing import Dict, Any, Tuple, List
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)

from shieldml.config import settings
from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.models.threshold")


def optimize_decision_threshold(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    beta: float = settings.F_BETA,
    n_thresholds: int = 200,
) -> Tuple[float, Dict[str, Any]]:
    """Finds the optimal decision threshold that maximizes the F-beta (F2) score.

    In fraud detection, false negatives (missed fraud) are much more damaging than
    false positives (temporary verification friction). Maximizing F2 prioritizes recall
    while maintaining acceptable precision.

    Args:
        y_true_val: Validation true labels (binary).
        y_prob_val: Validation predicted fraud probabilities.
        beta: Weighting parameter for recall (default 2.0).
        n_thresholds: Number of threshold evaluation points.

    Returns:
        Tuple[float, dict]: (optimal_threshold, search_summary_dict)
    """
    logger.info(f"Optimizing decision threshold (beta={beta}, search_steps={n_thresholds}) on validation set...")

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_fbeta = -1.0
    best_threshold = 0.50
    best_metrics = {}

    curve_data: List[Dict[str, float]] = []

    for t in thresholds:
        preds = (y_prob_val >= t).astype(int)
        prec = float(precision_score(y_true_val, preds, zero_division=0))
        rec = float(recall_score(y_true_val, preds, zero_division=0))
        f1 = float(f1_score(y_true_val, preds, zero_division=0))
        fb = float(fbeta_score(y_true_val, preds, beta=beta, zero_division=0))

        curve_data.append({
            "threshold": round(float(t), 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "f2": round(fb, 4),
        })

        if fb > best_fbeta:
            best_fbeta = fb
            best_threshold = float(t)
            best_metrics = {
                "threshold": round(float(t), 4),
                "f2_score": round(fb, 4),
                "f1_score": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            }

    logger.info(f"Threshold Optimization Complete:")
    logger.info(f"  Optimal Threshold: {best_threshold:.4f}")
    logger.info(f"  Validation F2:     {best_metrics['f2_score']:.4f}")
    logger.info(f"  Validation Recall: {best_metrics['recall']:.4f}")
    logger.info(f"  Validation Prec:   {best_metrics['precision']:.4f}")

    summary = {
        "optimal_threshold": best_threshold,
        "best_metrics": best_metrics,
        "curve_sample": curve_data[::10],  # Subsampled for JSON serialization
    }
    return best_threshold, summary


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    beta: float = settings.F_BETA,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics at a given frozen threshold.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Continuous model predicted probabilities for positive class (fraud).
        threshold: Frozen decision threshold.
        beta: F-beta score parameter (default 2.0).

    Returns:
        dict: Full performance report including ROC-AUC, PR-AUC, F2, F1, Recall, Precision, and CM.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))
    specificity = float(tn / max(1, tn + fp))

    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f2_score": round(f2, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "total_samples": int(len(y_true)),
        },
    }
