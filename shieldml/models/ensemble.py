"""Soft Voting Ensemble and Leakage-Free Probability Calibration."""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.models.ensemble")


class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    """Combines predictions from multiple trained models using weighted soft voting."""

    def __init__(
        self,
        estimators: List[Tuple[str, Any]],
        weights: Optional[List[float]] = None,
        threshold: float = 0.5,
    ):
        """
        Args:
            estimators: List of (name, fitted_model) tuples.
            weights: Optional weight for each estimator.
            threshold: Decision threshold for predict().
        """
        self.estimators = estimators
        if weights is None:
            self.weights = [1.0 / len(estimators)] * len(estimators)
        else:
            w_sum = sum(weights)
            self.weights = [w / w_sum for w in weights]
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Computes weighted probability across all constituent models.

        Args:
            X: Input feature matrix.

        Returns:
            np.ndarray: Array of shape (n_samples, 2) with [P(legit), P(fraud)].
        """
        accumulated_probs = np.zeros(len(X), dtype=np.float64)

        for (name, model), w in zip(self.estimators, self.weights):
            probs = model.predict_proba(X)
            # Probability of fraud (class 1)
            p1 = probs[:, 1] if probs.ndim == 2 else probs
            accumulated_probs += w * p1

        p0 = 1.0 - accumulated_probs
        return np.column_stack([p0, accumulated_probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classifies samples using the calibrated decision threshold."""
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= self.threshold).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Computes weighted composite feature importance if underlying models support it."""
        importances = []
        valid_weights = []

        for (name, model), w in zip(self.estimators, self.weights):
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
                norm_fi = fi / (np.sum(fi) + 1e-12)
                importances.append(norm_fi)
                valid_weights.append(w)

        if not importances:
            return np.array([])

        total_w = sum(valid_weights)
        combined = np.zeros_like(importances[0], dtype=np.float64)
        for fi, w in zip(importances, valid_weights):
            combined += (w / total_w) * fi
        return combined


class CalibratedModelWrapper(BaseEstimator, ClassifierMixin):
    """Applies post-hoc probability calibration (Platt scaling / Sigmoid or Isotonic).

    Compatible across all versions of scikit-learn without relying on deprecated parameters.
    """

    def __init__(self, base_model: Any, calibrator: Any, method: str = "sigmoid"):
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_probs = self.base_model.predict_proba(X)
        p1 = raw_probs[:, 1] if raw_probs.ndim == 2 else raw_probs

        if self.method == "sigmoid":
            p1_feat = p1.reshape(-1, 1)
            calibrated_p1 = self.calibrator.predict_proba(p1_feat)[:, 1]
        else:
            calibrated_p1 = self.calibrator.predict(p1)
            calibrated_p1 = np.clip(calibrated_p1, 0.0, 1.0)

        p0 = 1.0 - calibrated_p1
        return np.column_stack([p0, calibrated_p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)

    @property
    def estimators(self):
        return getattr(self.base_model, "estimators", [])

    @property
    def feature_importances_(self) -> np.ndarray:
        if hasattr(self.base_model, "feature_importances_"):
            return self.base_model.feature_importances_
        return np.array([])


def calibrate_model_probabilities(
    base_model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "sigmoid",
) -> CalibratedModelWrapper:
    """Calibrates model output probabilities strictly on validation data.

    Prevents leakage by fitting calibration strictly on the holdout validation set,
    ensuring that raw probabilities accurately reflect true empirical fraud frequencies.

    Args:
        base_model: Pre-trained classifier.
        X_val: Validation features (never seen during initial training).
        y_val: Validation labels.
        method: 'sigmoid' (Platt scaling) or 'isotonic'.

    Returns:
        CalibratedModelWrapper: Calibrated estimator ready for inference.
    """
    logger.info(f"Calibrating model probabilities using {method} on {len(X_val)} validation samples...")
    raw_probs = base_model.predict_proba(X_val)
    p1_val = raw_probs[:, 1] if raw_probs.ndim == 2 else raw_probs

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(p1_val, y_val)
    else:
        from sklearn.linear_model import LogisticRegression
        calibrator = LogisticRegression(solver="lbfgs")
        calibrator.fit(p1_val.reshape(-1, 1), y_val)

    logger.info("Probability calibration complete.")
    return CalibratedModelWrapper(base_model, calibrator, method=method)
