"""Baseline Model Implementation for ShieldML."""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

from shieldml.config import settings
from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.models.baselines")


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    class_weight: str = "balanced",
    max_iter: int = 1000,
    random_state: Optional[int] = None,
) -> LogisticRegression:
    """Trains a regularized Logistic Regression baseline with balanced class weights.

    Args:
        X_train: Standardized training feature array.
        y_train: Target labels.
        C: Inverse regularization strength.
        class_weight: 'balanced' or None.
        max_iter: Maximum solver iterations.
        random_state: Seed for solver.

    Returns:
        LogisticRegression: Trained linear baseline estimator.
    """
    seed = random_state if random_state is not None else settings.RANDOM_STATE
    logger.info(f"Training Logistic Regression baseline (C={C}, class_weight={class_weight})...")

    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    logger.info("Logistic Regression baseline training complete.")
    return model
