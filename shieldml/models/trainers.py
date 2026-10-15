"""Tree-based Model Trainers for Random Forest, XGBoost, and LightGBM."""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from shieldml.config import settings
from shieldml.models.imbalance import compute_scale_pos_weight
from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.models.trainers")


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 150,
    max_depth: Optional[int] = 12,
    min_samples_split: int = 6,
    min_samples_leaf: int = 3,
    class_weight: str = "balanced",
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    **kwargs,
) -> RandomForestClassifier:
    """Trains a Random Forest classifier tuned for fraud detection.

    Args:
        X_train: Training feature array.
        y_train: Training labels.
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples required to split an internal node.
        min_samples_leaf: Minimum samples required at a leaf node.
        class_weight: Class balancing strategy ('balanced' or 'balanced_subsample').
        n_jobs: Parallel worker count.
        random_state: Seed.

    Returns:
        RandomForestClassifier: Fitted estimator.
    """
    seed = random_state if random_state is not None else settings.RANDOM_STATE
    logger.info(f"Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=seed,
        **kwargs,
    )
    rf.fit(X_train, y_train)
    logger.info("Random Forest training complete.")
    return rf


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 180,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    scale_pos_weight: Optional[float] = None,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    **kwargs,
) -> xgb.XGBClassifier:
    """Trains an XGBoost classifier with cost-sensitive scale_pos_weight.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_estimators: Number of gradient boosted rounds.
        max_depth: Max depth of tree.
        learning_rate: Shrinkage rate.
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns when constructing each tree.
        scale_pos_weight: Balance of positive and negative weights.
        n_jobs: Number of threads.
        random_state: Seed.

    Returns:
        xgb.XGBClassifier: Fitted XGBoost model.
    """
    seed = random_state if random_state is not None else settings.RANDOM_STATE
    if scale_pos_weight is None:
        scale_pos_weight = compute_scale_pos_weight(y_train)

    logger.info(f"Training XGBoost (n_estimators={n_estimators}, max_depth={max_depth}, "
                f"learning_rate={learning_rate}, scale_pos_weight={scale_pos_weight:.2f})...")

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=n_jobs,
        **kwargs,
    )
    model.fit(X_train, y_train)
    logger.info("XGBoost training complete.")
    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 180,
    max_depth: int = 6,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    scale_pos_weight: Optional[float] = None,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    **kwargs,
) -> lgb.LGBMClassifier:
    """Trains a LightGBM classifier optimized for ultra-fast inference and high recall.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_estimators: Number of boosted trees.
        max_depth: Maximum tree depth.
        num_leaves: Maximum tree leaves for base learners.
        learning_rate: Boosting learning rate.
        subsample: Subsample ratio of training data.
        colsample_bytree: Subsample ratio of columns.
        scale_pos_weight: Imbalance ratio weight.
        n_jobs: Worker count.
        random_state: Seed.

    Returns:
        lgb.LGBMClassifier: Fitted LightGBM model.
    """
    seed = random_state if random_state is not None else settings.RANDOM_STATE
    if scale_pos_weight is None:
        scale_pos_weight = compute_scale_pos_weight(y_train)

    logger.info(f"Training LightGBM (n_estimators={n_estimators}, num_leaves={num_leaves}, "
                f"learning_rate={learning_rate}, scale_pos_weight={scale_pos_weight:.2f})...")

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
        **kwargs,
    )
    model.fit(X_train, y_train)
    logger.info("LightGBM training complete.")
    return model
