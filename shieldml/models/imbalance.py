"""Class Imbalance Handling Strategies for Financial Fraud Detection.

Supports:
- SMOTE (Synthetic Minority Over-sampling Technique)
- ADASYN (Adaptive Synthetic Sampling)
- Random Undersampling (majority class reduction)
- Cost-Sensitive Weighting (algorithmic penalty without data modification)

CRITICAL RULE:
Resampling methods MUST be applied strictly to training partitions only.
Validation and test sets must remain untampered with their authentic class distribution.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from shieldml.config import settings
from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.models.imbalance")


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Computes inverse-frequency class weights for cost-sensitive models.

    Args:
        y: Binary target array.

    Returns:
        dict: {0: weight_0, 1: weight_1}
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Computed balanced class weights: {weight_dict}")
    return weight_dict


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Computes scale_pos_weight for XGBoost / LightGBM (neg_count / pos_count)."""
    neg_count = int(np.sum(y == 0))
    pos_count = int(np.sum(y == 1))
    scale = float(neg_count / max(1, pos_count))
    logger.info(f"Computed scale_pos_weight: {scale:.2f} (neg={neg_count}, pos={pos_count})")
    return scale


def apply_sampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "smote",
    sampling_strategy: float = 0.25,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Applies class imbalance handling strictly to training data.

    Args:
        X_train: Training feature matrix.
        y_train: Training label vector.
        method: One of ['none', 'smote', 'adasyn', 'undersample', 'cost_sensitive'].
        sampling_strategy: Target minority-to-majority ratio (e.g. 0.25).
        random_state: Seed for reproducibility.

    Returns:
        Tuple[X_resampled, y_resampled, stats_dict]
    """
    seed = random_state if random_state is not None else settings.RANDOM_STATE
    orig_neg = int(np.sum(y_train == 0))
    orig_pos = int(np.sum(y_train == 1))
    method_lower = method.lower()

    logger.info(f"Applying imbalance strategy '{method_lower}' (Original: {orig_neg} legit, {orig_pos} fraud)...")

    if method_lower in ["none", "cost_sensitive"]:
        # No resampling performed
        stats = {
            "method": method_lower,
            "original_legit": orig_neg,
            "original_fraud": orig_pos,
            "resampled_legit": orig_neg,
            "resampled_fraud": orig_pos,
            "ratio": orig_pos / max(1, orig_neg),
        }
        return X_train, y_train, stats

    try:
        if method_lower == "smote":
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=seed, k_neighbors=5)
            X_res, y_res = sampler.fit_resample(X_train, y_train)

        elif method_lower == "adasyn":
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=seed, n_neighbors=5)
            X_res, y_res = sampler.fit_resample(X_train, y_train)

        elif method_lower in ["undersample", "random_undersample"]:
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed)
            X_res, y_res = sampler.fit_resample(X_train, y_train)

        else:
            raise ValueError(f"Unknown imbalance method: {method}")

    except Exception as e:
        logger.warning(f"Resampling with {method_lower} failed ({e}); falling back to un-sampled training data.")
        return X_train, y_train, {
            "method": "fallback_none",
            "error": str(e),
            "original_legit": orig_neg,
            "original_fraud": orig_pos,
            "resampled_legit": orig_neg,
            "resampled_fraud": orig_pos,
        }

    res_neg = int(np.sum(y_res == 0))
    res_pos = int(np.sum(y_res == 1))
    stats = {
        "method": method_lower,
        "original_legit": orig_neg,
        "original_fraud": orig_pos,
        "resampled_legit": res_neg,
        "resampled_fraud": res_pos,
        "ratio": res_pos / max(1, res_neg),
    }

    logger.info(f"Resampling complete ({method_lower}): {res_neg} legit, {res_pos} fraud (ratio: {stats['ratio']:.2f})")
    return X_res, y_res, stats
