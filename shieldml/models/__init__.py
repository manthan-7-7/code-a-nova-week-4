"""Model training, hyperparameter optimization, ensemble, thresholding, and evaluation."""

from shieldml.models.baselines import train_logistic_regression
from shieldml.models.trainers import (
    train_random_forest,
    train_xgboost,
    train_lightgbm,
)
from shieldml.models.imbalance import apply_sampling, compute_class_weights
from shieldml.models.hyperopt import optimize_hyperparameters
from shieldml.models.ensemble import SoftVotingEnsemble, calibrate_model_probabilities
from shieldml.models.threshold import optimize_decision_threshold, compute_classification_metrics
from shieldml.models.evaluate import ComprehensiveEvaluator

__all__ = [
    "train_logistic_regression",
    "train_random_forest",
    "train_xgboost",
    "train_lightgbm",
    "apply_sampling",
    "compute_class_weights",
    "optimize_hyperparameters",
    "SoftVotingEnsemble",
    "calibrate_model_probabilities",
    "optimize_decision_threshold",
    "compute_classification_metrics",
    "ComprehensiveEvaluator",
]
