"""Tests for SHAP explainability, local waterfall formatting, and global feature importance."""

import pytest
import numpy as np

from shieldml.models.trainers import train_lightgbm
from shieldml.explainability.shap_explainer import FraudShapExplainer


@pytest.fixture
def fitted_tree_and_names():
    np.random.seed(42)
    feature_names = [f"feat_{i}" for i in range(6)]
    X = np.random.randn(200, 6)
    y = (X[:, 0] * 2.0 + X[:, 1] > 1.0).astype(int)

    model = train_lightgbm(X, y, n_estimators=25, max_depth=3)
    return model, feature_names, X


def test_shap_explainer_local(fitted_tree_and_names):
    model, feature_names, X = fitted_tree_and_names
    explainer = FraudShapExplainer(model, feature_names)

    single_row = X[0]
    explanation = explainer.explain_single(single_row, raw_inputs={"amount": 120.0}, top_k=3)

    assert "base_value" in explanation
    assert "top_risk_drivers" in explanation
    assert "top_trust_drivers" in explanation
    assert "summary_text" in explanation
    assert len(explanation["all_attributions"]) == len(feature_names)


def test_shap_global_importance(fitted_tree_and_names):
    model, feature_names, X = fitted_tree_and_names
    explainer = FraudShapExplainer(model, feature_names)

    global_imp = explainer.get_global_importance(X, top_n=5)
    assert len(global_imp) == 5
    assert "feature" in global_imp[0]
    assert "mean_abs_shap" in global_imp[0]
    # Sorted descending
    assert global_imp[0]["mean_abs_shap"] >= global_imp[1]["mean_abs_shap"]
