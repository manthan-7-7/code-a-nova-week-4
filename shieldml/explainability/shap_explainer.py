"""SHAP TreeExplainer Wrapper and Natural Language Attribution Generator."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import shap

from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.explainability.shap_explainer")


class FraudShapExplainer:
    """Wraps SHAP TreeExplainer for fast real-time attributions and human-readable narratives."""

    def __init__(self, model: Any, feature_names: List[str]):
        """
        Args:
            model: Tree-based estimator (LightGBM, XGBoost, or RandomForest).
            feature_names: List of column names in transformed matrix order.
        """
        self.model = model
        self.feature_names = feature_names

        # If wrapped in calibrated wrapper or ensemble, extract the primary tree model (e.g. LightGBM or XGBoost)
        underlying = model
        if hasattr(underlying, "base_model"):
            underlying = underlying.base_model
        if hasattr(underlying, "estimators"):
            for _, est in underlying.estimators:
                if hasattr(est, "tree_") or hasattr(est, "booster_") or "LGBM" in est.__class__.__name__ or "XGB" in est.__class__.__name__:
                    underlying = est
                    break

        self.underlying_model = underlying
        logger.info(f"Initializing SHAP TreeExplainer with {self.underlying_model.__class__.__name__}...")

        try:
            self.explainer = shap.TreeExplainer(self.underlying_model)
            # Base value extraction
            bv = self.explainer.expected_value
            arr = np.asarray(bv).flatten()
            if len(arr) > 1:
                self.base_value = float(arr[1])
            elif len(arr) == 1:
                self.base_value = float(arr[0])
            else:
                self.base_value = 0.0
        except Exception as e:
            logger.warning(f"TreeExplainer initialization warning: {e}; attempting fallback background summary.")
            self.explainer = shap.Explainer(self.underlying_model)
            self.base_value = 0.0

        logger.info(f"SHAP Explainer initialized (Base Value / Prior: {self.base_value:.4f})")

    def explain_single(
        self,
        features_row: np.ndarray,
        raw_inputs: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Generates local SHAP feature attributions and explanatory narratives.

        Args:
            features_row: 1D or 2D array of transformed features (1 sample).
            raw_inputs: Original raw transaction values for human context.
            top_k: Number of positive and negative drivers to highlight.

        Returns:
            dict: Structured local SHAP explanation with factors and narratives.
        """
        if features_row.ndim == 1:
            X = features_row.reshape(1, -1)
        else:
            X = features_row

        shap_vals = self.explainer.shap_values(X)

        # Handle binary classification output formats
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            # Positive class (fraud) is index 1
            vals = shap_vals[1][0]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            vals = shap_vals[0, :, 1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
            vals = shap_vals[0]
        else:
            vals = np.array(shap_vals).flatten()

        raw_in = raw_inputs or {}
        feature_attributions: List[Dict[str, Any]] = []

        for idx, (f_name, shap_val) in enumerate(zip(self.feature_names, vals)):
            val = float(shap_val)
            norm_val = float(X[0, idx])
            narrative = self._generate_narrative(f_name, val, norm_val, raw_in)

            feature_attributions.append({
                "feature": f_name,
                "shap_value": round(val, 4),
                "transformed_value": round(norm_val, 4),
                "impact": "increases_fraud_risk" if val > 0 else "decreases_fraud_risk",
                "narrative": narrative,
            })

        # Sort by absolute SHAP impact
        sorted_attributions = sorted(feature_attributions, key=lambda x: abs(x["shap_value"]), reverse=True)

        risk_drivers = [a for a in sorted_attributions if a["shap_value"] > 0][:top_k]
        trust_drivers = [a for a in sorted_attributions if a["shap_value"] < 0][:top_k]

        return {
            "base_value": round(self.base_value, 4),
            "top_risk_drivers": risk_drivers,
            "top_trust_drivers": trust_drivers,
            "all_attributions": sorted_attributions,
            "summary_text": self._create_summary_text(risk_drivers, trust_drivers),
        }

    def _generate_narrative(
        self,
        feature_name: str,
        shap_val: float,
        transformed_val: float,
        raw_inputs: Dict[str, Any],
    ) -> str:
        """Translates raw and SHAP values into plain English financial explanations."""
        if "zscore" in feature_name:
            amt = raw_inputs.get("amount", "unknown")
            return f"Amount (${amt}) deviates significantly from historical merchant baseline (SHAP: {shap_val:+.3f})"
        elif "tx_count_user_1h" in feature_name:
            c = raw_inputs.get("tx_count_user_1h", "elevated")
            return f"High 1-hour transaction velocity ({c} txs in past hour) indicates rapid burst (SHAP: {shap_val:+.3f})"
        elif "tx_count_user_24h" in feature_name:
            c = raw_inputs.get("tx_count_user_24h", "elevated")
            return f"Daily transaction frequency ({c} txs in 24h) (SHAP: {shap_val:+.3f})"
        elif "is_new_device" in feature_name:
            return f"Unrecognized device fingerprint detected for user account (SHAP: {shap_val:+.3f})"
        elif "is_night" in feature_name:
            return f"Circadian risk: Transaction initiated during late-night anomaly window (00:00 - 05:00) (SHAP: {shap_val:+.3f})"
        elif "distance" in feature_name:
            dist = raw_inputs.get("distance_from_home", "elevated")
            return f"Transaction location is {dist} km from user home baseline (SHAP: {shap_val:+.3f})"
        elif "amount_to_mean_ratio" in feature_name:
            return f"Amount spikes sharply compared to 24-hour customer expenditure average (SHAP: {shap_val:+.3f})"
        elif "device_usage_count" in feature_name:
            return f"Device frequency in system: {raw_inputs.get('device_usage_count', 'observed')} (SHAP: {shap_val:+.3f})"
        else:
            direction = "elevates fraud probability" if shap_val > 0 else "supports transaction legitimacy"
            return f"{feature_name} ({direction}, SHAP: {shap_val:+.3f})"

    def _create_summary_text(
        self, risk_drivers: List[Dict[str, Any]], trust_drivers: List[Dict[str, Any]]
    ) -> str:
        """Creates high-level executive summary sentence."""
        if not risk_drivers:
            return "Transaction exhibits standard customer behavioral profiles with no anomalous risk indicators."

        top_risks = [r["feature"] for r in risk_drivers[:2]]
        return f"Primary fraud risk factors: {', '.join(top_risks)}. Review flagged features before authorization."

    def get_global_importance(self, X_val: np.ndarray, top_n: int = 15) -> List[Dict[str, Any]]:
        """Calculates global mean absolute SHAP values across a validation sample."""
        sample_size = min(len(X_val), 500)
        X_sample = X_val[:sample_size]

        shap_vals = self.explainer.shap_values(X_sample)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            vals = shap_vals[1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            vals = shap_vals[:, :, 1]
        else:
            vals = shap_vals

        mean_abs = np.mean(np.abs(vals), axis=0)
        importance_list = [
            {"feature": f_name, "mean_abs_shap": round(float(imp), 4)}
            for f_name, imp in zip(self.feature_names, mean_abs)
        ]
        return sorted(importance_list, key=lambda x: x["mean_abs_shap"], reverse=True)[:top_n]
