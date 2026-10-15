"""Comprehensive Model Evaluation and Error Slice Analysis for ShieldML."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from shieldml.config import settings
from shieldml.models.threshold import compute_classification_metrics
from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.models.evaluate")


class ComprehensiveEvaluator:
    """Evaluates candidate models across metrics, probability curves, and subpopulation slices."""

    def __init__(self, target_metric: str = "f2_score"):
        self.target_metric = target_metric

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """Calculates full metrics, ROC curve, and PR curve on a test partition.

        Args:
            model: Trained classifier with predict_proba().
            X_test: Test features.
            y_test: Test ground truth labels.
            threshold: Frozen decision threshold.
            model_name: Name for reporting.

        Returns:
            dict: Comprehensive evaluation dictionary.
        """
        probs = model.predict_proba(X_test)
        y_prob = probs[:, 1] if probs.ndim == 2 else probs

        metrics = compute_classification_metrics(y_test, y_prob, threshold=threshold)
        metrics["model_name"] = model_name

        # Compute ROC curve coordinates (subsampled for compactness)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        step = max(1, len(fpr) // 50)
        roc_data = [
            {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
            for f, t in zip(fpr[::step], tpr[::step])
        ]

        # Compute PR curve coordinates
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_step = max(1, len(prec) // 50)
        pr_data = [
            {"recall": round(float(r), 4), "precision": round(float(p), 4)}
            for r, p in zip(rec[::pr_step], prec[::pr_step])
        ]

        return {
            "metrics": metrics,
            "roc_curve": roc_data,
            "pr_curve": pr_data,
        }

    def compare_models(
        self,
        models_dict: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold_dict: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Compares multiple candidate models and ranks them by target metric (F2).

        Args:
            models_dict: Mapping of model_name -> fitted model.
            X_test: Test feature matrix.
            y_test: Test labels.
            threshold_dict: Mapping of model_name -> frozen threshold (or 0.5 default).

        Returns:
            pd.DataFrame: Comparative scorecard table.
        """
        records = []
        for name, model in models_dict.items():
            thresh = threshold_dict.get(name, 0.5) if threshold_dict else 0.5
            eval_res = self.evaluate_model(model, X_test, y_test, threshold=thresh, model_name=name)
            m = eval_res["metrics"]
            cm = m["confusion_matrix"]

            records.append({
                "Model": name,
                "Threshold": m["threshold"],
                "ROC-AUC": m["roc_auc"],
                "PR-AUC": m["pr_auc"],
                "F2-Score": m["f2_score"],
                "F1-Score": m["f1_score"],
                "Recall": m["recall"],
                "Precision": m["precision"],
                "True Negatives": cm["tn"],
                "False Positives": cm["fp"],
                "False Negatives": cm["fn"],
                "True Positives": cm["tp"],
            })

        df_compare = pd.DataFrame(records).sort_values(by="F2-Score", ascending=False).reset_index(drop=True)
        logger.info("\n=== MODEL COMPARISON LEADERBOARD (Untouched Test Set) ===\n" + df_compare.to_string(index=False))
        return df_compare

    def slice_based_error_analysis(
        self,
        model: Any,
        raw_test_df: pd.DataFrame,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Evaluates model performance across key business subpopulations to detect blindspots.

        Slices:
        1. Amount Tiers: Micro (<$25), Small ($25-$100), Medium ($100-$500), Large (>$500)
        2. Time of Day: Night (0-5 AM) vs Day (6 AM - 11 PM)
        3. Merchant Category: Category slices if available.

        Returns:
            dict: Breakdown of recall, precision, and fraud counts per slice.
        """
        probs = model.predict_proba(X_test)
        y_prob = probs[:, 1] if probs.ndim == 2 else probs
        y_pred = (y_prob >= threshold).astype(int)

        df = raw_test_df.copy().reset_index(drop=True)
        df["y_true"] = y_test
        df["y_prob"] = y_prob
        df["y_pred"] = y_pred

        # 1. Amount Tiers
        bins = [-1, 25, 100, 500, float("inf")]
        labels = ["Micro (<$25)", "Small ($25-$100)", "Medium ($100-$500)", "High-Ticket (>$500)"]
        df["amount_tier"] = pd.cut(df["amount"], bins=bins, labels=labels)

        amount_slices = {}
        for tier in labels:
            sub = df[df["amount_tier"] == tier]
            if len(sub) == 0:
                continue
            rec = float(np.sum((sub["y_true"] == 1) & (sub["y_pred"] == 1)) / max(1, np.sum(sub["y_true"] == 1)))
            prec = float(np.sum((sub["y_true"] == 1) & (sub["y_pred"] == 1)) / max(1, np.sum(sub["y_pred"] == 1)))
            amount_slices[tier] = {
                "total_txs": len(sub),
                "actual_frauds": int(np.sum(sub["y_true"] == 1)),
                "detected_frauds": int(np.sum((sub["y_true"] == 1) & (sub["y_pred"] == 1))),
                "recall": round(rec, 4),
                "precision": round(prec, 4),
            }

        # 2. Night vs Day Slice
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["is_night"] = ((df["timestamp"].dt.hour >= 0) & (df["timestamp"].dt.hour <= 5)).astype(int)
        time_slices = {}
        for is_n, lbl in [(1, "Night (00:00 - 05:59)"), (0, "Day (06:00 - 23:59)")]:
            sub = df[df["is_night"] == is_n]
            if len(sub) == 0:
                continue
            rec = float(np.sum((sub["y_true"] == 1) & (sub["y_pred"] == 1)) / max(1, np.sum(sub["y_true"] == 1)))
            prec = float(np.sum((sub["y_true"] == 1) & (sub["y_pred"] == 1)) / max(1, np.sum(sub["y_pred"] == 1)))
            time_slices[lbl] = {
                "total_txs": len(sub),
                "actual_frauds": int(np.sum(sub["y_true"] == 1)),
                "detected_frauds": int(np.sum((sub["y_true"] == 1) & (sub["y_pred"] == 1))),
                "recall": round(rec, 4),
                "precision": round(prec, 4),
            }

        return {
            "amount_tier_slices": amount_slices,
            "time_of_day_slices": time_slices,
        }
