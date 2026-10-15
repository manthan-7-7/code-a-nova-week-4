"""Visualization generator for evaluation curves, confusion matrices, and SHAP charts."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

from shieldml.config import settings
from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.explainability.visualizer")

# Clean, professional fintech visual styling (Emerald/Mint & crisp dark accents)
PALETTE = {
    "primary": "#10B981",    # Emerald
    "secondary": "#059669",  # Deep emerald
    "danger": "#EF4444",     # Red
    "warning": "#F59E0B",    # Amber
    "text": "#1E293B",       # Slate 800
    "bg": "#F8FAFC",         # Slate 50
    "grid": "#E2E8F0",       # Slate 200
}


def plot_roc_curve(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Path,
) -> None:
    """Plots and saves multi-model ROC Curves."""
    plt.figure(figsize=(8, 6), facecolor="white")
    ax = plt.subplot(111)
    ax.set_facecolor(PALETTE["bg"])

    colors = ["#10B981", "#3B82F6", "#8B5CF6", "#F59E0B", "#EC4899"]
    for i, (name, model) in enumerate(models_dict.items()):
        probs = model.predict_proba(X_test)
        y_prob = probs[:, 1] if probs.ndim == 2 else probs
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, y_prob)
        c = colors[i % len(colors)]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=c, lw=2.2)

    plt.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", lw=1.5, label="Random Baseline")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold", color=PALETTE["text"])
    plt.ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold", color=PALETTE["text"])
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=PALETTE["grid"])
    plt.grid(True, linestyle=":", alpha=0.6, color=PALETTE["grid"])
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curve to {save_path}")


def plot_precision_recall_curve(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Path,
) -> None:
    """Plots and saves Precision-Recall Curves (crucial for imbalanced fraud)."""
    plt.figure(figsize=(8, 6), facecolor="white")
    ax = plt.subplot(111)
    ax.set_facecolor(PALETTE["bg"])

    colors = ["#10B981", "#3B82F6", "#8B5CF6", "#F59E0B", "#EC4899"]
    for i, (name, model) in enumerate(models_dict.items()):
        probs = model.predict_proba(X_test)
        y_prob = probs[:, 1] if probs.ndim == 2 else probs
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        from sklearn.metrics import average_precision_score
        pr_auc = average_precision_score(y_test, y_prob)
        c = colors[i % len(colors)]
        plt.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.4f})", color=c, lw=2.2)

    baseline = np.sum(y_test == 1) / len(y_test)
    plt.axhline(y=baseline, color="#94A3B8", linestyle="--", label=f"No-Skill Baseline ({baseline:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Detection Rate)", fontsize=11, fontweight="bold", color=PALETTE["text"])
    plt.ylabel("Precision", fontsize=11, fontweight="bold", color=PALETTE["text"])
    plt.title("Precision-Recall (PR) Curve Comparison", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=PALETTE["grid"])
    plt.grid(True, linestyle=":", alpha=0.6, color=PALETTE["grid"])
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved PR curve to {save_path}")


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    save_path: Path,
    model_name: str = "Champion Model",
) -> None:
    """Generates an annotated confusion matrix heatmap."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    plt.figure(figsize=(7, 6), facecolor="white")
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Greens",
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
        xticklabels=["Legitimate (0)", "Fraud (1)"],
        yticklabels=["Legitimate (0)", "Fraud (1)"],
    )
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold", color=PALETTE["text"])
    plt.ylabel("Actual True Label", fontsize=11, fontweight="bold", color=PALETTE["text"])
    plt.title(f"Confusion Matrix: {model_name} (Threshold = {threshold:.3f})", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")


def plot_threshold_optimization_curve(
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
    optimal_threshold: float,
    save_path: Path,
    beta: float = 2.0,
) -> None:
    """Plots F2, Recall, Precision, and F1 across candidate decision thresholds."""
    from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

    thresholds = np.linspace(0.01, 0.99, 150)
    f2_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []

    for t in thresholds:
        preds = (y_prob_val >= t).astype(int)
        f2_scores.append(fbeta_score(y_val, preds, beta=beta, zero_division=0))
        prec_scores.append(precision_score(y_val, preds, zero_division=0))
        rec_scores.append(recall_score(y_val, preds, zero_division=0))
        f1_scores.append(f1_score(y_val, preds, zero_division=0))

    plt.figure(figsize=(9, 6), facecolor="white")
    ax = plt.subplot(111)
    ax.set_facecolor(PALETTE["bg"])

    plt.plot(thresholds, f2_scores, label=f"F2-Score (β={beta:.0f})", color="#10B981", lw=2.6)
    plt.plot(thresholds, rec_scores, label="Recall", color="#3B82F6", lw=1.8, linestyle="--")
    plt.plot(thresholds, prec_scores, label="Precision", color="#F59E0B", lw=1.8, linestyle=":")
    plt.plot(thresholds, f1_scores, label="F1-Score", color="#8B5CF6", lw=1.6, linestyle="-.")

    # Highlight optimal threshold
    plt.axvline(
        x=optimal_threshold,
        color="#EF4444",
        lw=2.0,
        linestyle="-",
        label=f"Optimal Threshold ({optimal_threshold:.3f})",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("Decision Probability Threshold", fontsize=11, fontweight="bold", color=PALETTE["text"])
    plt.ylabel("Validation Metric Value", fontsize=11, fontweight="bold", color=PALETTE["text"])
    plt.title("Threshold Optimization: Maximizing F2-Score on Holdout Validation Data", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower left", frameon=True, facecolor="white", edgecolor=PALETTE["grid"])
    plt.grid(True, linestyle=":", alpha=0.6, color=PALETTE["grid"])
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved threshold optimization plot to {save_path}")


def plot_shap_global_importance(
    global_importance: List[Dict[str, Any]],
    save_path: Path,
    top_n: int = 12,
) -> None:
    """Plots horizontal bar chart of top global SHAP feature importances."""
    features = [item["feature"] for item in global_importance[:top_n]][::-1]
    values = [item["mean_abs_shap"] for item in global_importance[:top_n]][::-1]

    plt.figure(figsize=(9, 7), facecolor="white")
    ax = plt.subplot(111)
    ax.set_facecolor(PALETTE["bg"])

    bars = plt.barh(features, values, color="#10B981", edgecolor="#059669", height=0.65)
    plt.xlabel("Mean Absolute SHAP Value (Impact on Model Fraud Probability)", fontsize=11, fontweight="bold", color=PALETTE["text"])
    plt.title("Global Feature Importance (SHAP TreeExplainer)", fontsize=13, fontweight="bold", pad=15)
    plt.grid(True, axis="x", linestyle=":", alpha=0.6, color=PALETTE["grid"])

    # Annotate bar values
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", ha="left", fontsize=9, color=PALETTE["text"])

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved global SHAP importance plot to {save_path}")


def plot_local_waterfall(
    explanation: Dict[str, Any],
    tx_title: str,
    save_path: Path,
) -> None:
    """Generates local waterfall attribution chart showing positive (risk) and negative (trust) factors."""
    drivers = explanation["top_risk_drivers"] + explanation["top_trust_drivers"]
    if not drivers:
        return

    # Sort descending by signed SHAP value
    drivers = sorted(drivers, key=lambda x: x["shap_value"])
    names = [d["feature"] for d in drivers]
    values = [d["shap_value"] for d in drivers]
    colors = ["#EF4444" if v > 0 else "#10B981" for v in values]

    plt.figure(figsize=(9, 6), facecolor="white")
    ax = plt.subplot(111)
    ax.set_facecolor(PALETTE["bg"])

    bars = plt.barh(names, values, color=colors, height=0.6)
    plt.axvline(x=0, color="#64748B", linestyle="-", lw=1.2)
    plt.xlabel("SHAP Value (Contribution to Fraud Likelihood)", fontsize=11, fontweight="bold", color=PALETTE["text"])
    plt.title(f"Local SHAP Explanation: {tx_title}", fontsize=13, fontweight="bold", pad=15)
    plt.grid(True, axis="x", linestyle=":", alpha=0.6, color=PALETTE["grid"])

    for bar, val in zip(bars, values):
        offset = 0.005 if val >= 0 else -0.005
        align = "left" if val >= 0 else "right"
        plt.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}", va="center", ha=align, fontsize=9, fontweight="bold")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved local SHAP chart to {save_path}")


def generate_all_evaluation_plots(
    models_dict: Dict[str, Any],
    champion_model: Any,
    explainer: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    optimal_threshold: float,
    output_dir: Path,
) -> None:
    """Generates and persists the complete suite of performance and explainability plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating full suite of visual evaluation artifacts...")

    # 1. ROC Curves
    plot_roc_curve(models_dict, X_test, y_test, output_dir / "roc_curve.png")

    # 2. Precision-Recall Curves
    plot_precision_recall_curve(models_dict, X_test, y_test, output_dir / "precision_recall_curve.png")

    # 3. Threshold Optimization Curve
    val_probs = champion_model.predict_proba(X_val)
    y_val_prob = val_probs[:, 1] if val_probs.ndim == 2 else val_probs
    plot_threshold_optimization_curve(
        y_val, y_val_prob, optimal_threshold, output_dir / "f2_threshold_optimization_curve.png"
    )

    # 4. Confusion Matrix on Test Set
    test_probs = champion_model.predict_proba(X_test)
    y_test_prob = test_probs[:, 1] if test_probs.ndim == 2 else test_probs
    plot_confusion_matrix(
        y_test, y_test_prob, optimal_threshold, output_dir / "confusion_matrix.png", model_name="Champion Model"
    )

    # 5. Global SHAP Feature Importance
    if explainer is not None:
        global_imp = explainer.get_global_importance(X_val)
        plot_shap_global_importance(global_imp, output_dir / "shap_summary_bar.png")

        # 6. Three Local Waterfall Charts (Fraud, Legit, Borderline)
        fraud_idx = np.where(y_test == 1)[0]
        legit_idx = np.where(y_test == 0)[0]

        if len(fraud_idx) > 0:
            tx_fraud_idx = fraud_idx[0]
            exp_fraud = explainer.explain_single(X_test[tx_fraud_idx])
            plot_local_waterfall(
                exp_fraud,
                f"Transaction #{tx_fraud_idx} (Confirmed Fraud)",
                output_dir / "shap_waterfall_tx1_fraud.png",
            )

        if len(legit_idx) > 0:
            tx_legit_idx = legit_idx[0]
            exp_legit = explainer.explain_single(X_test[tx_legit_idx])
            plot_local_waterfall(
                exp_legit,
                f"Transaction #{tx_legit_idx} (Legitimate Standard)",
                output_dir / "shap_waterfall_tx2_legit.png",
            )

        # Borderline (prob closest to threshold)
        diffs = np.abs(y_test_prob - optimal_threshold)
        borderline_idx = int(np.argmin(diffs))
        exp_borderline = explainer.explain_single(X_test[borderline_idx])
        plot_local_waterfall(
            exp_borderline,
            f"Transaction #{borderline_idx} (Borderline Assessment)",
            output_dir / "shap_waterfall_tx3_borderline.png",
        )

    logger.info("All evaluation plots generated successfully.")
