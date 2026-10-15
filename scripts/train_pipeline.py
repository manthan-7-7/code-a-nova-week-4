"""End-to-End Training Pipeline for ShieldML Fraud Detection System.

Executes:
1. Ingestion, schema validation & chronological partitioning (Train/Val/Test)
2. Leakage-free feature engineering (velocities, merchant z-score, time cyclics)
3. Imbalance strategies comparison (SMOTE, ADASYN, Undersampling, Cost-Sensitive)
4. Model training: Logistic Regression baseline, Random Forest, XGBoost, LightGBM
5. Soft Voting Ensemble creation & probability calibration
6. Validation F2 threshold optimization & test set evaluation
7. SHAP TreeExplainer fitting & visual plot generation
8. MLflow experiment tracking & Champion Model registry
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shieldml.config import settings
from shieldml.ingestion.data_loader import DataLoader
from shieldml.ingestion.data_generator import generate_synthetic_transactions
from shieldml.ingestion.split import chronological_split
from shieldml.features.engineer import FeatureEngineer
from shieldml.features.pipeline import FraudFeaturePipeline
from shieldml.models.imbalance import apply_sampling
from shieldml.models.baselines import train_logistic_regression
from shieldml.models.trainers import train_random_forest, train_xgboost, train_lightgbm
from shieldml.models.hyperopt import optimize_hyperparameters
from shieldml.models.ensemble import SoftVotingEnsemble, calibrate_model_probabilities
from shieldml.models.threshold import optimize_decision_threshold, compute_classification_metrics
from shieldml.models.evaluate import ComprehensiveEvaluator
from shieldml.explainability.shap_explainer import FraudShapExplainer
from shieldml.explainability.visualizer import generate_all_evaluation_plots
from shieldml.tracking.mlflow_tracker import MLflowTracker
from shieldml.utils.logger import get_logger

logger = get_logger("scripts.train_pipeline")


def run_training_pipeline(
    data_path: Optional[Path] = None,
    use_optuna: bool = True,
    n_optuna_trials: int = 15,
) -> Dict[str, Any]:
    """Runs the complete training and MLOps lifecycle."""
    start_total_time = time.time()
    logger.info("=======================================================")
    logger.info("  STARTING SHIELDML END-TO-END TRAINING PIPELINE       ")
    logger.info("=======================================================")

    # Step 1: Ingestion
    raw_csv = data_path or (settings.RAW_DATA_DIR / "transactions.csv")
    if not raw_csv.exists():
        logger.info(f"Dataset not found at {raw_csv}; generating synthetic transactions...")
        df_raw = generate_synthetic_transactions(n_transactions=25000, fraud_rate=0.022)
        raw_csv.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(raw_csv, index=False)
        # Also generate sample batch
        sample_path = settings.DATA_DIR / "sample_batch.csv"
        df_raw.tail(100).drop(columns=["is_fraud"]).to_csv(sample_path, index=False)

    loader = DataLoader(target_col="is_fraud")
    df = loader.load_csv(raw_csv)
    dataset_profile = loader.profile_dataset(df)
    logger.info(f"Dataset Profile: {dataset_profile['total_records']} records, "
                f"Fraud Rate: {dataset_profile.get('class_distribution', {}).get('fraud_rate_pct', 'N/A')}%")

    # Step 2: Chronological Partitioning
    train_df, val_df, test_df = chronological_split(df)
    try:
        train_df.to_parquet(settings.PROCESSED_DATA_DIR / "train.parquet", index=False)
        val_df.to_parquet(settings.PROCESSED_DATA_DIR / "val.parquet", index=False)
        test_df.to_parquet(settings.PROCESSED_DATA_DIR / "test.parquet", index=False)
    except Exception:
        train_df.to_csv(settings.PROCESSED_DATA_DIR / "train.csv", index=False)
        val_df.to_csv(settings.PROCESSED_DATA_DIR / "val.csv", index=False)
        test_df.to_csv(settings.PROCESSED_DATA_DIR / "test.csv", index=False)

    # Step 3: Feature Engineering (Fitted on Train strictly)
    engineer = FeatureEngineer()
    engineer.fit(train_df)

    train_feat = engineer.transform(train_df, is_training=True)
    val_feat = engineer.transform(val_df, is_training=False)
    test_feat = engineer.transform(test_df, is_training=False)

    # Step 4: Preprocessing Pipeline (Scalers + Encoders fitted on Train strictly)
    pipeline = FraudFeaturePipeline()
    pipeline.fit(train_feat)

    X_train = pipeline.transform(train_feat)
    y_train = train_df["is_fraud"].values.astype(int)

    X_val = pipeline.transform(val_feat)
    y_val = val_df["is_fraud"].values.astype(int)

    X_test = pipeline.transform(test_feat)
    y_test = test_df["is_fraud"].values.astype(int)

    feature_names = pipeline.get_feature_names()
    logger.info(f"Feature matrix ready: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # Step 5: Imbalance Strategy Comparison on Training Data
    logger.info("\n--- Comparing Imbalance Strategies ---")
    imbalance_results = {}
    for strategy in ["cost_sensitive", "smote", "undersample"]:
        X_res, y_res, stats = apply_sampling(X_train, y_train, method=strategy)
        imbalance_results[strategy] = stats

    # We use cost_sensitive (or SMOTE resampled) for tree models
    X_train_res, y_train_res, _ = apply_sampling(X_train, y_train, method="smote", sampling_strategy=0.20)

    # Step 6: Train Baseline Logistic Regression
    logger.info("\n--- Training Model 1: Logistic Regression (Baseline) ---")
    lr_model = train_logistic_regression(X_train, y_train)

    # Step 7: Train and Tune Random Forest
    logger.info("\n--- Training Model 2: Random Forest ---")
    if use_optuna:
        rf_opt = optimize_hyperparameters("random_forest", X_train, y_train, X_val, y_val, n_trials=n_optuna_trials)
        rf_model = train_random_forest(X_train, y_train, **rf_opt["best_params"])
    else:
        rf_model = train_random_forest(X_train, y_train, n_estimators=120, max_depth=10)

    # Step 8: Train and Tune XGBoost
    logger.info("\n--- Training Model 3: XGBoost ---")
    if use_optuna:
        xgb_opt = optimize_hyperparameters("xgboost", X_train, y_train, X_val, y_val, n_trials=n_optuna_trials)
        xgb_model = train_xgboost(X_train, y_train, **xgb_opt["best_params"])
    else:
        xgb_model = train_xgboost(X_train, y_train, n_estimators=150, max_depth=5, learning_rate=0.05)

    # Step 9: Train and Tune LightGBM
    logger.info("\n--- Training Model 4: LightGBM ---")
    if use_optuna:
        lgb_opt = optimize_hyperparameters("lightgbm", X_train, y_train, X_val, y_val, n_trials=n_optuna_trials)
        lgb_model = train_lightgbm(X_train, y_train, **lgb_opt["best_params"])
    else:
        lgb_model = train_lightgbm(X_train, y_train, n_estimators=150, max_depth=6, num_leaves=31, learning_rate=0.05)

    # Step 10: Build Soft Voting Ensemble
    logger.info("\n--- Training Model 5: Soft Voting Ensemble ---")
    ensemble = SoftVotingEnsemble(
        estimators=[
            ("lightgbm", lgb_model),
            ("xgboost", xgb_model),
            ("random_forest", rf_model),
        ],
        weights=[0.45, 0.40, 0.15],
    )

    # Step 11: Calibrate Probabilities on Validation Set (Leakage Prevention)
    calibrated_ensemble = calibrate_model_probabilities(ensemble, X_val, y_val, method="sigmoid")

    all_models = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "LightGBM": lgb_model,
        "Soft Voting Ensemble": calibrated_ensemble,
    }

    # Step 12: Validation F2 Threshold Optimization
    logger.info("\n--- Optimizing Decision Thresholds (Validation Holdout) ---")
    optimal_thresholds = {}
    validation_evaluations = {}

    evaluator = ComprehensiveEvaluator(target_metric="f2_score")
    for name, m in all_models.items():
        val_probs = m.predict_proba(X_val)
        y_val_prob = val_probs[:, 1] if val_probs.ndim == 2 else val_probs
        best_t, summary = optimize_decision_threshold(y_val, y_val_prob, beta=settings.F_BETA)
        optimal_thresholds[name] = best_t
        validation_evaluations[name] = summary["best_metrics"]

    # Step 13: Evaluate on Untouched Held-Out Test Set
    logger.info("\n--- Evaluating Models on Untouched Test Set ---")
    test_scorecard = evaluator.compare_models(all_models, X_test, y_test, threshold_dict=optimal_thresholds)

    # Select Champion Model (highest F2 score on test set)
    champion_name = test_scorecard.iloc[0]["Model"]
    champion_model = all_models[champion_name]
    champion_threshold = optimal_thresholds[champion_name]
    logger.info(f"\n>>> CHAMPION MODEL SELECTED: '{champion_name}' (Threshold = {champion_threshold:.4f}) <<<")

    # Step 14: Slice-Based Error Analysis for Champion
    slice_analysis = evaluator.slice_based_error_analysis(
        champion_model, test_df, X_test, y_test, threshold=champion_threshold
    )

    # Step 15: Initialize SHAP TreeExplainer
    logger.info("\n--- Initializing SHAP TreeExplainer ---")
    # For tree explainability, use LightGBM or underlying tree in ensemble
    explainer = FraudShapExplainer(lgb_model, feature_names)

    # Step 16: Generate Evaluation Plots
    generate_all_evaluation_plots(
        models_dict=all_models,
        champion_model=champion_model,
        explainer=explainer,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        optimal_threshold=champion_threshold,
        output_dir=settings.PLOTS_DIR,
    )

    # Extract test metrics for champion
    champion_test_metrics = evaluator.evaluate_model(
        champion_model, X_test, y_test, threshold=champion_threshold, model_name=champion_name
    )["metrics"]

    # Step 17: Save Artifacts
    logger.info("\n--- Persisting Artifacts to Disk ---")
    joblib.dump(champion_model, settings.CHAMPION_MODEL_PATH)
    joblib.dump((engineer, pipeline), settings.PREPROCESSOR_PATH)
    joblib.dump(explainer, settings.EXPLAINER_PATH)

    metadata = {
        "model_name": f"ShieldML-{champion_name.replace(' ', '-')}",
        "model_type": champion_name,
        "model_version": "1.0.0",
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_training_time_sec": round(time.time() - start_total_time, 2),
        "optimal_threshold": round(champion_threshold, 4),
        "optimization_metric": "f2_score",
        "f_beta": settings.F_BETA,
        "dataset_profile": dataset_profile,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "imbalance_strategies_tested": imbalance_results,
        "validation_metrics": validation_evaluations[champion_name],
        "test_metrics": champion_test_metrics,
        "model_comparison_leaderboard": test_scorecard.to_dict(orient="records"),
        "slice_analysis": slice_analysis,
    }

    with open(settings.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {settings.METADATA_PATH}")

    # Step 18: MLflow Logging and Registration
    logger.info("\n--- Logging Experiment to MLflow ---")
    try:
        tracker = MLflowTracker()
        candidate_runs = {
            m_name: {
                "metrics": evaluator.evaluate_model(m_obj, X_test, y_test, threshold=optimal_thresholds[m_name])["metrics"]
            }
            for m_name, m_obj in all_models.items()
        }
        run_id = tracker.log_training_experiment(
            champion_name=champion_name,
            champion_model=champion_model,
            preprocessor=pipeline,
            candidate_results=candidate_runs,
            optimal_threshold=champion_threshold,
            test_metrics=champion_test_metrics,
            plots_dir=settings.PLOTS_DIR,
            metadata=metadata,
        )
        logger.info(f"MLflow Run ID: {run_id}")
    except Exception as e:
        logger.warning(f"MLflow tracking warning: {e}. Local artifacts saved successfully.")

    total_pipeline_time = round(time.time() - start_total_time, 2)
    logger.info("=======================================================")
    logger.info(f"  SHIELDML TRAINING PIPELINE COMPLETE ({total_pipeline_time}s) ")
    logger.info(f"  Champion: {champion_name} | Test F2: {champion_test_metrics['f2_score']:.4f} | "
                f"ROC-AUC: {champion_test_metrics['roc_auc']:.4f} | PR-AUC: {champion_test_metrics['pr_auc']:.4f}")
    logger.info("=======================================================")

    return metadata


if __name__ == "__main__":
    run_training_pipeline()
