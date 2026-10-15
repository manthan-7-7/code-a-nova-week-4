"""In-memory Application State, Model Singleton, and Operational Metrics Collector."""

import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import joblib
import numpy as np

from shieldml.config import settings
from shieldml.features.engineer import FeatureEngineer
from shieldml.features.pipeline import FraudFeaturePipeline
from shieldml.explainability.shap_explainer import FraudShapExplainer
from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.api.state")


class MonitoringState:
    """Thread-safe sliding window monitoring tracker for SLA compliance and latency percentiles."""

    def __init__(self, window_size: int = settings.MONITORING_WINDOW_SIZE):
        self.window_size = window_size
        self.start_time = time.time()
        self.total_requests: int = 0
        self.total_predictions: int = 0
        self.fraud_count: int = 0
        self.error_count: int = 0

        self.latencies: List[float] = []
        self.recent_predictions: List[Dict[str, Any]] = []

    def record_prediction(self, latency_ms: float, is_fraud: bool, record_summary: Optional[Dict[str, Any]] = None):
        self.total_requests += 1
        self.total_predictions += 1
        if is_fraud:
            self.fraud_count += 1

        self.latencies.append(latency_ms)
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)

        if record_summary:
            self.recent_predictions.insert(0, record_summary)
            if len(self.recent_predictions) > 100:
                self.recent_predictions.pop()

    def record_error(self):
        self.total_requests += 1
        self.error_count += 1

    def get_metrics(self) -> Dict[str, Any]:
        lats = np.array(self.latencies) if self.latencies else np.array([0.0])
        count = len(self.latencies)

        mean_lat = float(np.mean(lats))
        p50 = float(np.percentile(lats, 50))
        p90 = float(np.percentile(lats, 90))
        p95 = float(np.percentile(lats, 95))
        p99 = float(np.percentile(lats, 99))

        sla_compliant = float(np.sum(lats <= settings.TARGET_LATENCY_MS))
        compliance_pct = (sla_compliant / count * 100.0) if count > 0 else 100.0

        fraud_rate = (self.fraud_count / max(1, self.total_predictions)) * 100.0

        return {
            "total_requests": self.total_requests,
            "total_predictions": self.total_predictions,
            "fraud_predictions_count": self.fraud_count,
            "recent_fraud_rate_pct": round(fraud_rate, 2),
            "error_count": self.error_count,
            "mean_latency_ms": round(mean_lat, 2),
            "p50_latency_ms": round(p50, 2),
            "p90_latency_ms": round(p90, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "sla_target_ms": settings.TARGET_LATENCY_MS,
            "sla_compliance_rate_pct": round(compliance_pct, 2),
            "window_samples": count,
        }


class AppState:
    """Global application state containing preloaded models, preprocessors, and monitoring."""

    def __init__(self):
        self.model: Optional[Any] = None
        self.preprocessor: Optional[FraudFeaturePipeline] = None
        self.engineer: Optional[FeatureEngineer] = None
        self.explainer: Optional[FraudShapExplainer] = None
        self.metadata: Dict[str, Any] = {}
        self.threshold: float = 0.50
        self.model_version: str = "1.0.0"
        self.is_ready: bool = False
        self.monitoring = MonitoringState()

    def load_artifacts(self):
        """Preloads champion model, preprocessors, metadata, and SHAP explainer into memory."""
        logger.info("Preloading ML artifacts into memory...")

        # Check if artifacts exist; if not, run automated bootstrap training pipeline
        if not settings.CHAMPION_MODEL_PATH.exists() or not settings.PREPROCESSOR_PATH.exists():
            logger.info("Champion model or preprocessor not detected on disk. Initiating automated bootstrap training...")
            try:
                from scripts.train_pipeline import run_training_pipeline
                run_training_pipeline(use_optuna=False)
                logger.info("Bootstrap training completed successfully.")
            except Exception as e:
                logger.error(f"Automated bootstrap training encountered an error: {e}", exc_info=True)

        # 1. Metadata
        if settings.METADATA_PATH.exists():
            with open(settings.METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                self.threshold = float(self.metadata.get("optimal_threshold", 0.50))
                self.model_version = str(self.metadata.get("model_version", "1.0.0"))
                logger.info(f"Loaded metadata. Frozen optimal threshold: {self.threshold:.4f}")
        else:
            logger.warning(f"Metadata file not found at {settings.METADATA_PATH}; using default threshold 0.50")

        # Allow environment override
        if settings.DECISION_THRESHOLD_OVERRIDE is not None:
            self.threshold = float(settings.DECISION_THRESHOLD_OVERRIDE)
            logger.info(f"Overriding threshold via environment to: {self.threshold:.4f}")

        # 2. Champion Model
        if settings.CHAMPION_MODEL_PATH.exists():
            self.model = joblib.load(settings.CHAMPION_MODEL_PATH)
            logger.info(f"Loaded Champion Model from {settings.CHAMPION_MODEL_PATH} ({self.model.__class__.__name__})")
        else:
            logger.warning(f"Champion model not found at {settings.CHAMPION_MODEL_PATH}")

        # 3. Preprocessor
        if settings.PREPROCESSOR_PATH.exists():
            saved_prep = joblib.load(settings.PREPROCESSOR_PATH)
            if isinstance(saved_prep, tuple):
                self.engineer, self.preprocessor = saved_prep
            else:
                self.preprocessor = saved_prep
            logger.info(f"Loaded Preprocessor from {settings.PREPROCESSOR_PATH}")
        else:
            logger.warning(f"Preprocessor artifact not found at {settings.PREPROCESSOR_PATH}")

        # 4. SHAP Explainer
        if settings.EXPLAINER_PATH.exists():
            self.explainer = joblib.load(settings.EXPLAINER_PATH)
            logger.info(f"Loaded precomputed SHAP explainer from {settings.EXPLAINER_PATH}")
        elif self.model is not None and self.preprocessor is not None:
            try:
                feature_names = self.preprocessor.get_feature_names()
                self.explainer = FraudShapExplainer(self.model, feature_names)
                logger.info("Initialized SHAP explainer directly from loaded champion model.")
            except Exception as e:
                logger.warning(f"Could not initialize SHAP explainer on startup: {e}")

        self.is_ready = bool(self.model is not None and self.preprocessor is not None)
        logger.info(f"AppState load complete. Ready status: {self.is_ready}")


# Global app state singleton
app_state = AppState()
