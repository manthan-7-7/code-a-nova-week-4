"""MLflow Experiment Tracking and Model Registry Integration."""

from pathlib import Path
from typing import Dict, Any, Optional
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from shieldml.config import settings
from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.tracking.mlflow_tracker")


class MLflowTracker:
    """Manages MLflow runs, child tuning runs, metrics, artifacts, and champion registration."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME

        mlflow.set_tracking_uri(self.tracking_uri)
        self.experiment = mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        logger.info(f"MLflow initialized: URI='{self.tracking_uri}', Experiment='{self.experiment_name}'")

    def log_training_experiment(
        self,
        champion_name: str,
        champion_model: Any,
        preprocessor: Any,
        candidate_results: Dict[str, Dict[str, Any]],
        optimal_threshold: float,
        test_metrics: Dict[str, Any],
        plots_dir: Path,
        metadata: Dict[str, Any],
    ) -> str:
        """Logs the complete training pipeline run, child model runs, artifacts, and registers champion.

        Returns:
            str: MLflow active run ID.
        """
        with mlflow.start_run(run_name=f"ShieldML-Pipeline-{champion_name}") as parent_run:
            run_id = parent_run.info.run_id
            logger.info(f"Started MLflow Parent Run: {run_id}")

            # 1. Log overall pipeline parameters
            mlflow.log_params({
                "champion_model_type": champion_name,
                "optimal_threshold": optimal_threshold,
                "optimization_metric": settings.OPTIMIZATION_METRIC,
                "f_beta": settings.F_BETA,
                "random_state": settings.RANDOM_STATE,
                "target_latency_ms": settings.TARGET_LATENCY_MS,
            })

            # 2. Log test set metrics for the champion
            flat_metrics = {
                f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))
            }
            if "confusion_matrix" in test_metrics:
                cm = test_metrics["confusion_matrix"]
                for cm_k in ["tn", "fp", "fn", "tp"]:
                    flat_metrics[f"test_cm_{cm_k}"] = cm[cm_k]

            mlflow.log_metrics(flat_metrics)

            # 3. Log child runs for all candidate models
            for cand_name, cand_info in candidate_results.items():
                with mlflow.start_run(run_name=f"Candidate-{cand_name}", nested=True):
                    if "params" in cand_info:
                        mlflow.log_params(cand_info["params"])
                    if "metrics" in cand_info:
                        m = cand_info["metrics"]
                        sub_metrics = {k: v for k, v in m.items() if isinstance(v, (int, float))}
                        mlflow.log_metrics(sub_metrics)

            # 4. Log visual plot artifacts
            if plots_dir.exists():
                for plot_file in plots_dir.glob("*.png"):
                    mlflow.log_artifact(str(plot_file), artifact_path="evaluation_plots")
                logger.info("Evaluation plots logged to MLflow artifacts.")

            # 5. Log metadata JSON
            metadata_file = plots_dir.parent / "metadata.json"
            if metadata_file.exists():
                mlflow.log_artifact(str(metadata_file), artifact_path="metadata")

            # 6. Log and register champion model
            try:
                model_info = mlflow.sklearn.log_model(
                    sk_model=champion_model,
                    artifact_path="model",
                    registered_model_name=settings.MLFLOW_REGISTERED_MODEL_NAME,
                )
                logger.info(f"Champion model logged & registered: {settings.MLFLOW_REGISTERED_MODEL_NAME} ({model_info.model_uri})")
            except Exception as e:
                logger.warning(f"Model registry warning (may require backend DB): {e}. Saving model artifact directly.")
                mlflow.sklearn.log_model(sk_model=champion_model, artifact_path="model")

            return run_id
