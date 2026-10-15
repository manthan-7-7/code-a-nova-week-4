"""Global configuration module for ShieldML."""

from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """ShieldML system settings and hyperparameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    PLOTS_DIR: Path = ARTIFACTS_DIR / "plots"

    # Artifact paths
    CHAMPION_MODEL_PATH: Path = ARTIFACTS_DIR / "champion_model.joblib"
    PREPROCESSOR_PATH: Path = ARTIFACTS_DIR / "preprocessor.joblib"
    METADATA_PATH: Path = ARTIFACTS_DIR / "metadata.json"
    EXPLAINER_PATH: Path = ARTIFACTS_DIR / "shap_explainer.joblib"

    # Application settings
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = ["*"]

    # Modeling & Validation
    RANDOM_STATE: int = 42
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # Threshold optimization
    OPTIMIZATION_METRIC: str = "f2"
    F_BETA: float = 2.0
    DECISION_THRESHOLD_OVERRIDE: Optional[float] = None

    # Latency & Monitoring SLA
    TARGET_LATENCY_MS: float = 50.0
    MONITORING_WINDOW_SIZE: int = 1000

    # MLflow Tracking
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "ShieldML-Fraud-Detection"
    MLFLOW_REGISTERED_MODEL_NAME: str = "ShieldML-Champion"

    # Feature engineering windows (in seconds)
    WINDOW_1H_SECS: int = 3600
    WINDOW_24H_SECS: int = 86400
    WINDOW_7D_SECS: int = 604800


# Global singleton instance
settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
settings.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
