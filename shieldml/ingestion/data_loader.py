"""Data Ingestion, Schema Validation, and Dataset Profiling for ShieldML."""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import pandas as pd
import numpy as np

from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.ingestion.data_loader")


class DataLoader:
    """Loads and validates transaction datasets for fraud detection."""

    REQUIRED_COLUMNS = [
        "transaction_id",
        "timestamp",
        "user_id",
        "amount",
        "merchant_id",
    ]
    OPTIONAL_COLUMNS = [
        "merchant_category",
        "device_id",
        "channel",
        "distance_from_home",
        "is_fraud",
    ]

    def __init__(self, target_col: str = "is_fraud"):
        self.target_col = target_col

    def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Loads and parses a CSV transaction file.

        Args:
            file_path: Path to the CSV file.

        Returns:
            pd.DataFrame: Cleaned and chronologically sorted DataFrame.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")

        logger.info(f"Loading transaction dataset from {path}")
        df = pd.read_csv(path)
        return self.validate_and_clean(df)

    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates schema, checks missing values, and sorts chronologically.

        Args:
            df: Input raw DataFrame.

        Returns:
            pd.DataFrame: Cleaned and validated DataFrame.
        """
        df = df.copy()

        # Check required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")

        # Ensure timestamp is datetime
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception as e:
            raise ValueError(f"Failed to parse 'timestamp' column as datetime: {e}")

        # Check amount is positive
        if (df["amount"] < 0).any():
            invalid_count = (df["amount"] < 0).sum()
            logger.warning(f"Detected {invalid_count} negative amounts; clipping to 0.0")
            df["amount"] = df["amount"].clip(lower=0.0)

        # Drop duplicates by transaction_id if present
        if "transaction_id" in df.columns:
            dups = df.duplicated(subset=["transaction_id"]).sum()
            if dups > 0:
                logger.warning(f"Removing {dups} duplicate transaction IDs")
                df = df.drop_duplicates(subset=["transaction_id"])

        # Sort chronologically (strict requirement to prevent temporal leakage)
        df = df.sort_values(by="timestamp").reset_index(drop=True)

        logger.info(f"Successfully validated {len(df)} transactions spanning "
                    f"{df['timestamp'].min()} to {df['timestamp'].max()}")
        return df

    def profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates comprehensive summary profile of the dataset.

        Args:
            df: Validated DataFrame.

        Returns:
            dict: Statistical summary of class distribution, missing values, and temporal range.
        """
        ts_series = pd.to_datetime(df["timestamp"])
        min_ts = ts_series.min()
        max_ts = ts_series.max()
        dur_days = round((max_ts - min_ts).total_seconds() / 86400, 2) if len(df) > 1 else 0.0

        profile: Dict[str, Any] = {
            "total_records": int(len(df)),
            "total_features": int(df.shape[1]),
            "start_time": str(min_ts),
            "end_time": str(max_ts),
            "duration_days": dur_days,
            "unique_users": int(df["user_id"].nunique()) if "user_id" in df.columns else None,
            "unique_merchants": int(df["merchant_id"].nunique()) if "merchant_id" in df.columns else None,
            "unique_devices": int(df["device_id"].nunique()) if "device_id" in df.columns else None,
            "amount_stats": {
                "mean": round(float(df["amount"].mean()), 2),
                "std": round(float(df["amount"].std()), 2),
                "median": round(float(df["amount"].median()), 2),
                "min": round(float(df["amount"].min()), 2),
                "max": round(float(df["amount"].max()), 2),
            },
            "missing_values": {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0},
        }

        if self.target_col in df.columns:
            target_counts = df[self.target_col].value_counts().to_dict()
            total = len(df)
            fraud_count = int(target_counts.get(1, 0))
            legit_count = int(target_counts.get(0, 0))
            fraud_rate = fraud_count / total if total > 0 else 0.0

            profile["class_distribution"] = {
                "legitimate_count": legit_count,
                "fraud_count": fraud_count,
                "fraud_rate_pct": round(fraud_rate * 100, 3),
                "imbalance_ratio": f"1:{int(legit_count / max(1, fraud_count))}",
            }

        return profile
