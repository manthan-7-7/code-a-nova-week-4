"""Scikit-Learn Compatible Preprocessing Pipeline for ShieldML."""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.features.pipeline")


class FraudFeaturePipeline(BaseEstimator, TransformerMixin):
    """Encapsulates scaling and encoding transformations fitted strictly on training data."""

    NUMERICAL_FEATURES: List[str] = [
        "amount",
        "log_amount",
        "distance_from_home",
        "log_distance",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "is_weekend",
        "is_night",
        "merchant_amount_zscore",
        "device_usage_count",
        "is_new_device_for_user",
        "tx_count_user_1h",
        "tx_amount_sum_user_1h",
        "tx_amount_mean_user_1h",
        "tx_count_user_24h",
        "tx_amount_sum_user_24h",
        "tx_amount_mean_user_24h",
        "amount_to_mean_ratio_24h",
        "tx_count_user_7d",
        "tx_amount_sum_user_7d",
        "tx_amount_mean_user_7d",
    ]

    CATEGORICAL_FEATURES: List[str] = [
        "merchant_category",
        "channel",
    ]

    def __init__(self):
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "FraudFeaturePipeline":
        """Fits scalers and encoders on training features only.

        Args:
            X: Training DataFrame containing engineered features.
            y: Optional target labels (ignored).

        Returns:
            FraudFeaturePipeline: Fitted instance.
        """
        logger.info(f"Fitting FraudFeaturePipeline on {len(X)} records...")

        # Ensure all required features are present
        missing_num = [f for f in self.NUMERICAL_FEATURES if f not in X.columns]
        missing_cat = [f for f in self.CATEGORICAL_FEATURES if f not in X.columns]
        if missing_num or missing_cat:
            raise ValueError(f"Missing columns for pipeline: num={missing_num}, cat={missing_cat}")

        # Build column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.NUMERICAL_FEATURES),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.CATEGORICAL_FEATURES,
                ),
            ],
            remainder="drop",
        )

        self.preprocessor.fit(X)

        # Extract output feature names
        cat_encoder = self.preprocessor.named_transformers_["cat"]
        cat_features_out = list(cat_encoder.get_feature_names_out(self.CATEGORICAL_FEATURES))
        self.feature_names_ = self.NUMERICAL_FEATURES + cat_features_out

        self.is_fitted_ = True
        logger.info(f"FraudFeaturePipeline fitted successfully. Output dimensions: {len(self.feature_names_)} features.")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms feature DataFrame into standardized model array.

        Args:
            X: Input DataFrame with engineered features.

        Returns:
            np.ndarray: Scaled and encoded 2D matrix.
        """
        if not self.is_fitted_ or self.preprocessor is None:
            raise RuntimeError("FraudFeaturePipeline is not fitted yet.")

        # Impute any missing numerical values with 0.0 or median
        X_clean = X.copy()
        for col in self.NUMERICAL_FEATURES:
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna(0.0)
            else:
                X_clean[col] = 0.0

        for col in self.CATEGORICAL_FEATURES:
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna("unknown").astype(str)
            else:
                X_clean[col] = "unknown"

        return self.preprocessor.transform(X_clean)

    def transform_to_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms and returns a named DataFrame (useful for SHAP explainability)."""
        arr = self.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_, index=X.index)

    def get_feature_names(self) -> List[str]:
        """Returns the list of feature names matching the transformed array columns."""
        return self.feature_names_
