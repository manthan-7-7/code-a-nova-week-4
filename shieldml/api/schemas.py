"""Pydantic Request and Response Schemas for ShieldML Serving."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class TransactionInput(BaseModel):
    """Payload representing an incoming financial transaction for fraud inference."""

    transaction_id: str = Field(..., examples=["tx_88492011"], description="Unique transaction identifier")
    timestamp: Optional[str] = Field(None, examples=["2024-01-15 14:32:00"], description="ISO or standard timestamp")
    user_id: str = Field(..., examples=["usr_00421"], description="Customer/User identifier")
    amount: float = Field(..., ge=0.0, examples=[245.50], description="Transaction monetary amount")
    merchant_id: str = Field(..., examples=["merch_0012"], description="Merchant identifier")
    merchant_category: Optional[str] = Field("retail", examples=["electronics"], description="Category of merchant")
    device_id: Optional[str] = Field("unknown", examples=["dev_09812"], description="Device hardware fingerprint")
    channel: Optional[str] = Field("mobile_app", examples=["web"], description="Transaction initiation channel")
    distance_from_home: Optional[float] = Field(0.0, ge=0.0, examples=[14.5], description="Distance in km from user home base")

    # Optional precomputed or past velocities (if known)
    tx_count_user_1h: Optional[int] = Field(0, ge=0, examples=[1])
    tx_amount_sum_user_1h: Optional[float] = Field(0.0, ge=0.0, examples=[45.0])
    tx_count_user_24h: Optional[int] = Field(0, ge=0, examples=[3])
    tx_amount_sum_user_24h: Optional[float] = Field(0.0, ge=0.0, examples=[120.0])
    tx_count_user_7d: Optional[int] = Field(0, ge=0, examples=[10])
    tx_amount_sum_user_7d: Optional[float] = Field(0.0, ge=0.0, examples=[480.0])

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("Transaction amount must be non-negative.")
        return float(v)


class FeatureDriver(BaseModel):
    """Individual SHAP feature attribution factor."""

    feature: str
    shap_value: float
    transformed_value: float
    impact: str
    narrative: str


class ExplanationResponse(BaseModel):
    """Explainable AI attribution response."""

    base_value: float
    top_risk_drivers: List[FeatureDriver]
    top_trust_drivers: List[FeatureDriver]
    summary_text: str


class PredictionResponse(BaseModel):
    """Output prediction and risk assessment for a single transaction."""

    transaction_id: str
    fraud_probability: float
    decision_threshold: float
    is_fraud: bool
    action: Literal["APPROVE", "FLAG_CHALLENGE", "BLOCK"]
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    latency_ms: float
    explanation: Optional[ExplanationResponse] = None
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction payload containing multiple transactions."""

    transactions: List[TransactionInput]


class BatchPredictionResponse(BaseModel):
    """Output for batch prediction processing."""

    total_processed: int
    flagged_fraud_count: int
    fraud_rate_pct: float
    total_processing_time_ms: float
    mean_latency_per_tx_ms: float
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    decision_threshold: float
    uptime_seconds: float
    memory_usage_mb: float


class ModelInfoResponse(BaseModel):
    """Champion model metadata and training lineage schema."""

    model_name: str
    model_version: str
    model_type: str
    training_timestamp: Optional[str]
    optimal_threshold: float
    optimization_metric: str
    f_beta: float
    feature_count: int
    feature_names: List[str]
    validation_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    sla_target_ms: float


class MetricsResponse(BaseModel):
    """Live operational monitoring and latency statistics."""

    total_requests: int
    total_predictions: int
    fraud_predictions_count: int
    recent_fraud_rate_pct: float
    error_count: int
    mean_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    sla_target_ms: float
    sla_compliance_rate_pct: float
    window_samples: int
