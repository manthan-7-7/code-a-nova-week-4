"""FastAPI Route Handlers for Inference, Monitoring, and System Info."""

import io
import time
import os
try:
    import psutil
except ImportError:
    psutil = None
from typing import List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Query, status
from fastapi.responses import FileResponse, StreamingResponse

from shieldml.config import settings
from shieldml.api.schemas import (
    TransactionInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    MetricsResponse,
    ExplanationResponse,
)
from shieldml.api.state import app_state
from shieldml.utils.timer import Timer
from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.api.routes")
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Real-Time Single Transaction Fraud Inference",
    description="Scores an incoming transaction against the champion model, evaluates against optimal threshold, and computes local SHAP attributions.",
)
async def predict_single(transaction: TransactionInput):
    """Real-time sub-50ms fraud scoring and local SHAP explanation endpoint."""
    if not app_state.is_ready:
        app_state.monitoring.record_error()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ShieldML model artifacts are not loaded. Train models or check artifact paths.",
        )

    timer = Timer().start()
    try:
        # 1. Feature extraction
        raw_dict = transaction.model_dump()
        if app_state.engineer is not None:
            feat_dict = app_state.engineer.transform_single(raw_dict)
        else:
            feat_dict = raw_dict

        # Convert to single-row DataFrame for pipeline
        feat_df = pd.DataFrame([feat_dict])
        X_trans = app_state.preprocessor.transform(feat_df)

        # 2. Model Inference
        probs = app_state.model.predict_proba(X_trans)
        prob = float(probs[0, 1] if probs.ndim == 2 else probs[0])

        # 3. Decision threshold comparison
        threshold = app_state.threshold
        is_fraud = bool(prob >= threshold)

        # Determine action and risk level
        if prob >= max(0.80, threshold):
            action = "BLOCK"
            risk_level = "CRITICAL"
        elif prob >= threshold:
            action = "FLAG_CHALLENGE"
            risk_level = "HIGH"
        elif prob >= threshold * 0.60:
            action = "FLAG_CHALLENGE"
            risk_level = "MEDIUM"
        else:
            action = "APPROVE"
            risk_level = "LOW"

        # 4. SHAP Local Explanation
        explanation_data = None
        if app_state.explainer is not None:
            raw_expl = app_state.explainer.explain_single(
                features_row=X_trans[0], raw_inputs=raw_dict, top_k=5
            )
            explanation_data = ExplanationResponse(
                base_value=raw_expl["base_value"],
                top_risk_drivers=raw_expl["top_risk_drivers"],
                top_trust_drivers=raw_expl["top_trust_drivers"],
                summary_text=raw_expl["summary_text"],
            )

        latency_ms = timer.stop()

        # Record monitoring
        app_state.monitoring.record_prediction(
            latency_ms=latency_ms,
            is_fraud=is_fraud,
            record_summary={
                "transaction_id": transaction.transaction_id,
                "timestamp": transaction.timestamp or time.strftime("%Y-%m-%d %H:%M:%S"),
                "amount": transaction.amount,
                "merchant_id": transaction.merchant_id,
                "user_id": transaction.user_id,
                "fraud_probability": round(prob, 4),
                "is_fraud": is_fraud,
                "action": action,
                "latency_ms": round(latency_ms, 2),
            },
        )

        return PredictionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=round(prob, 4),
            decision_threshold=round(threshold, 4),
            is_fraud=is_fraud,
            action=action,
            risk_level=risk_level,
            latency_ms=round(latency_ms, 2),
            explanation=explanation_data,
            model_version=app_state.model_version,
        )

    except Exception as e:
        app_state.monitoring.record_error()
        logger.error(f"Inference error on {transaction.transaction_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Transaction Fraud Scoring",
    description="Processes a batch of transactions via JSON payload.",
)
async def predict_batch(batch_req: BatchPredictionRequest):
    """Processes multiple transactions in a vectorized batch."""
    if not app_state.is_ready:
        app_state.monitoring.record_error()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ShieldML model artifacts are not loaded.",
        )

    timer = Timer().start()
    txs = batch_req.transactions
    if not txs:
        raise HTTPException(status_code=400, detail="Empty transaction batch.")

    raw_dicts = [t.model_dump() for t in txs]
    df_raw = pd.DataFrame(raw_dicts)

    # Feature transformation
    if app_state.engineer is not None:
        if "timestamp" not in df_raw.columns or df_raw["timestamp"].isna().any():
            df_raw["timestamp"] = pd.Timestamp.now()
        df_feat = app_state.engineer.transform(df_raw, is_training=False)
    else:
        df_feat = df_raw

    X_trans = app_state.preprocessor.transform(df_feat)
    probs = app_state.model.predict_proba(X_trans)
    y_prob = probs[:, 1] if probs.ndim == 2 else probs

    threshold = app_state.threshold
    predictions = []
    flagged_count = 0

    for idx, (tx, p) in enumerate(zip(txs, y_prob)):
        prob = float(p)
        is_f = bool(prob >= threshold)
        if is_f:
            flagged_count += 1
            action = "BLOCK" if prob >= 0.80 else "FLAG_CHALLENGE"
            risk = "CRITICAL" if prob >= 0.80 else "HIGH"
        else:
            action = "APPROVE"
            risk = "LOW"

        predictions.append(
            PredictionResponse(
                transaction_id=tx.transaction_id,
                fraud_probability=round(prob, 4),
                decision_threshold=round(threshold, 4),
                is_fraud=is_f,
                action=action,
                risk_level=risk,
                latency_ms=0.0,
                explanation=None,
                model_version=app_state.model_version,
            )
        )

    total_time_ms = timer.stop()
    mean_lat = total_time_ms / len(txs)

    app_state.monitoring.total_requests += 1
    app_state.monitoring.total_predictions += len(txs)
    app_state.monitoring.fraud_count += flagged_count

    return BatchPredictionResponse(
        total_processed=len(txs),
        flagged_fraud_count=flagged_count,
        fraud_rate_pct=round(flagged_count / len(txs) * 100.0, 2),
        total_processing_time_ms=round(total_time_ms, 2),
        mean_latency_per_tx_ms=round(mean_lat, 2),
        predictions=predictions,
    )


@router.post(
    "/predict/batch-csv",
    summary="Batch CSV File Upload and Scoring",
    description="Upload a CSV file of transactions, detect fraud, and return the scored results.",
)
async def predict_batch_csv(file: UploadFile = File(...)):
    """Uploads CSV file, runs inference, and returns JSON summary with flagged fraud transactions."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {e}")

    # Validate essential columns
    for col in ["transaction_id", "amount", "user_id", "merchant_id"]:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"CSV missing mandatory column: '{col}'")

    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp.now()

    timer = Timer().start()
    if app_state.engineer is not None:
        df_feat = app_state.engineer.transform(df, is_training=False)
    else:
        df_feat = df

    X_trans = app_state.preprocessor.transform(df_feat)
    probs = app_state.model.predict_proba(X_trans)
    y_prob = probs[:, 1] if probs.ndim == 2 else probs

    threshold = app_state.threshold
    df["fraud_probability"] = np.round(y_prob, 4)
    df["is_fraud_predicted"] = (df["fraud_probability"] >= threshold).astype(int)
    df["decision"] = np.where(df["fraud_probability"] >= 0.80, "BLOCK", np.where(df["is_fraud_predicted"] == 1, "FLAG_CHALLENGE", "APPROVE"))

    total_time_ms = timer.stop()
    flagged = int(df["is_fraud_predicted"].sum())

    # Return top 50 sample records for frontend preview
    sample_preview = df.head(50).to_dict(orient="records")

    return {
        "filename": file.filename,
        "total_rows": len(df),
        "flagged_fraud_count": flagged,
        "fraud_rate_pct": round(flagged / len(df) * 100.0, 2),
        "total_time_ms": round(total_time_ms, 2),
        "mean_latency_per_tx_ms": round(total_time_ms / len(df), 2),
        "sample_preview": sample_preview,
    }


@router.get("/metrics", response_model=MetricsResponse, summary="Operational Metrics and SLA Statistics")
async def get_metrics():
    """Returns operational KPIs: total predictions, fraud rate, mean/P95 latency, SLA compliance."""
    return app_state.monitoring.get_metrics()


@router.get("/health", response_model=HealthResponse, summary="System Health and Readiness Check")
async def get_health():
    """Checks service health, model loaded status, uptime, and memory footprint."""
    uptime = time.time() - app_state.monitoring.start_time
    if psutil is not None:
        try:
            mem_mb = float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
        except Exception:
            mem_mb = 0.0
    else:
        mem_mb = 0.0

    return HealthResponse(
        status="healthy" if app_state.is_ready else "degraded",
        model_loaded=app_state.is_ready,
        model_name=app_state.model.__class__.__name__ if app_state.model else "none",
        model_version=app_state.model_version,
        decision_threshold=app_state.threshold,
        uptime_seconds=round(uptime, 2),
        memory_usage_mb=round(mem_mb, 2),
    )


@router.get("/model/info", response_model=ModelInfoResponse, summary="Champion Model Information")
async def get_model_info():
    """Returns champion model lineage, hyperparameters, threshold, validation performance."""
    if not app_state.is_ready:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    feature_names = app_state.preprocessor.get_feature_names()
    return ModelInfoResponse(
        model_name=app_state.metadata.get("model_name", "ShieldML-Champion"),
        model_version=app_state.model_version,
        model_type=app_state.metadata.get("model_type", app_state.model.__class__.__name__),
        training_timestamp=app_state.metadata.get("training_timestamp"),
        optimal_threshold=app_state.threshold,
        optimization_metric=app_state.metadata.get("optimization_metric", "f2_score"),
        f_beta=settings.F_BETA,
        feature_count=len(feature_names),
        feature_names=feature_names,
        validation_metrics=app_state.metadata.get("validation_metrics", {}),
        test_metrics=app_state.metadata.get("test_metrics", {}),
        sla_target_ms=settings.TARGET_LATENCY_MS,
    )


@router.get("/api/plots/{plot_name}", summary="Retrieve Generated Evaluation Plots")
async def get_plot(plot_name: str):
    """Serves static PNG plots from artifacts/plots/."""
    safe_name = Path(plot_name).name
    plot_path = settings.PLOTS_DIR / safe_name
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail=f"Plot '{plot_name}' not found.")
    return FileResponse(plot_path, media_type="image/png")


@router.get("/api/monitoring/recent", summary="Recent Transaction Stream for Dashboard")
async def get_recent_transactions():
    """Returns recent predictions for live dashboard stream."""
    return app_state.monitoring.recent_predictions


@router.get("/api/metadata", summary="Raw Metadata and Model Comparison Data")
async def get_metadata():
    """Returns the complete metadata.json content for UI visualization."""
    return app_state.metadata
