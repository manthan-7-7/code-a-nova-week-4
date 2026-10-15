"""FastAPI Application Factory for ShieldML."""

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from shieldml.config import settings
from shieldml.api.routes import router
from shieldml.api.state import app_state
from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to preload models and artifacts on startup."""
    logger.info("Initializing ShieldML Application...")
    app_state.load_artifacts()
    yield
    logger.info("Shutting down ShieldML Application...")


def create_app() -> FastAPI:
    """Builds and configures the FastAPI application instance."""
    app = FastAPI(
        title="ShieldML — Real-Time Financial Fraud Detection API",
        description="High-performance machine learning inference service with sub-50ms latency, "
                    "F2-score threshold optimization, automated feature engineering, and SHAP explainability.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router)

    # Mount static assets for the frontend
    frontend_dir = settings.BASE_DIR / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/", summary="Financial Fraud Analytics Dashboard", include_in_schema=False)
        async def serve_dashboard():
            index_path = frontend_dir / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
            return {"message": "ShieldML API is running. Visit /docs for OpenAPI documentation."}

    return app


# Application entrypoint
app = create_app()
