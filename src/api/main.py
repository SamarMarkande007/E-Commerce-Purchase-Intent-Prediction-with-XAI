"""FastAPI service for real-time purchase-intent scoring.

Loads the saved pipeline (Phase 6) and builds a SHAP explainer (Phase 7)
once at startup — not per-request, which would be far too slow. Run
locally with::

    uvicorn src.api.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import ExplanationFactor, HealthResponse, PredictionResponse, SessionInput
from src.config.settings import get_settings
from src.explain.shap_utils import SessionExplainer
from src.features.engineering import engineer_features
from src.utils.exceptions import ModelNotFoundError
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

#: Populated once at startup by the lifespan context manager below.
_state: dict = {}


def _confidence_label(probability: float, threshold: float) -> str:
    """Turn a raw probability into a simple, human-readable confidence label.

    Not a calibrated statistical confidence interval — just how far the
    probability sits from the decision threshold, bucketed for display.
    A probability right at the threshold is a genuine toss-up ("low"
    confidence in the decision either way); one far from it is a
    decisive case either direction.

    Args:
        probability: Predicted purchase probability.
        threshold: The decision threshold in use.

    Returns:
        One of ``"low"``, ``"medium"``, ``"high"``.
    """
    distance = abs(probability - threshold)
    if distance >= 0.3:
        return "high"
    if distance >= 0.1:
        return "medium"
    return "low"


def _load_model_state() -> dict:
    """Load the pipeline, metadata, and SHAP explainer from disk.

    Raises:
        ModelNotFoundError: If the saved pipeline or metadata file is
            missing — a clear error at startup, rather than a confusing
            failure on the first request.
    """
    settings = get_settings()
    models_dir = settings.paths.models_dir

    pipeline_path = models_dir / "best_pipeline.joblib"
    metadata_path = models_dir / "best_model_metadata.json"

    if not pipeline_path.exists():
        raise ModelNotFoundError(
            f"No saved model found at {pipeline_path}. "
            f"Run notebooks/03_modeling.ipynb to train and save one first."
        )
    if not metadata_path.exists():
        raise ModelNotFoundError(f"No model metadata found at {metadata_path}.")

    pipeline = joblib.load(pipeline_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    explainer = SessionExplainer(pipeline)

    logger.info(
        "Model loaded: %s (%s), decision_threshold=%.4f",
        metadata["model_name"],
        metadata["imbalance_strategy"],
        metadata["decision_threshold"],
    )

    return {
        "settings": settings,
        "pipeline": pipeline,
        "explainer": explainer,
        "metadata": metadata,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the API starts, not on every request."""
    settings = get_settings()
    setup_logging(level="INFO", log_file=settings.paths.log_file)
    logger.info("Starting Purchase-Intent API...")

    _state.update(_load_model_state())
    yield

    logger.info("Shutting down Purchase-Intent API.")
    _state.clear()


app = FastAPI(
    title="Purchase-Intent XAI API",
    description="Real-time purchase-intent scoring with SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness check — confirms the model is loaded and ready."""
    metadata = _state["metadata"]
    return HealthResponse(
        status="ok",
        model_name=metadata["model_name"],
        decision_threshold=metadata["decision_threshold"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(session: SessionInput) -> PredictionResponse:
    """Score one e-commerce session and explain the prediction.

    Args:
        session: Raw session fields (see SessionInput). Pydantic
            validates types/ranges before this function ever runs;
            malformed input is rejected with a 422 automatically.

    Returns:
        The purchase probability, the threshold-based decision, the
        threshold itself, and the top 5 SHAP contributors for this
        specific session.

    Raises:
        HTTPException: 500 if scoring fails after validation passes
            (e.g. an unexpected internal error), with the underlying
            cause logged server-side.
    """
    pipeline = _state["pipeline"]
    explainer = _state["explainer"]
    metadata = _state["metadata"]

    try:
        raw_df = pd.DataFrame([session.model_dump()])
        engineered_df = engineer_features(raw_df)

        probability = float(pipeline.predict_proba(engineered_df)[:, 1][0])
        threshold = float(metadata["decision_threshold"])

        top_factors_raw = explainer.explain(engineered_df, top_n=5)
        top_factors = [ExplanationFactor(**factor) for factor in top_factors_raw]

        return PredictionResponse(
            purchase_probability=probability,
            will_purchase=probability >= threshold,
            confidence=_confidence_label(probability, threshold),
            decision_threshold=threshold,
            top_factors=top_factors,
        )

    except Exception as exc:
        logger.exception("Prediction failed for session input: %s", session.model_dump())
        raise HTTPException(
            status_code=500, detail="Prediction failed. See server logs for details."
        ) from exc
