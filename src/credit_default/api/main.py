"""Inference API.

Exposes the champion model behind a validated HTTP boundary, records every
prediction for downstream drift monitoring, and publishes Prometheus metrics.

The custom ``prediction_probability`` histogram matters more than it looks: in
credit default the ground-truth label arrives months after the prediction, so
accuracy cannot be monitored live. A shift in the *distribution of scores* is the
earliest signal available that the input population has moved.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from credit_default.api.model import ModelHandle, load_model
from credit_default.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    Prediction,
    PredictionRequest,
    PredictionResponse,
)
from credit_default.api.sinks import NullSink, PredictionSink, build_sink
from credit_default.config import (
    AUDIT_ATTRIBUTES,
    FEATURES,
    PROTECTED_ATTRIBUTES,
    get_settings,
)

logger = logging.getLogger(__name__)

DECISION_THRESHOLD = 0.5

PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Distribution of predicted default probabilities",
    buckets=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


class AppState:
    """Holds the objects built once at startup."""

    model: ModelHandle | None = None
    sink: PredictionSink = NullSink()


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")

    try:
        state.model = load_model(settings)
    except Exception:
        # /health reports degraded rather than crash-looping the container, which
        # makes the failure visible in the orchestrator instead of as a restart storm.
        logger.exception("Model failed to load; the API will report degraded health.")
        state.model = None

    state.sink = build_sink(settings)
    yield
    state.sink.close()


app = FastAPI(
    title="Credit Default Prediction API",
    description="Serves the champion credit-default model with monitoring hooks.",
    version="0.1.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


def _record(
    records: list[dict[str, Any]],
) -> None:
    """Persist served predictions, never letting a sink failure surface to the caller."""
    try:
        state.sink.write(records)
    except Exception:
        logger.exception("Failed to record %d predictions", len(records))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = state.model is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if state.model is None:
        raise HTTPException(status_code=503, detail="No model is loaded.")
    settings = get_settings()
    return ModelInfoResponse(
        model_source=state.model.source,
        model_version=state.model.version,
        registered_model_name=settings.registered_model_name,
        alias=settings.champion_alias,
        threshold=DECISION_THRESHOLD,
        features=FEATURES,
        excluded_attributes=([] if settings.use_protected_attributes else PROTECTED_ATTRIBUTES),
        audited_attributes=AUDIT_ATTRIBUTES,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, background: BackgroundTasks) -> PredictionResponse:
    if state.model is None:
        raise HTTPException(status_code=503, detail="No model is loaded.")

    payload = [a.model_dump() for a in request.applications]
    frame = pd.DataFrame(payload)

    try:
        probabilities, labels = state.model.predict(frame, DECISION_THRESHOLD)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed.") from exc

    for probability in probabilities:
        PREDICTION_PROBABILITY.observe(probability)

    now = dt.datetime.now(dt.UTC)
    background.add_task(
        _record,
        [
            {
                "id": str(uuid.uuid4()),
                "predicted_at": now,
                "model_version": state.model.version,
                "probability": probability,
                "prediction": label,
                "features": features,
            }
            for probability, label, features in zip(probabilities, labels, payload, strict=True)
        ],
    )

    return PredictionResponse(
        predictions=[
            Prediction(probability=p, prediction=v)  # type: ignore[arg-type]
            for p, v in zip(probabilities, labels, strict=True)
        ],
        model_version=state.model.version,
        threshold=DECISION_THRESHOLD,
    )


@app.get("/", include_in_schema=False)
def root() -> Response:
    return Response(status_code=307, headers={"Location": "/docs"})
