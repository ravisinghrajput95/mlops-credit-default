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
from typing import Annotated, Any

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from credit_default.api.auth import (
    Authenticator,
    bearer_scheme,
    build_authenticator,
    credentials_secret,
)
from credit_default.api.model import ModelHandle, load_model
from credit_default.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    Prediction,
    PredictionRequest,
    PredictionResponse,
    Reason,
)
from credit_default.api.sinks import NullSink, PredictionSink, build_sink
from credit_default.config import (
    AUDIT_ATTRIBUTES,
    FEATURES,
    PROTECTED_ATTRIBUTES,
    get_settings,
)

logger = logging.getLogger(__name__)

EXPLANATION_SECONDS = Histogram(
    "explanation_duration_seconds",
    "Time spent computing SHAP explanations",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Distribution of predicted default probabilities",
    buckets=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


class AppState:
    """Holds the objects built once at startup."""

    model: ModelHandle | None = None
    sink: PredictionSink = NullSink()
    # Open until the lifespan replaces it, so that a test constructing the app
    # without running startup gets the documented default rather than an
    # AttributeError. Every deployed path goes through the lifespan.
    auth: Authenticator = Authenticator([], required=False)


state = AppState()


def require_caller(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> str:
    """Resolve the caller's name, or reject with 401.

    Returns a name rather than a bool so the caller can be written to the
    prediction log: a credit decision should record who asked for it.
    """
    return state.auth.identify(credentials_secret(credentials))


Caller = Annotated[str, Depends(require_caller)]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")

    # Deliberately not wrapped in try/except, unlike everything else in this
    # function. A missing model degrades; a broken auth configuration must not.
    # "No keys configured" would mean the endpoint stops being able to reject
    # anyone, and an API that has quietly stopped checking is worse than one that
    # is plainly failing to start.
    state.auth = build_authenticator(settings.api_keys.get_secret_value(), settings.require_auth)

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
def model_info(caller: Caller) -> ModelInfoResponse:
    if state.model is None:
        raise HTTPException(status_code=503, detail="No model is loaded.")
    settings = get_settings()
    return ModelInfoResponse(
        model_source=state.model.source,
        model_version=state.model.version,
        registered_model_name=settings.registered_model_name,
        alias=settings.champion_alias,
        threshold=state.model.threshold,
        features=FEATURES,
        excluded_attributes=([] if settings.use_protected_attributes else PROTECTED_ATTRIBUTES),
        audited_attributes=AUDIT_ATTRIBUTES,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest, background: BackgroundTasks, caller: Caller
) -> PredictionResponse:
    if state.model is None:
        raise HTTPException(status_code=503, detail="No model is loaded.")

    # The key is stripped before the frame is built: it identifies the decision,
    # it is not evidence about the applicant. Letting it reach the pipeline would
    # be a straightforward leak of an arbitrary identifier into the model.
    application_ids = [a.application_id or str(uuid.uuid4()) for a in request.applications]
    payload = [a.model_dump(exclude={"application_id"}) for a in request.applications]
    frame = pd.DataFrame(payload)

    try:
        probabilities, labels = state.model.predict(frame)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed.") from exc

    for probability in probabilities:
        PREDICTION_PROBABILITY.observe(probability)

    reasons: list[list[Reason]] | None = None
    if request.explain:
        try:
            with EXPLANATION_SECONDS.time():
                raw = state.model.explain(frame, [bool(v) for v in labels])
            reasons = [[Reason(**r.to_dict()) for r in row] for row in raw]
        except Exception:
            # An explanation failure must not deny someone a decision; the caller
            # sees reasons omitted rather than a 500.
            logger.exception("Explanation failed; returning predictions without reasons")

    now = dt.datetime.now(dt.UTC)
    background.add_task(
        _record,
        [
            {
                "id": str(uuid.uuid4()),
                "application_id": application_id,
                # Who asked. An adverse-action decision that cannot be traced to
                # a requester is a hole in the same audit trail the SHAP reasons
                # exist to fill.
                "caller": caller,
                "predicted_at": now,
                "model_version": state.model.version,
                "probability": probability,
                "prediction": label,
                "features": features,
            }
            for application_id, probability, label, features in zip(
                application_ids, probabilities, labels, payload, strict=True
            )
        ],
    )

    return PredictionResponse(
        predictions=[
            Prediction(
                application_id=application_id,
                probability=p,
                prediction=v,  # type: ignore[arg-type]
                reasons=reasons[i] if reasons else None,
            )
            for i, (application_id, p, v) in enumerate(
                zip(application_ids, probabilities, labels, strict=True)
            )
        ],
        model_version=state.model.version,
        threshold=state.model.threshold,
    )


@app.get("/", include_in_schema=False)
def root() -> Response:
    return Response(status_code=307, headers={"Location": "/docs"})
