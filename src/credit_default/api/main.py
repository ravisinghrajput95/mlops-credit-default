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
import threading
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
    Settings,
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

    # Set once the background load has finished, whether it succeeded or failed.
    # Callers that need determinism (tests, mainly) wait on it rather than
    # sleeping; nothing on the serving path blocks on it.
    load_complete: threading.Event = threading.Event()
    # Distinguishes "not loaded yet" from "tried and failed". /health conflated
    # the two before, which made a slow start and a broken model look identical
    # to whoever was paged.
    load_failed: bool = False


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


def _load_model_into_state(settings: Settings) -> None:
    """Load the champion into ``state``. Runs on a worker thread, never the startup path.

    Deliberately does not set ``state.model = None`` on failure. The attribute is
    already ``None`` unless something else deliberately populated it, and
    clobbering it here would let a slow, failing background load overwrite a model
    that a caller had injected.
    """
    try:
        state.model = load_model(settings)
        logger.info("Model loaded; the API is now ready to serve.")
    except Exception:
        # /health stays 200 so the orchestrator does not kill a live process over
        # a model problem; /ready turns 503 so no traffic is routed to it.
        state.load_failed = True
        logger.exception("Model failed to load; the API will report degraded health.")
    finally:
        state.load_complete.set()


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

    state.sink = build_sink(settings)

    # The model is loaded on a worker thread rather than here, and that is a
    # correctness fix rather than an optimisation.
    #
    # uvicorn binds the listening socket only *after* lifespan startup returns.
    # Loading the model here therefore kept the port closed for the entire load,
    # so a health probe got a refused connection rather than an answer -- which is
    # precisely the crash loop the degraded-health design exists to avoid. On
    # Cloud Run the socket stayed shut for 7m12s and every deploy failed with
    # ERROR_CONNECTION_FAILED, with no application log to say why.
    #
    # The graceful degradation only ever covered a load that *failed*. A load that
    # was merely *slow* defeated it, because there was nothing listening to report
    # the degradation. Binding first and loading after is what makes the promise
    # true for both.
    state.load_complete = threading.Event()
    state.load_failed = False
    threading.Thread(
        target=_load_model_into_state,
        args=(settings,),
        name="model-loader",
        daemon=True,
    ).start()

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
    """Liveness: is this process alive? Always 200 while it can answer at all.

    A model problem is not a reason to kill a running process, so this never
    fails the way ``/ready`` does. The two probes answer different questions and
    pointing both at this one -- which is what the deployment used to do -- means
    the orchestrator cannot tell "starting up" from "cannot serve".
    """
    if state.model is not None:
        return HealthResponse(status="ok", model_loaded=True)
    if state.load_failed:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(status="loading", model_loaded=False)


@app.get("/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    """Readiness: may this instance be sent traffic yet?

    503 until the model is actually loaded, so a startup probe waits for a slow
    load instead of timing out on a closed port, and a revision whose model
    cannot load never takes traffic from the one already serving.
    """
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="degraded" if state.load_failed else "loading",
        )
    return HealthResponse(status="ok", model_loaded=True)


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
