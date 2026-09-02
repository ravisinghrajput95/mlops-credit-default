"""API authentication.

The interesting assertions are the ones about failure: that a misconfiguration
stops the process rather than opening the door, that a bad key is reported as an
authentication failure rather than an authorisation one, and that no future
endpoint can be added without a deliberate decision about who may call it.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from credit_default.api import main as api_main
from credit_default.api.auth import (
    MIN_KEY_LENGTH,
    ApiKey,
    Authenticator,
    build_authenticator,
    parse_api_keys,
)
from credit_default.api.model import ModelHandle
from credit_default.api.sinks import NullSink
from credit_default.config import FEATURES, TARGET, get_settings
from credit_default.features.pipeline import build_pipeline
from tests.conftest import make_frame
from tests.test_api import RecordingSink

GOOD = "T5nQb2kZ8vJw3yR7pLxA6mCdHfGsEu91"  # 32 chars, well over the floor
OTHER = "Z9aB8cD7eF6gH5iJ4kL3mN2oP1qR0sTu"

# Endpoints that must stay reachable without a credential, and why. Prometheus
# scrapes /metrics with no auth, Cloud Run's liveness probe hits /health and its
# startup probe hits /ready -- putting a key in front of any of them turns a
# monitoring gap into an outage, or a probe failure into a crash loop.
PUBLIC_PATHS = {
    "/health",
    "/ready",
    "/metrics",
    "/",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/openapi.json",
}


@pytest.fixture
def authenticated_client(monkeypatch):
    """The app with authentication switched on and one caller configured."""
    frame = make_frame()
    pipeline = build_pipeline(
        XGBClassifier(n_estimators=40, max_depth=3, eval_metric="logloss", random_state=0)
    )
    pipeline.fit(frame[FEATURES], frame[TARGET])

    monkeypatch.setattr(api_main.state, "model", ModelHandle(pipeline, "test-1", "local"))
    monkeypatch.setattr(api_main.state, "sink", NullSink())
    monkeypatch.setattr(
        api_main.state, "auth", Authenticator([ApiKey("batch-scoring", GOOD)], required=True)
    )
    with TestClient(api_main.app) as client:
        # The model loads on a worker thread now; let that attempt finish before
        # injecting the test model, so the loader cannot race the assignment.
        api_main.state.load_complete.wait(timeout=30)
        monkeypatch.setattr(api_main.state, "model", ModelHandle(pipeline, "test-1", "local"))
        monkeypatch.setattr(api_main.state, "sink", NullSink())
        monkeypatch.setattr(
            api_main.state, "auth", Authenticator([ApiKey("batch-scoring", GOOD)], required=True)
        )
        yield client


# ------------------------------------------------------------ parsing ------


def test_named_keys_are_parsed():
    keys = parse_api_keys(f"analytics:{GOOD},batch:{OTHER}")
    assert [k.name for k in keys] == ["analytics", "batch"]
    assert keys[0].secret == GOOD


def test_a_bare_secret_is_accepted_but_unnamed():
    """Permitted, because refusing to start over a missing label is a poor trade.
    The log then says no more than 'someone with a valid key', which is the
    reason to bother naming them."""
    assert parse_api_keys(GOOD) == [ApiKey("unnamed", GOOD)]


def test_whitespace_and_empty_entries_are_tolerated():
    assert len(parse_api_keys(f"  a:{GOOD} , , b:{OTHER} ,")) == 2


def test_a_short_key_is_rejected_with_a_way_to_generate_one():
    with pytest.raises(ValueError, match=f"minimum is {MIN_KEY_LENGTH}") as caught:
        parse_api_keys("dev:test")
    assert "token_urlsafe" in str(caught.value)


def test_duplicate_caller_names_are_rejected():
    """Two callers sharing a name make the prediction log ambiguous, which
    defeats the reason the names exist."""
    with pytest.raises(ValueError, match="Duplicate"):
        parse_api_keys(f"batch:{GOOD},batch:{OTHER}")


# ------------------------------------------------------- fail closed -------


def test_demanding_auth_without_keys_refuses_to_construct():
    """The central safety property: no keys must not silently mean no checking."""
    with pytest.raises(ValueError, match="Refusing to start"):
        Authenticator([], required=True)


def test_the_app_refuses_to_start_when_auth_is_demanded_without_keys(monkeypatch):
    """Asserted through the real lifespan, not just the unit: the failure has to
    actually stop the process, which is the part that protects anything."""
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setenv("API_KEYS", "")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="Refusing to start"), TestClient(api_main.app):
            pass  # pragma: no cover - the context manager raises on entry
    finally:
        get_settings.cache_clear()


def test_auth_off_is_announced_loudly(caplog):
    """Off by default is a real trade; it is never made silently."""
    with caplog.at_level(logging.WARNING):
        build_authenticator("", required=False)
    assert "AUTHENTICATION IS OFF" in caplog.text


def test_configured_caller_names_are_logged_but_secrets_are_not(caplog):
    with caplog.at_level(logging.INFO):
        build_authenticator(f"analytics:{GOOD}", required=True)
    assert "analytics" in caplog.text
    assert GOOD not in caplog.text


# ---------------------------------------------------------- rejection ------


def test_a_missing_key_is_401_with_a_challenge():
    authenticator = Authenticator([ApiKey("batch", GOOD)], required=True)
    with pytest.raises(HTTPException) as caught:
        authenticator.identify(None)
    assert caught.value.status_code == 401
    assert caught.value.headers["WWW-Authenticate"] == "Bearer"


def test_an_unrecognised_key_is_401_and_never_403():
    """403 means 'we know who you are and you may not'. There is no authorisation
    layer here, so every rejection is a failure to authenticate."""
    authenticator = Authenticator([ApiKey("batch", GOOD)], required=True)
    with pytest.raises(HTTPException) as caught:
        authenticator.identify(OTHER)
    assert caught.value.status_code == 401


def test_a_rejected_key_is_never_written_to_the_log(caplog):
    authenticator = Authenticator([ApiKey("batch", GOOD)], required=True)
    with caplog.at_level(logging.WARNING), pytest.raises(HTTPException):
        authenticator.identify(OTHER)
    assert OTHER not in caplog.text


def test_a_valid_key_resolves_to_its_caller_name():
    authenticator = Authenticator(
        [ApiKey("analytics", GOOD), ApiKey("batch", OTHER)], required=True
    )
    assert authenticator.identify(OTHER) == "batch"


def test_auth_off_identifies_everyone_as_anonymous():
    """A value rather than a null, so the prediction log always has a caller
    column and nothing downstream special-cases its absence."""
    assert Authenticator([], required=False).identify(None) == "anonymous"


# ---------------------------------------------------------- endpoints ------


def test_predict_without_a_key_is_rejected(authenticated_client, application):
    response = authenticated_client.post("/predict", json={"applications": [application]})
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_predict_with_a_valid_key_succeeds(authenticated_client, application):
    response = authenticated_client.post(
        "/predict",
        json={"applications": [application]},
        headers={"Authorization": f"Bearer {GOOD}"},
    )
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1


def test_predict_with_a_wrong_key_is_rejected(authenticated_client, application):
    response = authenticated_client.post(
        "/predict",
        json={"applications": [application]},
        headers={"Authorization": f"Bearer {OTHER}"},
    )
    assert response.status_code == 401


def test_model_info_is_protected(authenticated_client):
    assert authenticated_client.get("/model-info").status_code == 401
    assert (
        authenticated_client.get(
            "/model-info", headers={"Authorization": f"Bearer {GOOD}"}
        ).status_code
        == 200
    )


def test_health_stays_open_because_probes_use_it(authenticated_client):
    """Cloud Run's liveness probe hits /health. Authenticating it turns a
    deployment into a crash loop."""
    assert authenticated_client.get("/health").status_code == 200


def test_readiness_stays_open_because_the_startup_probe_uses_it(authenticated_client):
    """Cloud Run's startup probe hits /ready before any caller could hold a key.

    A credential on this endpoint means the revision never becomes ready, which
    presents as a failed deploy rather than as an authentication error -- the
    single least debuggable way for this to go wrong.
    """
    assert authenticated_client.get("/ready").status_code == 200


def test_metrics_stays_open_because_prometheus_scrapes_it(authenticated_client):
    assert authenticated_client.get("/metrics").status_code == 200


def test_the_caller_is_recorded_against_every_decision(
    authenticated_client, application, monkeypatch
):
    """Attribution, not just admission: a credit decision should record who
    asked for it, next to the score and the reasons."""
    sink = RecordingSink()
    monkeypatch.setattr(api_main.state, "sink", sink)

    authenticated_client.post(
        "/predict",
        json={"applications": [application]},
        headers={"Authorization": f"Bearer {GOOD}"},
    )
    assert sink.records[0]["caller"] == "batch-scoring"


def test_every_route_is_either_deliberately_public_or_authenticated():
    """The guard against the next endpoint. Adding a route without deciding who
    may call it should fail here rather than ship open."""

    def uses_auth(dependant) -> bool:
        if dependant.call is api_main.require_caller:
            return True
        return any(uses_auth(sub) for sub in dependant.dependencies)

    unprotected = [
        route.path
        for route in api_main.app.routes
        if isinstance(route, APIRoute)
        and route.path not in PUBLIC_PATHS
        and not uses_auth(route.dependant)
    ]
    assert not unprotected, f"routes neither public nor authenticated: {unprotected}"
