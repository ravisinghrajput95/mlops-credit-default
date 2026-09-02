"""API contract tests.

The app is exercised with a real fitted pipeline but no MLflow server and no
prediction sink, so the suite runs anywhere with no external dependencies.
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from credit_default.api import main as api_main
from credit_default.api.model import ModelHandle
from credit_default.api.sinks import NullSink, PredictionSink
from credit_default.config import FEATURES, TARGET
from credit_default.features.pipeline import build_pipeline
from tests.conftest import make_frame


class RecordingSink(PredictionSink):
    """Captures what the API would have persisted."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, records):
        self.records.extend(records)


@pytest.fixture
def client(monkeypatch):
    frame = make_frame()
    # XGBoost rather than a linear model: the champion is gradient-boosted, and
    # SHAP's TreeExplainer only supports tree models, so a linear stand-in would
    # exercise a different explanation path from the one that ships.
    pipeline = build_pipeline(
        XGBClassifier(n_estimators=40, max_depth=3, eval_metric="logloss", random_state=0)
    )
    pipeline.fit(frame[FEATURES], frame[TARGET])

    monkeypatch.setattr(api_main.state, "model", ModelHandle(pipeline, "test-1", "local"))
    monkeypatch.setattr(api_main.state, "sink", NullSink())
    # lifespan would try to load a real model; the state above is what we want tested.
    with TestClient(api_main.app) as test_client:
        # The model now loads on a worker thread, so the load attempt has to be
        # allowed to finish before the test model is injected -- otherwise the
        # loader is still running and races with the assignment below.
        api_main.state.load_complete.wait(timeout=30)
        monkeypatch.setattr(api_main.state, "model", ModelHandle(pipeline, "test-1", "local"))
        monkeypatch.setattr(api_main.state, "sink", NullSink())
        yield test_client


def test_health_reports_ok_when_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_reports_degraded_without_a_model(client, monkeypatch):
    """A missing model must surface as degraded, not as a crash loop."""
    monkeypatch.setattr(api_main.state, "model", None)
    monkeypatch.setattr(api_main.state, "load_failed", True)
    body = client.get("/health").json()
    assert body == {"status": "degraded", "model_loaded": False}


def test_health_distinguishes_a_slow_load_from_a_failed_one(client, monkeypatch):
    """ "Not loaded yet" and "tried and failed" must not look the same.

    They resolve differently -- one on its own, one never -- so an operator
    reading the dashboard needs to be able to tell them apart. Reporting both as
    "degraded" hid a healthy cold start inside the same signal as a broken model.
    """
    monkeypatch.setattr(api_main.state, "model", None)
    monkeypatch.setattr(api_main.state, "load_failed", False)
    assert client.get("/health").json() == {"status": "loading", "model_loaded": False}

    monkeypatch.setattr(api_main.state, "load_failed", True)
    assert client.get("/health").json() == {"status": "degraded", "model_loaded": False}


def test_readiness_and_liveness_disagree_while_the_model_is_still_loading(client, monkeypatch):
    """The whole point of splitting the two probes.

    /health answers 200 so the orchestrator does not kill a process that is
    merely still starting; /ready answers 503 so no traffic is routed to an
    instance that cannot serve it. Pointing both probes at /health -- which is
    what the deployment used to do -- makes those two answers impossible to give
    at the same time.
    """
    monkeypatch.setattr(api_main.state, "model", None)
    monkeypatch.setattr(api_main.state, "load_failed", False)

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503


def test_readiness_turns_green_once_the_model_is_loaded(client):
    assert client.get("/ready").status_code == 200
    assert client.get("/ready").json() == {"status": "ok", "model_loaded": True}


def test_startup_completes_before_the_model_finishes_loading(monkeypatch):
    """The defect this whole split exists to fix, asserted directly.

    uvicorn binds the listening socket only *after* lifespan startup returns, so
    loading the model on the startup path keeps the port closed for the entire
    load. Nothing is listening to report the degradation the design promises, and
    a health probe gets a refused connection instead of an answer. On Cloud Run
    that was 7m12s of closed socket and a deploy that failed with
    ERROR_CONNECTION_FAILED and no application log to explain it.

    So: startup must complete while the load is still in flight. If someone moves
    the load back onto the startup path, entering the context manager below will
    block until the loader is released and this test will time out rather than
    quietly regressing.
    """
    monkeypatch.setattr(api_main.state, "model", None)

    loader_entered = threading.Event()
    release_loader = threading.Event()

    def slow_load(settings):
        loader_entered.set()
        release_loader.wait(timeout=30)
        raise RuntimeError("deliberately never produces a model")

    monkeypatch.setattr(api_main, "load_model", slow_load)

    with TestClient(api_main.app) as test_client:
        assert loader_entered.wait(timeout=10), "the background loader never started"
        assert not api_main.state.load_complete.is_set(), (
            "startup waited for the load to finish, which is the bug: "
            "the port stays closed for as long as the load takes"
        )

        # The socket is open and answering while the model is still in flight,
        # which is exactly what the old arrangement could not do.
        assert test_client.get("/health").json() == {
            "status": "loading",
            "model_loaded": False,
        }
        assert test_client.get("/ready").status_code == 503

        release_loader.set()
        assert api_main.state.load_complete.wait(timeout=15)
        assert api_main.state.load_failed is True
        assert test_client.get("/health").json() == {
            "status": "degraded",
            "model_loaded": False,
        }


def test_model_info_lists_the_expected_features(client):
    body = client.get("/model-info").json()
    assert body["features"] == FEATURES
    assert body["model_version"] == "test-1"
    assert TARGET not in body["features"]


def test_predict_returns_a_probability_per_application(client, application):
    response = client.post("/predict", json={"applications": [application, application]})
    assert response.status_code == 200

    body = response.json()
    assert len(body["predictions"]) == 2
    for item in body["predictions"]:
        assert 0.0 <= item["probability"] <= 1.0
        assert item["prediction"] in (0, 1)


def test_prediction_label_agrees_with_threshold(client, application):
    body = client.post("/predict", json={"applications": [application]}).json()
    item = body["predictions"][0]
    assert item["prediction"] == int(item["probability"] >= body["threshold"])


@pytest.mark.parametrize(
    ("field", "value"),
    [("AGE", 5), ("SEX", 9), ("LIMIT_BAL", -1), ("PAY_0", 99), ("PAY_AMT1", -5)],
)
def test_malformed_input_is_rejected_with_422(client, application, field, value):
    """Bad input must be refused at the edge, not scored."""
    bad = {**application, field: value}
    assert client.post("/predict", json={"applications": [bad]}).status_code == 422


def test_missing_field_is_rejected(client, application):
    bad = {k: v for k, v in application.items() if k != "AGE"}
    assert client.post("/predict", json={"applications": [bad]}).status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/predict", json={"applications": []}).status_code == 422


def test_undocumented_education_code_is_accepted_and_cleaned(client, application):
    """EDUCATION=5 is undocumented but real; the API cleans it exactly as training did."""
    response = client.post("/predict", json={"applications": [{**application, "EDUCATION": 5}]})
    assert response.status_code == 200


def test_predictions_are_recorded_for_monitoring(client, application, monkeypatch):
    sink = RecordingSink()
    monkeypatch.setattr(api_main.state, "sink", sink)

    client.post("/predict", json={"applications": [application, application]})

    assert len(sink.records) == 2
    record = sink.records[0]
    assert {
        "id",
        "application_id",
        "predicted_at",
        "probability",
        "prediction",
        "features",
    } <= record.keys()


def test_the_application_id_is_echoed_so_an_outcome_can_be_reported_later(client, application):
    """The label pipeline lives or dies on this. An outcome arrives months after
    the score, and a caller who was never told the key has no way to report it."""
    body = client.post(
        "/predict", json={"applications": [{**application, "application_id": "APP-42"}]}
    ).json()
    assert body["predictions"][0]["application_id"] == "APP-42"


def test_an_application_id_is_generated_when_the_caller_omits_one(client, application):
    """Optional for the caller, never absent from the record: an unkeyed
    prediction is one that can never be learned from."""
    body = client.post("/predict", json={"applications": [application]}).json()
    assert body["predictions"][0]["application_id"]


def test_the_application_id_never_reaches_the_model(client, application, monkeypatch):
    """It identifies the decision; it is not evidence about the applicant.
    Letting an arbitrary key into the feature frame would be a plain leak."""
    sink = RecordingSink()
    monkeypatch.setattr(api_main.state, "sink", sink)

    client.post("/predict", json={"applications": [{**application, "application_id": "APP-7"}]})

    assert "application_id" not in sink.records[0]["features"]
    assert set(sink.records[0]["features"]) == set(FEATURES)


def test_two_applications_get_distinct_generated_keys(client, application):
    """Identical payloads are still two separate decisions with two outcomes."""
    body = client.post("/predict", json={"applications": [application, application]}).json()
    first, second = (p["application_id"] for p in body["predictions"])
    assert first != second


def test_metrics_endpoint_exposes_the_prediction_histogram(client, application):
    client.post("/predict", json={"applications": [application]})
    body = client.get("/metrics").text
    assert "prediction_probability" in body


@pytest.mark.parametrize("attribute", ["SEX", "MARRIAGE", "AGE"])
def test_prediction_is_invariant_to_protected_attributes(client, application, attribute):
    """Counterfactual test: change only a protected attribute, get the same score.

    This is the end-to-end proof that exclusion actually holds. Unit tests confirm
    the column never reaches the estimator; this confirms the whole serving path
    behaves accordingly, which is what an auditor would actually ask to see.
    """
    variants = {"SEX": [1, 2], "MARRIAGE": [1, 2, 3], "AGE": [25, 45, 65]}[attribute]

    probabilities = []
    for value in variants:
        body = {"applications": [{**application, attribute: value}]}
        response = client.post("/predict", json=body)
        assert response.status_code == 200
        probabilities.append(response.json()["predictions"][0]["probability"])

    assert len(set(probabilities)) == 1, (
        f"changing {attribute} changed the prediction: {probabilities}"
    )


def test_model_info_declares_what_the_model_may_not_use(client):
    body = client.get("/model-info").json()
    assert set(body["excluded_attributes"]) == {"SEX", "MARRIAGE", "AGE"}
    # The attributes stay in the request contract so they can still be audited.
    for attribute in body["excluded_attributes"]:
        assert attribute in body["features"]


def test_explanations_are_off_by_default(client, application):
    """They cost latency, so a bulk scoring caller should not pay for them."""
    body = client.post("/predict", json={"applications": [application]}).json()
    assert body["predictions"][0]["reasons"] is None


def test_explanations_are_returned_when_requested(client, application):
    response = client.post("/predict", json={"applications": [application], "explain": True})
    assert response.status_code == 200

    reasons = response.json()["predictions"][0]["reasons"]
    assert reasons, "explain=true must return reasons"
    for reason in reasons:
        assert reason["description"]
        assert reason["direction"] in ("increased_risk", "decreased_risk")


def test_a_declined_application_gets_adverse_action_reasons(client, application):
    """The reasons for a refusal must be the factors that raised risk."""
    delinquent = {**application, **{f"PAY_{i}": 2 for i in (0, 2, 3, 4, 5, 6)}}
    delinquent.update({f"PAY_AMT{i}": 0 for i in range(1, 7)})

    body = client.post("/predict", json={"applications": [delinquent], "explain": True}).json()
    prediction = body["predictions"][0]

    if prediction["prediction"] == 1:
        assert all(r["direction"] == "increased_risk" for r in prediction["reasons"])


def test_explanations_never_cite_a_protected_attribute(client, application):
    body = client.post("/predict", json={"applications": [application], "explain": True}).json()
    cited = {r["feature"] for r in body["predictions"][0]["reasons"]}
    assert not (cited & {"SEX", "MARRIAGE", "AGE"})


def test_one_set_of_reasons_per_application(client, application):
    body = client.post("/predict", json={"applications": [application] * 3, "explain": True}).json()
    assert len(body["predictions"]) == 3
    assert all(p["reasons"] for p in body["predictions"])


def test_an_unexplainable_model_still_serves_predictions(client, application, monkeypatch):
    """A failure to explain must not deny someone a decision.

    Explanations are best-effort: the caller gets reasons omitted rather than a
    500, because refusing to answer at all is the worse failure.
    """

    def boom(*_args, **_kwargs):
        raise RuntimeError("explainer unavailable")

    monkeypatch.setattr(api_main.state.model, "explain", boom)

    response = client.post("/predict", json={"applications": [application], "explain": True})
    assert response.status_code == 200
    assert response.json()["predictions"][0]["reasons"] is None
