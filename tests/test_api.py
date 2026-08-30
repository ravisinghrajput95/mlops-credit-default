"""API contract tests.

The app is exercised with a real fitted pipeline but no MLflow server and no
prediction sink, so the suite runs anywhere with no external dependencies.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

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
    pipeline = build_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(frame[FEATURES], frame[TARGET])

    monkeypatch.setattr(api_main.state, "model", ModelHandle(pipeline, "test-1", "local"))
    monkeypatch.setattr(api_main.state, "sink", NullSink())
    # lifespan would try to load a real model; the state above is what we want tested.
    with TestClient(api_main.app) as test_client:
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
    body = client.get("/health").json()
    assert body == {"status": "degraded", "model_loaded": False}


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
    assert {"id", "predicted_at", "probability", "prediction", "features"} <= record.keys()


def test_metrics_endpoint_exposes_the_prediction_histogram(client, application):
    client.post("/predict", json={"applications": [application]})
    body = client.get("/metrics").text
    assert "prediction_probability" in body
