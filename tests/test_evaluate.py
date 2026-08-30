"""The quality gate's decision logic.

Registry interaction is covered by the compose-stack integration run; these tests
pin the decision rules themselves, which are what actually block a bad release.
"""

from __future__ import annotations

import pytest

from credit_default.config import Settings
from credit_default.evaluate import GateResult, evaluate


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, reports_dir=tmp_path, min_pr_auc=0.5)


def _stub_aliases(monkeypatch, challenger, champion):
    """Bypass MLflow and feed the gate the two metric sets directly."""
    import credit_default.evaluate as module

    monkeypatch.setattr(module.pd, "read_parquet", lambda _: None)
    monkeypatch.setattr(module.mlflow, "set_tracking_uri", lambda _: None)
    monkeypatch.setattr(
        module,
        "_score_alias",
        lambda alias, test, settings: challenger if alias == "challenger" else champion,
    )


def test_missing_challenger_fails_closed(monkeypatch, settings):
    _stub_aliases(monkeypatch, None, None)
    result = evaluate(settings)
    assert not result.passed
    assert "No challenger" in result.reasons[0]


def test_first_model_passes_on_the_absolute_floor(monkeypatch, settings):
    _stub_aliases(monkeypatch, {"pr_auc": 0.56}, None)
    assert evaluate(settings).passed


def test_model_below_the_floor_is_rejected(monkeypatch, settings):
    """A prior-only classifier scores the base rate; it must never ship."""
    _stub_aliases(monkeypatch, {"pr_auc": 0.22}, None)
    result = evaluate(settings)
    assert not result.passed
    assert "absolute floor" in result.reasons[0]


def test_silent_regression_against_the_champion_is_caught(monkeypatch, settings):
    """Above the floor but clearly worse than the incumbent -- the subtle failure."""
    _stub_aliases(monkeypatch, {"pr_auc": 0.52}, {"pr_auc": 0.60})
    result = evaluate(settings)
    assert not result.passed
    assert "regressed" in result.reasons[0]


def test_regression_within_tolerance_is_allowed(monkeypatch, settings):
    """Run-to-run noise must not block every release."""
    _stub_aliases(monkeypatch, {"pr_auc": 0.595}, {"pr_auc": 0.60})
    assert evaluate(settings).passed


def test_improvement_passes(monkeypatch, settings):
    _stub_aliases(monkeypatch, {"pr_auc": 0.65}, {"pr_auc": 0.60})
    assert evaluate(settings).passed


def test_gate_result_serialises_for_ci_artifacts():
    payload = GateResult(True, [], {"pr_auc": 0.6}, None).to_dict()
    assert payload["passed"] is True
    assert payload["challenger"]["pr_auc"] == 0.6
