"""The search must actually search, and must not touch the test set."""

from __future__ import annotations

import json

import pytest

from credit_default.config import Settings
from credit_default.train import _tuned_params, candidates
from credit_default.tuning import SearchResult


@pytest.fixture
def settings(tmp_path):
    s = Settings(data_dir=tmp_path, reports_dir=tmp_path)
    s.ensure_dirs()
    return s


def test_defaults_are_used_when_no_search_has_run(settings):
    """A clean clone must train without first spending minutes on a search."""
    assert _tuned_params(settings) is None


def test_tuned_parameters_are_picked_up_when_present(settings):
    payload = {"best_params": {"max_depth": 7, "n_estimators": 300}}
    (settings.reports_dir / "best_params.json").write_text(json.dumps(payload))
    assert _tuned_params(settings) == {"max_depth": 7, "n_estimators": 300}


def test_a_malformed_file_is_ignored_rather_than_crashing(settings):
    (settings.reports_dir / "best_params.json").write_text("{not json")
    assert _tuned_params(settings) is None


def test_candidates_record_whether_they_were_tuned():
    untuned = {c.name: c for c in candidates(0)}["xgboost"]
    tuned = {c.name: c for c in candidates(0, {"max_depth": 7})}["xgboost"]
    assert untuned.params["tuned"] is False
    assert tuned.params["tuned"] is True
    assert tuned.params["max_depth"] == 7


def test_search_result_reports_improvement_against_the_baseline():
    result = SearchResult(best_params={}, best_score=0.56, baseline_score=0.55, n_trials=10)
    assert result.improvement == pytest.approx(0.01)
    assert result.to_dict()["improvement"] == pytest.approx(0.01)


def test_a_search_that_finds_nothing_reports_zero_not_a_failure():
    """Tuning that buys nothing is a valid, reportable outcome."""
    result = SearchResult(best_params={}, best_score=0.55, baseline_score=0.55, n_trials=10)
    assert result.improvement == 0.0
