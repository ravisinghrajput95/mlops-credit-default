"""Drift detection must fire on real shifts and stay quiet otherwise.

A monitor that always alerts is ignored; a monitor that never alerts is useless.
Both directions are tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from credit_default.config import Settings
from credit_default.monitoring.drift import run, summarise
from tests.conftest import make_frame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from simulate_drift import apply_drift

# Evidently switches to small-sample statistical tests below ~1000 rows, which are
# noisy enough to flag untouched columns. Drift tests use a realistic sample size so
# they exercise the distance-based path that production actually uses.
DRIFT_TEST_ROWS = 3000


@pytest.fixture
def big_frame():
    return make_frame(rows=DRIFT_TEST_ROWS, seed=7)


@pytest.fixture
def settings(tmp_path, big_frame):
    """A settings object backed by a temporary reference/current pair."""
    settings = Settings(data_dir=tmp_path, reports_dir=tmp_path, max_drifted_share=0.3)
    settings.ensure_dirs()
    big_frame.to_parquet(settings.reference_parquet, index=False)
    big_frame.to_parquet(settings.current_parquet, index=False)
    return settings


def test_identical_cohorts_report_no_drift(settings):
    """The negative control: comparing data to itself must not alert."""
    summary = run("cohort", settings)
    assert summary["drifted_columns"] == 0
    assert summary["passed"] is True


def test_injected_drift_is_detected(settings, big_frame):
    drifted = apply_drift(big_frame, severity=0.9, seed=1)
    drifted.to_parquet(settings.current_parquet, index=False)

    summary = run("cohort", settings)
    assert summary["drifted_columns"] > 0
    assert summary["passed"] is False


def test_the_shifted_columns_are_the_ones_flagged(settings, big_frame):
    """Detection must be specific, not just a global alarm."""
    drifted = apply_drift(big_frame, severity=0.9, seed=1)
    drifted.to_parquet(settings.current_parquet, index=False)

    flagged = {n for n, i in run("cohort", settings)["columns"].items() if i["drifted"]}
    assert "LIMIT_BAL" in flagged
    assert flagged & {"PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"}
    # AGE and SEX were never touched by the simulator.
    assert "AGE" not in flagged
    assert "SEX" not in flagged


def test_a_report_is_written_for_humans(settings):
    run("cohort", settings)
    assert Path(settings.reports_dir / "drift_report.html").exists()
    assert Path(settings.reports_dir / "drift_summary.json").exists()


def test_summarise_handles_an_empty_report():
    summary = summarise({"metrics": []})
    assert summary["drifted_columns"] == 0
    assert summary["monitored_columns"] == 0


def test_unknown_source_is_rejected(settings):
    with pytest.raises(ValueError, match="Unknown current-data source"):
        run("nowhere", settings)
