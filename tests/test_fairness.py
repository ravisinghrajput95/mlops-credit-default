"""Fairness auditing must measure real disparities and survive edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_default.config import PROTECTED_ATTRIBUTES, model_features
from credit_default.fairness import audit, audit_attribute, band_age, flatten_for_mlflow


def test_protected_attributes_are_excluded_by_default():
    numeric, categorical = model_features()
    for attribute in PROTECTED_ATTRIBUTES:
        assert attribute not in numeric
        assert attribute not in categorical


def test_protected_attributes_can_be_included_deliberately():
    """The flag exists so the cost of exclusion can be measured, not asserted."""
    numeric, categorical = model_features(use_protected=True)
    assert "AGE" in numeric
    assert "SEX" in categorical


def test_estimator_never_sees_protected_attributes(frame):
    """The strong guarantee: they are in the input frame but dropped before the model."""
    from sklearn.linear_model import LogisticRegression

    from credit_default.config import FEATURES, TARGET
    from credit_default.features.pipeline import build_pipeline

    pipeline = build_pipeline(LogisticRegression(max_iter=300))
    pipeline.fit(frame[FEATURES], frame[TARGET])

    encoded = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    for attribute in PROTECTED_ATTRIBUTES:
        assert not any(name.startswith(attribute) for name in encoded)


def test_a_biased_model_is_detected():
    """A model that approves only one group must show a large selection gap."""
    frame = pd.DataFrame({"SEX": [1] * 50 + [2] * 50})
    y_true = pd.Series([0, 1] * 50)
    # Predicts default for every member of group 2 and nobody in group 1.
    probabilities = np.array([0.1] * 50 + [0.9] * 50)

    result = audit_attribute(frame, "SEX", y_true, (probabilities >= 0.5).astype(int))
    assert result.selection_rate_gap == pytest.approx(1.0)


def test_an_even_handed_model_shows_no_gap():
    """The negative control: identical treatment must report a zero gap."""
    frame = pd.DataFrame({"SEX": [1, 2] * 50})
    y_true = pd.Series([0, 1] * 50)
    probabilities = np.full(100, 0.9)

    result = audit_attribute(frame, "SEX", y_true, (probabilities >= 0.5).astype(int))
    assert result.selection_rate_gap == pytest.approx(0.0)


def test_base_rate_gap_separates_real_difference_from_model_bias():
    """Groups that genuinely differ should show it, so disparity is not misread."""
    frame = pd.DataFrame({"SEX": [1] * 50 + [2] * 50})
    y_true = pd.Series([0] * 50 + [1] * 50)  # group 2 always defaults
    probabilities = np.array([0.1] * 50 + [0.9] * 50)

    result = audit_attribute(frame, "SEX", y_true, (probabilities >= 0.5).astype(int))
    assert result.base_rate_gap == pytest.approx(1.0)


def test_age_is_banded_before_auditing():
    banded = band_age(pd.Series([21, 35, 50, 70]))
    assert list(banded.astype(str)) == ["18-30", "30-45", "45-60", "60-120"]


def test_audit_covers_every_requested_attribute(frame):
    probabilities = np.random.default_rng(0).random(len(frame))
    result = audit(frame, frame["default_payment_next_month"], probabilities, ["SEX", "AGE"])
    assert set(result) == {"SEX", "AGE"}


def test_audit_skips_attributes_absent_from_the_frame(frame):
    probabilities = np.random.default_rng(0).random(len(frame))
    result = audit(frame, frame["default_payment_next_month"], probabilities, ["NOT_A_COLUMN"])
    assert result == {}


def test_single_group_reports_no_gap():
    """An attribute with one observed value cannot have a disparity."""
    frame = pd.DataFrame({"SEX": [1] * 20})
    y_true = pd.Series([0, 1] * 10)
    result = audit_attribute(frame, "SEX", y_true, np.ones(20, dtype=int))
    assert result.selection_rate_gap == 0.0


def test_metrics_flatten_to_scalars_for_tracking(frame):
    probabilities = np.random.default_rng(0).random(len(frame))
    audits = audit(frame, frame["default_payment_next_month"], probabilities, ["SEX"])
    flat = flatten_for_mlflow(audits)
    assert "fairness_SEX_selection_gap" in flat
    assert all(isinstance(v, float) for v in flat.values())
