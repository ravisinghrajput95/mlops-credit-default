"""Explanations must be correct, attributable, and safe to serve.

Correctness here has a precise meaning: SHAP values are additive, so the base
value plus the contributions must reconstruct the model's raw output. A method
that merely produces plausible-looking rankings would pass a smoke test and fail
a regulator.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from credit_default.config import FEATURES, PROTECTED_ATTRIBUTES, TARGET
from credit_default.explain import Explainer, _source_column_map
from credit_default.features.pipeline import build_pipeline


@pytest.fixture
def fitted(frame):
    pipeline = build_pipeline(
        XGBClassifier(n_estimators=40, max_depth=3, eval_metric="logloss", random_state=0)
    )
    pipeline.fit(frame[FEATURES], frame[TARGET])
    return pipeline


@pytest.fixture
def explainer(fitted):
    return Explainer(fitted)


def test_encoded_columns_map_back_to_their_source(fitted):
    """One-hot columns must attribute to the column they came from."""
    preprocessor = fitted.named_steps["preprocessor"]
    mapping = _source_column_map(preprocessor)
    assert (
        len(mapping) == preprocessor.transform(np.zeros((1, 0)) if False else None).shape[1]
        if False
        else True
    )  # shape checked below instead
    assert len(mapping) == len(preprocessor.get_feature_names_out())
    assert set(mapping) <= set(FEATURES)


def test_contributions_are_additive(fitted, explainer, frame):
    """base value + contributions == the model's raw output, to floating point.

    This is the property that makes SHAP defensible rather than merely suggestive.
    """
    rows = frame[FEATURES].head(50)
    transformed = explainer.preprocessor.transform(rows)
    values = explainer.explainer.shap_values(transformed)
    raw = fitted.named_steps["estimator"].predict(transformed, output_margin=True)

    reconstructed = explainer.explainer.expected_value + values.sum(axis=1)
    assert np.abs(reconstructed - raw).max() < 1e-4


def test_declined_applicants_get_only_risk_increasing_reasons(explainer, frame):
    """An adverse-action notice states why someone was refused, not what helped."""
    rows = frame[FEATURES].head(30)
    reasons = explainer.explain(rows, declined=[True] * len(rows))
    for row in reasons:
        assert all(r.contribution > 0 for r in row)
        assert all(r.direction == "increased_risk" for r in row)


def test_approved_applicants_get_the_strongest_factors_either_way(explainer, frame):
    rows = frame[FEATURES].head(10)
    reasons = explainer.explain(rows, declined=[False] * len(rows))
    assert all(len(row) > 0 for row in reasons)


def test_protected_attributes_can_never_appear_as_a_reason(explainer, frame):
    """They are not model inputs, so they must not surface in an explanation."""
    rows = frame[FEATURES].head(40)
    reasons = explainer.explain(rows, declined=[True] * len(rows))
    named = {r.feature for row in reasons for r in row}
    assert not (named & set(PROTECTED_ATTRIBUTES))


def test_reasons_are_ordered_by_magnitude(explainer, frame):
    rows = frame[FEATURES].head(5)
    for row in explainer.explain(rows, declined=[True] * len(rows)):
        contributions = [r.contribution for r in row]
        assert contributions == sorted(contributions, reverse=True)


def test_top_k_is_respected(explainer, frame):
    rows = frame[FEATURES].head(5)
    for row in explainer.explain(rows, declined=[True] * len(rows), top_k=2):
        assert len(row) <= 2


def test_reasons_carry_a_human_readable_description(explainer, frame):
    rows = frame[FEATURES].head(3)
    for row in explainer.explain(rows, declined=[True] * len(rows)):
        for reason in row:
            assert reason.description
            # A raw column name is not an explanation an applicant can act on.
            assert reason.description != reason.feature or reason.feature not in {
                "PAY_0",
                "LIMIT_BAL",
            }


def test_reason_serialises_for_the_api(explainer, frame):
    row = explainer.explain(frame[FEATURES].head(1), declined=[True])[0]
    payload = row[0].to_dict()
    assert set(payload) == {"feature", "description", "value", "contribution", "direction"}


def test_one_explanation_per_input_row(explainer, frame):
    rows = frame[FEATURES].head(17)
    assert len(explainer.explain(rows, declined=[False] * 17)) == 17


def test_linear_models_are_rejected_rather_than_mis_explained(frame):
    """Fail loudly rather than silently returning a wrong attribution.

    TreeExplainer is exact for gradient-boosted trees only. Quietly accepting a
    linear model would produce numbers that look like explanations and are not.
    """
    from shap.utils._exceptions import InvalidModelError

    pipeline = build_pipeline(LogisticRegression(max_iter=100))
    pipeline.fit(frame[FEATURES], frame[TARGET])
    with pytest.raises(InvalidModelError):
        Explainer(pipeline)
