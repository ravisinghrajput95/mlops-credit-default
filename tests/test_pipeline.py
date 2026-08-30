"""The fused preprocessing/estimator pipeline is what prevents train/serve skew."""

from __future__ import annotations

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from credit_default.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET
from credit_default.features.pipeline import build_pipeline, build_preprocessor


def test_preprocessor_drops_unexpected_columns(frame):
    """A stray column must never reach the estimator."""
    polluted = frame.copy()
    polluted["leaked_target_proxy"] = frame[TARGET]

    preprocessor = build_preprocessor().fit(polluted)
    names = list(preprocessor.get_feature_names_out())
    assert "leaked_target_proxy" not in names


def test_pipeline_is_self_contained(frame):
    """Raw input goes in; no separate preprocessing step is needed at inference."""
    pipeline = build_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(frame[FEATURES], frame[TARGET])

    # Deliberately passing raw, untransformed rows.
    probabilities = pipeline.predict_proba(frame[FEATURES].head(5))[:, 1]
    assert len(probabilities) == 5
    assert all(0.0 <= p <= 1.0 for p in probabilities)


def test_unseen_category_does_not_raise(frame):
    """handle_unknown='ignore' -- production sending a novel code must degrade, not 500."""
    pipeline = build_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(frame[FEATURES], frame[TARGET])

    novel = frame[FEATURES].head(1).copy()
    novel.loc[:, "PAY_0"] = 99  # a repayment code never seen in training

    # sklearn warns rather than raising, and encodes the unknown level as zeros.
    with pytest.warns(UserWarning, match="unknown categories"):
        assert pipeline.predict_proba(novel).shape == (1, 2)


def test_pipeline_learns_the_planted_signal(frame):
    """Sanity check that the pipeline can actually fit; guards against silent breakage."""
    train, test = frame.iloc[:300], frame.iloc[300:]
    pipeline = build_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(train[FEATURES], train[TARGET])

    auc = roc_auc_score(test[TARGET], pipeline.predict_proba(test[FEATURES])[:, 1])
    assert auc > 0.7, f"expected the planted PAY_0 signal to be learnable, got AUC={auc:.3f}"


def test_feature_lists_are_disjoint_and_complete():
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert sorted(FEATURES) == sorted(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert TARGET not in FEATURES, "the target must never appear as a feature"
