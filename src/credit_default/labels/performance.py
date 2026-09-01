"""Scoring the model on labels that actually arrived.

This is the loop the drift monitor exists to substitute for. Drift is the signal
available *now*; this is the truth, available a quarter late. Both are needed,
and neither replaces the other.

The hard part is not computing a metric. It is that **production labels are not a
random sample of applicants** -- the model chose which ones you get to see. Score
only the approvals and you are measuring the model on the population it already
decided was safe, over a truncated range of its own output. The number that comes
back is lower than the offline test number, every time, and it is not a
regression. Reading it as one is how a healthy model gets retrained out of
existence.

Three estimates are reported side by side because the comparison is the point:

``observed``
    Approved applicants only, unweighted. What a naive "delayed accuracy"
    dashboard shows. Biased, and biased downward.
``ipw``
    Every observed applicant, weighted by the inverse of the probability they
    were observable at all. Approvals were certain to be observed; holdout
    approvals had a known small chance. Horvitz-Thompson reweighting therefore
    reconstructs the full applicant population from a sample whose selection
    mechanism is known exactly -- which it is, because we built it.
``full``
    Everyone, using ground truth for applicants who were declined. **Not
    available in production** -- a declined applicant never opens an account, so
    the counterfactual outcome does not exist. It is computable here only because
    the demo replays a labelled dataset, and it is reported purely to check
    whether ``ipw`` recovers it. Treat it as the answer key, not as a metric.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)


def selection_probability(predictions: pd.Series, holdout_fraction: float) -> np.ndarray:
    """Probability each applicant's outcome becomes observable.

    Derived from the served decision alone, which is what makes the correction
    usable in production: approved (``prediction == 0``) applicants are always
    observed; declined applicants only via the random holdout, at a rate the
    policy itself sets.
    """
    if not 0.0 < holdout_fraction <= 1.0:
        raise ValueError(
            f"holdout_fraction must be in (0, 1], got {holdout_fraction}. Without a "
            "holdout the declined range has selection probability zero and no "
            "reweighting can recover it."
        )
    return np.where(predictions.to_numpy() == 1, holdout_fraction, 1.0)


@dataclass
class PerformanceEstimate:
    """One retrospective performance estimate, with the caveats attached."""

    name: str
    rows: int
    effective_rows: float
    positive_rate: float
    pr_auc: float
    roc_auc: float
    brier: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "effective_rows": round(self.effective_rows, 1),
            "positive_rate": round(self.positive_rate, 4),
            "pr_auc": round(self.pr_auc, 4),
            "roc_auc": round(self.roc_auc, 4),
            "brier": round(self.brier, 4),
        }


def _effective_rows(weights: np.ndarray) -> float:
    """Kish effective sample size.

    Reweighting buys back the missing population but spends precision doing it:
    a handful of rows carrying a weight of 50 contribute far less information
    than their raw count suggests. Reporting this alongside the estimate is what
    keeps a confident-looking IPW number honest.
    """
    total = float(weights.sum())
    return total * total / float((weights**2).sum()) if total else 0.0


def estimate(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray | None = None,
    name: str = "",
) -> PerformanceEstimate:
    """Weighted retrospective metrics; NaN where a metric is undefined."""
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=float)
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    single_class = len(np.unique(labels)) < 2
    if single_class:
        logger.warning("'%s' has a single outcome class; ranking metrics are undefined.", name)

    return PerformanceEstimate(
        name=name,
        rows=len(labels),
        effective_rows=_effective_rows(weights),
        positive_rate=float(np.average(labels, weights=weights)) if len(labels) else float("nan"),
        pr_auc=(
            float("nan")
            if single_class
            else float(average_precision_score(labels, probabilities, sample_weight=weights))
        ),
        roc_auc=(
            float("nan")
            if single_class
            else float(roc_auc_score(labels, probabilities, sample_weight=weights))
        ),
        brier=(
            float(brier_score_loss(labels, probabilities, sample_weight=weights))
            if len(labels)
            else float("nan")
        ),
    )


def bootstrap_std_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    draws: int = 200,
    seed: int = 0,
) -> float:
    """Standard error of the weighted PR-AUC, by resampling rows with replacement.

    Needed because the IPW estimate leans on a small holdout: the point estimate
    can look reassuringly close to the truth while being far too noisy to act on.
    An estimate without a spread is not evidence, it is a coincidence waiting to
    be quoted.
    """
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        index = rng.integers(0, len(labels), len(labels))
        sample = labels[index]
        if len(np.unique(sample)) < 2:
            continue
        values.append(
            average_precision_score(sample, probabilities[index], sample_weight=weights[index])
        )
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def compare(
    joined: pd.DataFrame,
    holdout_fraction: float,
    truth: pd.DataFrame | None = None,
    seed: int = 0,
) -> dict[str, PerformanceEstimate]:
    """Produce the observed / IPW / full comparison from a point-in-time join.

    ``joined`` is the output of ``backfill.join_labels``; ``truth`` is the full
    labelled population, supplied only in the demo to compute the answer key.
    """
    estimates: dict[str, PerformanceEstimate] = {}

    approved = joined[joined["prediction"] == 0]
    estimates["observed"] = estimate(
        approved["label"].to_numpy(),
        approved["probability"].to_numpy(),
        name="observed (approved only, unweighted)",
    )

    weights = 1.0 / selection_probability(joined["prediction"], holdout_fraction)
    estimates["ipw"] = estimate(
        joined["label"].to_numpy(),
        joined["probability"].to_numpy(),
        weights,
        name="IPW-corrected (holdout reweighted)",
    )

    if truth is not None:
        estimates["full"] = estimate(
            truth["label"].to_numpy(),
            truth["probability"].to_numpy(),
            name="full population (answer key, not observable in production)",
        )

    return estimates
