"""Choosing the decision threshold from business cost, not from convention.

A classifier outputs a probability; turning that into an approve/decline decision
needs a cutoff. The default of 0.5 is not a neutral choice — it is the optimal
cutoff only when a false positive and a false negative cost exactly the same.

In lending they differ by roughly an order of magnitude:

* A **false negative** — approving someone who then defaults — loses a large part
  of the outstanding balance.
* A **false positive** — declining someone who would have repaid — loses the
  margin on one account, and some goodwill.

So 0.5 systematically under-declines. The cutoff here is chosen by minimising
expected cost over a grid, using the ratio of those two costs.

The threshold is tuned on **out-of-fold predictions over the training set**, never
on the test set. Tuning on test would make the reported test metrics optimistic:
the cutoff would have been fitted to the same data used to judge it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Candidate cutoffs. Finer than the precision the data can justify, but cheap.
GRID = np.round(np.arange(0.05, 0.95, 0.01), 2)


@dataclass
class ThresholdChoice:
    threshold: float
    expected_cost: float
    cost_at_half: float
    approvals_declined: float  # share of applicants declined at this cutoff

    def to_dict(self) -> dict[str, float]:
        return {
            "decision_threshold": round(self.threshold, 4),
            "expected_cost_per_applicant": round(self.expected_cost, 4),
            "expected_cost_at_0.5": round(self.cost_at_half, 4),
            "cost_reduction_vs_0.5": round(self.cost_at_half - self.expected_cost, 4),
            "decline_rate": round(self.approvals_declined, 4),
        }


def expected_cost(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    cost_false_negative: float,
    cost_false_positive: float,
) -> float:
    """Mean cost per applicant at a given cutoff, in the caller's cost units."""
    predicted = probabilities >= threshold
    actual = y_true == 1

    false_negatives = int((~predicted & actual).sum())
    false_positives = int((predicted & ~actual).sum())
    total = cost_false_negative * false_negatives + cost_false_positive * false_positives
    return float(total / len(y_true))


def choose_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    cost_false_negative: float,
    cost_false_positive: float,
) -> ThresholdChoice:
    """Pick the cutoff minimising expected cost."""
    costs = [
        expected_cost(y_true, probabilities, t, cost_false_negative, cost_false_positive)
        for t in GRID
    ]
    best_index = int(np.argmin(costs))
    best = float(GRID[best_index])

    return ThresholdChoice(
        threshold=best,
        expected_cost=costs[best_index],
        cost_at_half=expected_cost(
            y_true, probabilities, 0.5, cost_false_negative, cost_false_positive
        ),
        approvals_declined=float((probabilities >= best).mean()),
    )


def tune_on_training_data(
    pipeline: Pipeline,
    train: pd.DataFrame,
    features: list[str],
    target: str,
    cost_false_negative: float,
    cost_false_positive: float,
    seed: int,
    folds: int = 5,
) -> ThresholdChoice:
    """Tune the cutoff on out-of-fold training predictions.

    ``cross_val_predict`` gives every training row a prediction from a model that
    did not see it, so the probabilities are honest without spending the test set.
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out_of_fold = cross_val_predict(
        clone(pipeline),
        train[features],
        train[target],
        cv=splitter,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    choice = choose_threshold(
        train[target].to_numpy(), out_of_fold, cost_false_negative, cost_false_positive
    )
    logger.info(
        "Cost-optimal threshold %.2f (expected cost %.4f vs %.4f at 0.50, %.1f%% declined)",
        choice.threshold,
        choice.expected_cost,
        choice.cost_at_half,
        choice.approvals_declined * 100,
    )
    return choice
