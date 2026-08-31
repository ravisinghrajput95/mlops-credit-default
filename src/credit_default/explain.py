"""Per-decision explanations, for adverse-action reasons.

When a credit application is declined, ECOA and Regulation B require the lender
to tell the applicant the **principal reasons** — not a score, and not "our model
said so". A model that cannot explain an individual decision is not deployable in
consumer lending regardless of how well it ranks.

SHAP gives an exact additive attribution for tree models: each feature's
contribution to this prediction, summing to the difference between the model's
output and its base value. That "sums to the prediction" property is what makes
it defensible to a regulator, as opposed to a global importance ranking that says
nothing about the individual in front of you.

Two things this module has to get right:

**Attribute back to source columns, not encoded ones.** The model sees
``PAY_0_2``; the applicant needs to hear "your recent repayment history". One-hot
columns are summed back into the column they came from, using the fitted
transformer's own structure rather than by parsing feature names — ``PAY_0_-1``
and ``PAY_AMT1`` are not distinguishable by prefix.

**Report the reasons that drove the decision.** For a decline, that means the
features that pushed risk *up*. Listing what helped an applicant who was refused
is not an adverse-action reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 4

# Plain-language descriptions. A reason code an applicant cannot act on is not
# much better than no reason at all.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "LIMIT_BAL": "credit limit",
    "PAY_0": "repayment status in the most recent month",
    "PAY_2": "repayment status two months ago",
    "PAY_3": "repayment status three months ago",
    "PAY_4": "repayment status four months ago",
    "PAY_5": "repayment status five months ago",
    "PAY_6": "repayment status six months ago",
    "EDUCATION": "education category",
    "BILL_AMT1": "most recent statement balance",
    "BILL_AMT2": "statement balance two months ago",
    "BILL_AMT3": "statement balance three months ago",
    "BILL_AMT4": "statement balance four months ago",
    "BILL_AMT5": "statement balance five months ago",
    "BILL_AMT6": "statement balance six months ago",
    "PAY_AMT1": "amount paid in the most recent month",
    "PAY_AMT2": "amount paid two months ago",
    "PAY_AMT3": "amount paid three months ago",
    "PAY_AMT4": "amount paid four months ago",
    "PAY_AMT5": "amount paid five months ago",
    "PAY_AMT6": "amount paid six months ago",
}


@dataclass
class Reason:
    feature: str
    description: str
    value: float
    contribution: float  # signed: positive raises predicted default risk

    @property
    def direction(self) -> str:
        return "increased_risk" if self.contribution > 0 else "decreased_risk"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "description": self.description,
            "value": self.value,
            "contribution": round(float(self.contribution), 5),
            "direction": self.direction,
        }


def _source_column_map(preprocessor: ColumnTransformer) -> list[str]:
    """Map each encoded output column back to the source column it came from.

    Built from the fitted transformer's structure rather than by parsing names:
    the one-hot encoder emits ``<column>_<value>``, and values containing
    underscores or negative numbers make prefix-matching ambiguous.
    """
    mapping: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        categories = getattr(transformer, "categories_", None)
        if categories is None:
            mapping.extend(columns)  # one output per input, order preserved
            continue
        # OneHotEncoder(drop="first") emits len(categories) - 1 columns each.
        drop_idx = getattr(transformer, "drop_idx_", None)
        for index, column in enumerate(columns):
            emitted = len(categories[index]) - (0 if drop_idx is None else 1)
            mapping.extend([column] * emitted)
    return mapping


class Explainer:
    """Wraps a fitted pipeline and produces per-row reasons."""

    def __init__(self, pipeline: Pipeline) -> None:
        import shap

        self.preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["estimator"]
        self.column_map = _source_column_map(self.preprocessor)
        # TreeExplainer is exact for gradient-boosted trees and fast enough to run
        # inline; the sampling-based explainers are not.
        self.explainer = shap.TreeExplainer(estimator)
        logger.info("Explainer ready over %d encoded columns", len(self.column_map))

    def _aggregate(self, shap_row: np.ndarray) -> dict[str, float]:
        """Sum encoded contributions back into their source columns."""
        totals: dict[str, float] = {}
        for value, column in zip(shap_row, self.column_map, strict=True):
            totals[column] = totals.get(column, 0.0) + float(value)
        return totals

    def explain(
        self,
        frame: pd.DataFrame,
        declined: list[bool],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[list[Reason]]:
        """Return the principal reasons for each row's decision.

        For a declined application these are the factors that raised risk, which
        is what an adverse-action notice has to state. For an approved one the
        strongest factors either way are returned, since there is no notice to
        issue and the useful answer is simply what drove the score.
        """
        transformed = self.preprocessor.transform(frame)
        values = self.explainer.shap_values(transformed)
        if values.ndim == 3:  # some versions return one array per class
            values = values[:, :, 1]

        reasons: list[list[Reason]] = []
        for row_index in range(len(frame)):
            totals = self._aggregate(values[row_index])
            ordered = sorted(totals.items(), key=lambda kv: -abs(kv[1]))
            if declined[row_index]:
                # Only risk-increasing factors are adverse-action reasons.
                ordered = [(c, v) for c, v in totals.items() if v > 0]
                ordered.sort(key=lambda kv: -kv[1])

            reasons.append(
                [
                    Reason(
                        feature=column,
                        description=FEATURE_DESCRIPTIONS.get(column, column),
                        value=float(frame.iloc[row_index][column]),
                        contribution=contribution,
                    )
                    for column, contribution in ordered[:top_k]
                ]
            )
        return reasons
