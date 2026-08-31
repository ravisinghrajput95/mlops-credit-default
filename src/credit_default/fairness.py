"""Fairness auditing across protected groups.

Two ideas drive this module.

**Excluding an attribute is not the same as removing its influence.** Dropping
SEX from the feature set does not make a model blind to sex, because other
features correlate with it. "Fairness through unawareness" is a known failure
mode, and the only way to know whether it worked here is to measure. That is why
the protected attributes stay in the dataset and in the API contract even when
the estimator never sees them -- you cannot audit a group you did not record.

**There is no single fairness number.** Two standard and mutually incompatible
definitions are reported side by side:

* *Demographic parity* -- do groups get approved at the same rate? Ignores that
  groups may genuinely differ in risk.
* *Equal opportunity* -- among customers who actually defaulted, are they caught
  at the same rate? Conditions on the outcome, so it tolerates real base-rate
  differences but is blind to disparate impact.

They cannot both be satisfied unless base rates are equal. Choosing between them
is a policy decision for legal and compliance, not something a library settles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Age is continuous, so it is banded before group metrics are computed.
AGE_BANDS = [(18, 30), (30, 45), (45, 60), (60, 120)]


@dataclass
class GroupMetrics:
    """Outcomes for one demographic group."""

    group: str
    n: int
    selection_rate: float  # share predicted to default
    true_positive_rate: float  # recall among actual defaulters
    false_positive_rate: float
    base_rate: float  # actual default rate in this group

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "n": self.n,
            "selection_rate": round(self.selection_rate, 4),
            "true_positive_rate": round(self.true_positive_rate, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "base_rate": round(self.base_rate, 4),
        }


@dataclass
class AttributeAudit:
    """Disparities across the groups of a single attribute."""

    attribute: str
    groups: list[GroupMetrics] = field(default_factory=list)

    @property
    def selection_rate_gap(self) -> float:
        """Demographic parity: max minus min approval rate across groups."""
        rates = [g.selection_rate for g in self.groups if g.n > 0]
        return max(rates) - min(rates) if len(rates) > 1 else 0.0

    @property
    def equal_opportunity_gap(self) -> float:
        """Largest difference in recall among customers who actually defaulted."""
        rates = [g.true_positive_rate for g in self.groups if g.n > 0]
        return max(rates) - min(rates) if len(rates) > 1 else 0.0

    @property
    def base_rate_gap(self) -> float:
        """How much the groups genuinely differ, before the model is involved.

        Reported because a selection-rate gap smaller than the base-rate gap means
        the model is compressing a real difference, not inventing one.
        """
        rates = [g.base_rate for g in self.groups if g.n > 0]
        return max(rates) - min(rates) if len(rates) > 1 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "selection_rate_gap": round(self.selection_rate_gap, 4),
            "equal_opportunity_gap": round(self.equal_opportunity_gap, 4),
            "base_rate_gap": round(self.base_rate_gap, 4),
            "groups": [g.to_dict() for g in self.groups],
        }


def band_age(ages: pd.Series) -> pd.Series:
    """Bucket continuous age so group metrics are meaningful."""
    labels = [f"{lo}-{hi}" for lo, hi in AGE_BANDS]
    edges = [AGE_BANDS[0][0]] + [hi for _, hi in AGE_BANDS]
    return pd.cut(ages, bins=edges, labels=labels, right=False, include_lowest=True)


def _rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    positives = y_true == 1
    negatives = ~positives
    selection = float(y_pred.mean()) if len(y_pred) else 0.0
    tpr = float(y_pred[positives].mean()) if positives.any() else 0.0
    fpr = float(y_pred[negatives].mean()) if negatives.any() else 0.0
    base = float(positives.mean()) if len(y_true) else 0.0
    return selection, tpr, fpr, base


def audit_attribute(
    frame: pd.DataFrame,
    attribute: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> AttributeAudit:
    values = band_age(frame[attribute]) if attribute == "AGE" else frame[attribute]
    audit = AttributeAudit(attribute=attribute)

    for value in sorted(values.dropna().unique(), key=str):
        mask = (values == value).to_numpy()
        if mask.sum() == 0:
            continue
        selection, tpr, fpr, base = _rates(y_true.to_numpy()[mask], y_pred[mask])
        audit.groups.append(
            GroupMetrics(
                group=str(value),
                n=int(mask.sum()),
                selection_rate=selection,
                true_positive_rate=tpr,
                false_positive_rate=fpr,
                base_rate=base,
            )
        )
    return audit


def audit(
    frame: pd.DataFrame,
    y_true: pd.Series,
    probabilities: np.ndarray,
    attributes: list[str],
    threshold: float = 0.5,
) -> dict[str, AttributeAudit]:
    """Audit predictions across every named attribute."""
    y_pred = (probabilities >= threshold).astype(int)
    return {a: audit_attribute(frame, a, y_true, y_pred) for a in attributes if a in frame}


def flatten_for_mlflow(audits: dict[str, AttributeAudit]) -> dict[str, float]:
    """Reduce an audit to scalar metrics so runs can be compared over time."""
    metrics: dict[str, float] = {}
    for name, a in audits.items():
        metrics[f"fairness_{name}_selection_gap"] = round(a.selection_rate_gap, 4)
        metrics[f"fairness_{name}_equal_opportunity_gap"] = round(a.equal_opportunity_gap, 4)
    return metrics
