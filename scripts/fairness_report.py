#!/usr/bin/env python
"""Measure what excluding protected attributes actually costs.

Trains the same model twice -- once with SEX, MARRIAGE and AGE available, once
without -- and compares both predictive performance and group fairness.

The point is to replace an assertion ("we removed the protected attributes, so
the model is fair") with a measurement. Two results are worth watching for:

* The accuracy cost may be small. If so, the tradeoff is easy, and saying so with
  a number is far stronger than claiming it.
* The fairness gaps may barely move. That is the expected outcome, not a bug:
  other features correlate with the protected ones, so the model can still
  reconstruct them. This is why "fairness through unawareness" is insufficient,
  and why the attributes are retained for auditing rather than deleted.
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from credit_default.config import (
    AUDIT_ATTRIBUTES,
    FEATURES,
    PROTECTED_ATTRIBUTES,
    TARGET,
    get_settings,
    model_features,
)
from credit_default.fairness import audit
from credit_default.features.pipeline import build_pipeline

logger = logging.getLogger(__name__)


def _train_variant(train: pd.DataFrame, test: pd.DataFrame, use_protected: bool, seed: int):
    numeric, categorical = model_features(use_protected)
    pipeline = build_pipeline(
        XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=-1,
        ),
        numeric,
        categorical,
    )
    pipeline.fit(train[FEATURES], train[TARGET])
    probabilities = pipeline.predict_proba(test[FEATURES])[:, 1]

    return {
        "n_features": len(numeric) + len(categorical),
        "pr_auc": float(average_precision_score(test[TARGET], probabilities)),
        "roc_auc": float(roc_auc_score(test[TARGET], probabilities)),
        "audits": audit(test, test[TARGET], probabilities, AUDIT_ATTRIBUTES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    settings.ensure_dirs()

    train = pd.read_parquet(settings.train_parquet)
    test = pd.read_parquet(settings.test_parquet)

    logger.info("Training WITH protected attributes (%s)", ", ".join(PROTECTED_ATTRIBUTES))
    with_protected = _train_variant(train, test, True, settings.random_seed)

    logger.info("Training WITHOUT them")
    without = _train_variant(train, test, False, settings.random_seed)

    print("\n" + "=" * 78)
    print("PREDICTIVE PERFORMANCE")
    print("=" * 78)
    print(f"{'variant':<34}{'features':>10}{'PR-AUC':>12}{'ROC-AUC':>12}")
    print(
        f"{'with protected attributes':<34}{with_protected['n_features']:>10}"
        f"{with_protected['pr_auc']:>12.4f}{with_protected['roc_auc']:>12.4f}"
    )
    print(
        f"{'without (shipped default)':<34}{without['n_features']:>10}"
        f"{without['pr_auc']:>12.4f}{without['roc_auc']:>12.4f}"
    )
    print(
        f"{'cost of exclusion':<34}{'':>10}"
        f"{without['pr_auc'] - with_protected['pr_auc']:>+12.4f}"
        f"{without['roc_auc'] - with_protected['roc_auc']:>+12.4f}"
    )

    print("\n" + "=" * 78)
    print("FAIRNESS GAPS  (lower is more equal; 'base' = real difference in the data)")
    print("=" * 78)
    print(f"{'attribute':<14}{'selection gap':>28}{'equal-opportunity gap':>28}")
    print(f"{'':14}{'with':>13}{'without':>15}{'with':>13}{'without':>15}")
    for name in AUDIT_ATTRIBUTES:
        a, b = with_protected["audits"].get(name), without["audits"].get(name)
        if not a or not b:
            continue
        print(
            f"{name:<14}{a.selection_rate_gap:>13.4f}{b.selection_rate_gap:>15.4f}"
            f"{a.equal_opportunity_gap:>13.4f}{b.equal_opportunity_gap:>15.4f}"
        )
        print(f"{'  (base rate gap in the data: ' + format(b.base_rate_gap, '.4f') + ')':<70}")

    payload = {
        "with_protected": {k: v for k, v in with_protected.items() if k != "audits"}
        | {"audits": {n: a.to_dict() for n, a in with_protected["audits"].items()}},
        "without_protected": {k: v for k, v in without.items() if k != "audits"}
        | {"audits": {n: a.to_dict() for n, a in without["audits"].items()}},
        "excluded": PROTECTED_ATTRIBUTES,
    }
    destination = args.output or (settings.reports_dir / "fairness_comparison.json")
    with open(destination, "w") as handle:  # noqa: PTH123
        json.dump(payload, handle, indent=2)
    print(f"\nWritten to {destination}")


if __name__ == "__main__":
    main()
