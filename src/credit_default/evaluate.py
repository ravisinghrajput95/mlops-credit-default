"""Model quality gate.

Compares the current ``challenger`` against two bars and exits non-zero if either
is missed, so a bad model fails the build instead of reaching production:

1. An absolute floor (``min_pr_auc``) -- catches a catastrophically broken run.
2. A relative bar against the incumbent ``champion`` -- catches a silent
   regression, which is the failure mode an absolute floor alone will miss.

Exit code 1 means "do not promote".
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException

from credit_default.config import FEATURES, TARGET, Settings, get_settings
from credit_default.train import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    passed: bool
    reasons: list[str]
    challenger: dict[str, float]
    champion: dict[str, float] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "reasons": self.reasons,
            "challenger": self.challenger,
            "champion": self.champion,
        }


def _score_alias(alias: str, test: pd.DataFrame, settings: Settings) -> dict[str, float] | None:
    """Score the model behind an alias, or return None if the alias is unset."""
    uri = f"models:/{settings.registered_model_name}@{alias}"
    try:
        model = mlflow.sklearn.load_model(uri)
    except MlflowException:
        logger.info("No model registered under @%s", alias)
        return None
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    return compute_metrics(test[TARGET], probabilities)


def evaluate(settings: Settings | None = None) -> GateResult:
    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    test = pd.read_parquet(settings.test_parquet)

    challenger = _score_alias(settings.challenger_alias, test, settings)
    if challenger is None:
        return GateResult(False, ["No challenger model is registered."], {}, None)

    champion = _score_alias(settings.champion_alias, test, settings)

    reasons: list[str] = []
    if challenger["pr_auc"] < settings.min_pr_auc:
        reasons.append(
            f"PR-AUC {challenger['pr_auc']:.4f} is below the absolute floor "
            f"{settings.min_pr_auc:.4f}."
        )

    if champion is not None:
        delta = challenger["pr_auc"] - champion["pr_auc"]
        if delta < -settings.max_pr_auc_regression:
            reasons.append(
                f"PR-AUC regressed by {abs(delta):.4f} against the champion "
                f"({champion['pr_auc']:.4f} -> {challenger['pr_auc']:.4f}), which exceeds "
                f"the tolerance of {settings.max_pr_auc_regression:.4f}."
            )

    return GateResult(not reasons, reasons, challenger, champion)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the model quality gate.")
    parser.add_argument("--output", default=None, help="Write the gate result as JSON.")
    args = parser.parse_args()

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")
    settings = get_settings()
    result = evaluate(settings)

    payload = json.dumps(result.to_dict(), indent=2)
    output = args.output or (settings.reports_dir / "gate.json")
    settings.ensure_dirs()
    with open(output, "w") as handle:  # noqa: PTH123
        handle.write(payload)

    if result.passed:
        logger.info("Quality gate PASSED (pr_auc=%.4f)", result.challenger["pr_auc"])
        return

    for reason in result.reasons:
        logger.error("Quality gate FAILED: %s", reason)
    sys.exit(1)


if __name__ == "__main__":
    main()
