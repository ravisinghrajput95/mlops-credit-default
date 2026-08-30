"""Promote the challenger to champion.

Promotion is a separate, explicit command rather than something ``train.py`` does
on its own. Automatically promoting whatever scored best would mean an unattended
retrain could ship a model to production with no human in the loop -- fine for a
demo, wrong as a default in a credit-risk system where the cost of a bad model is
borne by customers. The quality gate must pass before promotion is allowed.
"""

from __future__ import annotations

import argparse
import logging
import sys

import mlflow
from mlflow.tracking import MlflowClient

from credit_default.config import get_settings
from credit_default.evaluate import evaluate

logger = logging.getLogger(__name__)


def promote(force: bool = False) -> int:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    result = evaluate(settings)
    if not result.passed and not force:
        for reason in result.reasons:
            logger.error("Refusing to promote: %s", reason)
        return 1

    if not result.passed:
        logger.warning("Quality gate failed but --force was given; promoting anyway.")

    challenger = client.get_model_version_by_alias(
        settings.registered_model_name, settings.challenger_alias
    )
    client.set_registered_model_alias(
        settings.registered_model_name, settings.champion_alias, challenger.version
    )
    logger.info(
        "Promoted %s v%s to @%s (pr_auc=%.4f)",
        settings.registered_model_name,
        challenger.version,
        settings.champion_alias,
        result.challenger.get("pr_auc", float("nan")),
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote the challenger to champion.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Promote even if the quality gate fails (requires a deliberate override).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")
    sys.exit(promote(force=args.force))


if __name__ == "__main__":
    main()
