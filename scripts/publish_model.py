#!/usr/bin/env python
"""Copy the current champion model to GCS for Cloud Run to load.

The cloud deployment runs no MLflow server -- Cloud SQL has no always-free tier,
so hosting one would be the single line item that costs money. Instead the
champion artifact is published to object storage and Cloud Run loads it from
there at startup.

This is a deliberate, manual step for the same reason promotion is: it is the
moment a model actually starts serving real traffic.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import mlflow
import mlflow.models
import mlflow.sklearn

from credit_default.config import get_settings

logger = logging.getLogger(__name__)


def publish(destination: str, alias: str | None = None) -> str:
    settings = get_settings()
    alias = alias or settings.champion_alias
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    uri = f"models:/{settings.registered_model_name}@{alias}"
    logger.info("Loading %s", uri)
    model = mlflow.sklearn.load_model(uri)

    # Re-saving a model drops everything not passed explicitly, so the metadata is
    # carried across by hand. The decision threshold lives here, and losing it
    # silently reverts serving to the 0.5 default -- which is exactly what
    # happened the first time this script ran against Cloud Run.
    metadata = mlflow.models.Model.load(uri).metadata or {}
    logger.info("Carrying metadata across: %s", metadata or "(none)")

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "model"
        mlflow.sklearn.save_model(
            model,
            str(staged),
            skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
            metadata=metadata,
        )

        if destination.startswith("gs://"):
            import gcsfs

            fs = gcsfs.GCSFileSystem()
            target = destination.rstrip("/")
            if fs.exists(target):
                logger.info("Replacing the existing artifact at %s", target)
                fs.rm(target, recursive=True)
            fs.put(str(staged), target, recursive=True)
        else:
            target_path = Path(destination)
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(staged, target_path)
            target = str(target_path)

    logger.info("Published %s to %s", alias, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the champion model to GCS.")
    parser.add_argument(
        "destination",
        nargs="?",
        default=None,
        help="gs://bucket/models/champion (defaults to GCS_MODEL_URI).",
    )
    parser.add_argument("--alias", default=None, help="Registry alias to publish.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    destination = args.destination or get_settings().gcs_model_uri
    if not destination:
        raise SystemExit("Pass a destination or set GCS_MODEL_URI.")
    publish(destination, args.alias)


if __name__ == "__main__":
    main()
