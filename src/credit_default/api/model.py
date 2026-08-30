"""Model loading for the serving layer.

Two sources are supported. ``registry`` pulls the model behind the ``champion``
alias from MLflow, which is what runs in the compose stack. ``local`` loads a
directory saved by ``train.py``, which lets the API and its tests start with no
MLflow server reachable -- necessary in CI and for the Cloud Run image, which
carries its model rather than depending on a tracking server at boot.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from credit_default.config import FEATURES, Settings
from credit_default.data.schema import clean

logger = logging.getLogger(__name__)


class ModelHandle:
    """A loaded model plus the metadata the API reports about it."""

    def __init__(self, model: Any, version: str | None, source: str) -> None:
        self.model = model
        self.version = version
        self.source = source

    def predict(self, frame: pd.DataFrame, threshold: float) -> tuple[list[float], list[int]]:
        """Score a frame, applying the same cleaning used during training."""
        prepared = clean(frame)[FEATURES]
        probabilities = self.model.predict_proba(prepared)[:, 1]
        labels = (probabilities >= threshold).astype(int)
        return [float(p) for p in probabilities], [int(v) for v in labels]


def load_model(settings: Settings) -> ModelHandle:
    import mlflow.sklearn

    if settings.model_source == "registry":
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        uri = f"models:/{settings.registered_model_name}@{settings.champion_alias}"
        model = mlflow.sklearn.load_model(uri)

        version: str | None = None
        try:
            from mlflow.tracking import MlflowClient

            version = str(
                MlflowClient()
                .get_model_version_by_alias(
                    settings.registered_model_name, settings.champion_alias
                )
                .version
            )
        except Exception:  # version is informational only
            logger.warning("Loaded the champion model but could not resolve its version.")

        logger.info("Loaded %s (version %s)", uri, version)
        return ModelHandle(model, version, "registry")

    path = settings.local_model_path
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Run 'make train' first, or set MODEL_SOURCE=registry."
        )
    model = mlflow.sklearn.load_model(str(path))
    logger.info("Loaded local model from %s", path)
    return ModelHandle(model, "local", "local")
