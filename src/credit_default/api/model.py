"""Model loading for the serving layer.

Three sources are supported:

``registry``
    Pulls the model behind the ``champion`` alias from MLflow. This is what the
    local compose stack uses.
``local``
    Loads a directory saved by ``train.py``, so the API and its tests start with
    no MLflow server reachable. Used in CI and for offline demos.
``gcs``
    Loads the artifact from object storage. This is what Cloud Run uses: the
    cloud deployment runs no MLflow server, because Cloud SQL has no always-free
    tier and the tracking server would need one.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from credit_default.config import FEATURES, Settings
from credit_default.data.schema import clean

logger = logging.getLogger(__name__)


def _threshold_from(path_or_uri: str, settings: Settings) -> float:
    """Read the tuned cutoff out of the saved MLflow model's metadata."""
    import mlflow.models

    try:
        info = mlflow.models.Model.load(path_or_uri)
        value = (info.metadata or {}).get("decision_threshold")
        if value is not None:
            return float(value)
    except Exception:
        logger.warning("Could not read decision_threshold from %s", path_or_uri)
    logger.info("Falling back to the default threshold %.2f", settings.default_decision_threshold)
    return settings.default_decision_threshold


class ModelHandle:
    """A loaded model plus the metadata the API reports about it."""

    def __init__(
        self,
        model: Any,
        version: str | None,
        source: str,
        threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.version = version
        self.source = source
        # The cutoff this model was tuned for, read from its own metadata. A
        # constant in the API would silently drift away from the model it serves.
        self.threshold = threshold
        self._explainer: Any | None = None

    @property
    def explainer(self) -> Any:
        """Built on first use.

        Constructing a TreeExplainer costs a moment, and an API that never
        receives explain=true should not pay it at startup -- particularly on
        Cloud Run, where startup time is cold-start latency for a real caller.
        """
        if self._explainer is None:
            from credit_default.explain import Explainer

            self._explainer = Explainer(self.model)
        return self._explainer

    def explain(self, frame: pd.DataFrame, declined: list[bool]) -> Any:
        """Principal reasons per row, in the same order as the input."""
        return self.explainer.explain(clean(frame)[FEATURES], declined)

    def predict(
        self, frame: pd.DataFrame, threshold: float | None = None
    ) -> tuple[list[float], list[int]]:
        """Score a frame, applying the same cleaning used during training."""
        cutoff = self.threshold if threshold is None else threshold
        prepared = clean(frame)[FEATURES]
        probabilities = self.model.predict_proba(prepared)[:, 1]
        labels = (probabilities >= cutoff).astype(int)
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
                .get_model_version_by_alias(settings.registered_model_name, settings.champion_alias)
                .version
            )
        except Exception:  # version is informational only
            logger.warning("Loaded the champion model but could not resolve its version.")

        logger.info("Loaded %s (version %s)", uri, version)
        return ModelHandle(model, version, "registry", _threshold_from(uri, settings))

    if settings.model_source == "gcs":
        if not settings.gcs_model_uri:
            raise ValueError("GCS_MODEL_URI must be set when MODEL_SOURCE=gcs")
        model = mlflow.sklearn.load_model(settings.gcs_model_uri)
        logger.info("Loaded model from %s", settings.gcs_model_uri)
        return ModelHandle(
            model,
            settings.gcs_model_uri.rstrip("/").split("/")[-1],
            "gcs",
            _threshold_from(settings.gcs_model_uri, settings),
        )

    path = settings.local_model_path
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Run 'make train' first, or set MODEL_SOURCE=registry."
        )
    model = mlflow.sklearn.load_model(str(path))
    logger.info("Loaded local model from %s", path)
    return ModelHandle(model, "local", "local", _threshold_from(str(path), settings))
