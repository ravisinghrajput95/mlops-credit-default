"""Where served predictions get written.

The drift monitor needs to see what the model actually scored in production, so
every request is persisted. The destination differs by environment -- Postgres in
the local compose stack, append-only Parquet on GCS in the cloud (which keeps the
deployment inside GCP's always-free tier, as Cloud SQL has none) -- but the API
code path is identical either way. Selecting a sink is a config change.

Writes are best-effort: a monitoring sink that is down must never take the
prediction endpoint down with it.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any

from credit_default.config import Settings

logger = logging.getLogger(__name__)


class PredictionSink(ABC):
    """Destination for served-prediction records."""

    @abstractmethod
    def write(self, records: list[dict[str, Any]]) -> None: ...

    def close(self) -> None:  # pragma: no cover - most sinks need no teardown
        return None


class NullSink(PredictionSink):
    """Discards records. Used in tests and when monitoring is switched off."""

    def write(self, records: list[dict[str, Any]]) -> None:
        return None


class PostgresSink(PredictionSink):
    """Local compose stack. Creates its table on first use."""

    DDL = """
    CREATE TABLE IF NOT EXISTS predictions (
        id            UUID PRIMARY KEY,
        predicted_at  TIMESTAMPTZ NOT NULL,
        model_version TEXT,
        probability   DOUBLE PRECISION NOT NULL,
        prediction    SMALLINT NOT NULL,
        features      JSONB NOT NULL
    );
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        with self._connect() as conn:
            conn.execute(self.DDL)

    def _connect(self) -> Any:
        import psycopg

        return psycopg.connect(self._dsn, autocommit=True)

    def write(self, records: list[dict[str, Any]]) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO predictions"
                " (id, predicted_at, model_version, probability, prediction, features)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (
                        r["id"],
                        r["predicted_at"],
                        r["model_version"],
                        r["probability"],
                        r["prediction"],
                        json.dumps(r["features"]),
                    )
                    for r in records
                ],
            )


class GCSParquetSink(PredictionSink):
    """Cloud deployment. One Parquet object per batch, partitioned by date.

    Object storage rather than a database because Cloud SQL has no always-free
    tier; the drift job reads the whole prefix with a glob, so many small objects
    are fine at demo volume.
    """

    def __init__(self, prefix: str) -> None:
        if not prefix:
            raise ValueError("gcs_prediction_prefix must be set when prediction_sink='gcs'")
        self.prefix = prefix.rstrip("/")

    def write(self, records: list[dict[str, Any]]) -> None:
        import pandas as pd

        frame = pd.json_normalize(records)
        stamp = dt.datetime.now(dt.UTC)
        path = f"{self.prefix}/date={stamp:%Y-%m-%d}/{stamp:%H%M%S}-{uuid.uuid4().hex[:8]}.parquet"
        frame.to_parquet(path, index=False)


def build_sink(settings: Settings) -> PredictionSink:
    """Construct the configured sink, degrading to NullSink if it cannot start.

    Monitoring is important but it is not worth failing startup over: an API that
    serves predictions without logging them beats an API that will not boot.
    """
    try:
        if settings.prediction_sink == "postgres":
            return PostgresSink(settings.postgres_dsn)
        if settings.prediction_sink == "gcs":
            return GCSParquetSink(settings.gcs_prediction_prefix)
    except Exception:
        logger.exception(
            "Could not initialise the '%s' prediction sink; falling back to NullSink. "
            "Predictions will be served but not recorded.",
            settings.prediction_sink,
        )
        return NullSink()
    return NullSink()
