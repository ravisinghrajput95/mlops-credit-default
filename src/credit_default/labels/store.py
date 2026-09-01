"""Where late-arriving outcomes land.

Mirrors ``api.sinks``: an interface with a Parquet implementation for the
hermetic local demo and a Postgres implementation for the running compose stack,
so the join and reporting code above it never knows which is in use.

Every outcome carries **two** timestamps, and keeping them apart is the entire
point of this module:

``outcome_at``
    Event time. When the performance window closed and the outcome became
    *defined*. This is a property of the loan, not of our systems.
``observed_at``
    Ingestion time. When the outcome actually reached this store. This is a
    property of our systems, not of the loan.

A single ``labelled_at`` column would collapse the two and silently destroy the
ability to ask "what did we know on the 3rd of March?" -- which is the only
question a backtest is allowed to ask. Answering it with today's knowledge is how
an offline evaluation ends up better than anything production can reproduce.

Reads are therefore filtered on ``observed_at``, never on ``outcome_at``: the
store's job is to reproduce what was *known* at a point in time. Whether a
prediction is old enough to be scored at all is a separate question, handled by
``labels.backfill``.
"""

from __future__ import annotations

import datetime as dt
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from credit_default.config import Settings

logger = logging.getLogger(__name__)

OUTCOME_COLUMNS = ["application_id", "outcome_at", "observed_at", "label"]


@dataclass(frozen=True)
class Outcome:
    """One matured outcome, as reported by the servicing system."""

    application_id: str
    outcome_at: dt.datetime
    observed_at: dt.datetime
    label: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "application_id": self.application_id,
            "outcome_at": self.outcome_at,
            "observed_at": self.observed_at,
            "label": int(self.label),
        }


def _empty() -> pd.DataFrame:
    """An empty frame with the right dtypes, so downstream joins still type-check."""
    return pd.DataFrame(
        {
            "application_id": pd.Series(dtype="object"),
            "outcome_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "observed_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "label": pd.Series(dtype="int64"),
        }
    )


class OutcomeStore(ABC):
    """Durable record of outcomes, readable as of a point in time."""

    @abstractmethod
    def record(self, outcomes: list[Outcome]) -> int:
        """Append outcomes, ignoring any whose application is already recorded.

        Idempotent on purpose. A label feed that is replayed after a failed run
        must not double-count, and an outcome is not allowed to change once
        reported -- a restated label is a new decision, not an edit.
        """

    @abstractmethod
    def read(self, as_of: dt.datetime | None = None) -> pd.DataFrame:
        """Outcomes known by ``as_of`` (``observed_at <= as_of``); all of them if None."""


class NullOutcomeStore(OutcomeStore):
    """Discards outcomes. Used when the label pipeline is switched off."""

    def record(self, outcomes: list[Outcome]) -> int:
        return 0

    def read(self, as_of: dt.datetime | None = None) -> pd.DataFrame:
        return _empty()


class ParquetOutcomeStore(OutcomeStore):
    """Single-file store for the offline demo and the test suite."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _load(self) -> pd.DataFrame:
        if not self.path.exists():
            return _empty()
        frame = pd.read_parquet(self.path)
        for column in ("outcome_at", "observed_at"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        return frame

    def record(self, outcomes: list[Outcome]) -> int:
        if not outcomes:
            return 0
        existing = self._load()
        known = set(existing["application_id"])
        fresh = [o.to_dict() for o in outcomes if o.application_id not in known]
        if not fresh:
            return 0

        combined = pd.concat([existing, pd.DataFrame(fresh)], ignore_index=True)
        for column in ("outcome_at", "observed_at"):
            combined[column] = pd.to_datetime(combined[column], utc=True)
        combined["label"] = combined["label"].astype("int64")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        combined[OUTCOME_COLUMNS].to_parquet(self.path, index=False)
        return len(fresh)

    def read(self, as_of: dt.datetime | None = None) -> pd.DataFrame:
        frame = self._load()
        if as_of is not None:
            frame = frame[frame["observed_at"] <= pd.Timestamp(as_of)]
        return frame.reset_index(drop=True)


class PostgresOutcomeStore(OutcomeStore):
    """Live path, alongside the prediction sink in the compose stack."""

    DDL = """
    CREATE TABLE IF NOT EXISTS outcomes (
        application_id TEXT PRIMARY KEY,
        outcome_at     TIMESTAMPTZ NOT NULL,
        observed_at    TIMESTAMPTZ NOT NULL,
        label          SMALLINT NOT NULL
    );
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        with self._connect() as conn:
            conn.execute(self.DDL)

    def _connect(self) -> Any:
        import psycopg

        return psycopg.connect(self._dsn, autocommit=True)

    def record(self, outcomes: list[Outcome]) -> int:
        if not outcomes:
            return 0
        # ON CONFLICT DO NOTHING is the idempotency guarantee: replaying a feed
        # is a no-op rather than a restatement.
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO outcomes (application_id, outcome_at, observed_at, label)"
                " VALUES (%s, %s, %s, %s) ON CONFLICT (application_id) DO NOTHING",
                [(o.application_id, o.outcome_at, o.observed_at, int(o.label)) for o in outcomes],
            )
            return int(cur.rowcount)

    def read(self, as_of: dt.datetime | None = None) -> pd.DataFrame:
        # Columns are a module constant, not caller input.
        query = f"SELECT {', '.join(OUTCOME_COLUMNS)} FROM outcomes"
        params: tuple[Any, ...] = ()
        if as_of is not None:
            query += " WHERE observed_at <= %s"
            params = (as_of,)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        if not rows:
            return _empty()

        frame = pd.DataFrame(rows, columns=OUTCOME_COLUMNS)
        for column in ("outcome_at", "observed_at"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        frame["label"] = frame["label"].astype("int64")
        return frame


def build_outcome_store(settings: Settings) -> OutcomeStore:
    """Construct the configured store, degrading to a null store if it cannot start.

    Same posture as the prediction sink: recording outcomes is important, but it
    is a reporting path, and it must not be able to fail a training run.
    """
    try:
        if settings.label_store == "parquet":
            return ParquetOutcomeStore(settings.outcomes_parquet)
        if settings.label_store == "postgres":
            return PostgresOutcomeStore(settings.postgres_dsn)
    except Exception:
        logger.exception(
            "Could not initialise the '%s' outcome store; falling back to NullOutcomeStore.",
            settings.label_store,
        )
        return NullOutcomeStore()
    return NullOutcomeStore()
