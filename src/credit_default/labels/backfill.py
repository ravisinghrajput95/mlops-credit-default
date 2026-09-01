"""Joining late-arriving outcomes back to the predictions that earned them.

This is the module where point-in-time correctness lives. An evaluation dated
``as_of`` is only allowed to use two kinds of fact:

* predictions whose performance window had already closed by then, so the
  outcome was *defined*; and
* outcomes that had already been reported by then, so the label was *known*.

Violating either produces a backtest that no production system can reproduce.
The second is the subtle one and the more damaging: the label existed in the
world, so a careless join finds it, but nobody had it yet on the date being
simulated. Every offline number computed that way is optimistic, and the gap only
shows up once the model is live and the number refuses to reappear.

``join_labels_ignoring_time`` is the wrong implementation, kept deliberately. It
is what a first pass at this join looks like, and shipping it next to the correct
one means the size of the error can be *measured* rather than described -- see
``scripts/label_report.py`` and ``tests/test_labels.py``.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_PREDICTION_COLUMNS = ["application_id", "predicted_at", "probability", "prediction"]


@dataclass
class LabelJoin:
    """Predictions that can legitimately be scored as of a date, and what is missing.

    The counts matter as much as the frame. A performance number computed over
    40% coverage is not a performance number, and the only way a caller can know
    that is if the join reports what it excluded and why.
    """

    as_of: dt.datetime
    frame: pd.DataFrame
    total: int
    matured: int
    observed: int
    pending: int
    censored: int

    @property
    def coverage(self) -> float:
        """Share of matured predictions whose label has actually been reported."""
        return float(self.observed / self.matured) if self.matured else 0.0

    @property
    def maturity(self) -> float:
        """Share of all predictions whose performance window has closed."""
        return float(self.matured / self.total) if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "total_predictions": self.total,
            "matured": self.matured,
            "observed": self.observed,
            "pending": self.pending,
            "censored": self.censored,
            "coverage": round(self.coverage, 4),
            "maturity": round(self.maturity, 4),
        }


def _prepare(predictions: pd.DataFrame, window_days: int) -> pd.DataFrame:
    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in predictions.columns]
    if missing:
        raise ValueError(
            f"Served predictions are missing {missing}. A prediction with no stable "
            "application_id can never be joined to its outcome."
        )
    frame = predictions.copy()
    frame["predicted_at"] = pd.to_datetime(frame["predicted_at"], utc=True)
    frame["matures_at"] = frame["predicted_at"] + pd.Timedelta(float(window_days), unit="D")
    return frame


def join_labels(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    as_of: dt.datetime,
    window_days: int,
) -> LabelJoin:
    """Point-in-time-correct join of outcomes onto predictions.

    ``outcomes`` should already be filtered to what the store had observed by
    ``as_of`` (``OutcomeStore.read(as_of=...)`` does this). The observation filter
    is applied again here anyway, because a join that silently trusts its input to
    have been filtered correctly is a leak waiting for its first careless caller.
    """
    stamp = pd.Timestamp(as_of)
    frame = _prepare(predictions, window_days)

    # Gate 1: the outcome must be defined by now.
    matured = frame[frame["matures_at"] <= stamp]

    # Gate 2: the label must have been reported by now.
    usable = outcomes.copy()
    if not usable.empty:
        usable["observed_at"] = pd.to_datetime(usable["observed_at"], utc=True)
        usable = usable[usable["observed_at"] <= stamp]

    joined = matured.merge(
        usable[["application_id", "outcome_at", "observed_at", "label"]]
        if not usable.empty
        else usable,
        on="application_id",
        how="inner",
    )

    unlabelled = matured[~matured["application_id"].isin(joined["application_id"])]
    # A declined applicant never opens an account, so their outcome is not late --
    # it does not exist. Approved-but-missing is simply still in the post. The
    # split is approximate early on, because a declined applicant drawn into the
    # random holdout is counted as censored until their file lands.
    censored = int((unlabelled["prediction"] == 1).sum())
    pending = int(len(unlabelled) - censored)

    result = LabelJoin(
        as_of=as_of,
        frame=joined.reset_index(drop=True),
        total=len(frame),
        matured=len(matured),
        observed=len(joined),
        pending=pending,
        censored=censored,
    )
    logger.info(
        "as_of %s: %d/%d matured (%.1f%%), %d labelled (coverage %.1f%%), %d pending, %d censored",
        stamp.date(),
        result.matured,
        result.total,
        result.maturity * 100,
        result.observed,
        result.coverage * 100,
        result.pending,
        result.censored,
    )
    return result


def load_served_predictions(settings: Any, source: str = "parquet") -> pd.DataFrame:
    """Load the decisions to be joined, from wherever serving recorded them.

    Mirrors ``monitoring.drift.load_current``: the demo replays a cohort to a
    Parquet file so the whole loop runs offline, while ``postgres`` reads what the
    running API actually served.
    """
    if source == "parquet":
        path = settings.served_predictions_parquet
        if not path.exists():
            raise FileNotFoundError(
                f"No served predictions at {path}. Run 'make labels-replay' first."
            )
        return pd.read_parquet(path)

    if source == "postgres":
        import psycopg

        with psycopg.connect(settings.postgres_dsn) as conn:
            rows = conn.execute(
                "SELECT application_id, predicted_at, model_version, probability, prediction"
                " FROM predictions WHERE application_id IS NOT NULL ORDER BY predicted_at"
            ).fetchall()
        if not rows:
            raise ValueError(
                "No served predictions carry an application_id yet. Predictions written "
                "before the label pipeline existed have no key and can never be joined."
            )
        return pd.DataFrame(
            rows,
            columns=[
                "application_id",
                "predicted_at",
                "model_version",
                "probability",
                "prediction",
            ],
        )

    raise ValueError(f"Unknown served-prediction source: {source}")


def join_labels_ignoring_time(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """The WRONG join, kept so the cost of getting it wrong can be measured.

    Joins every prediction to every outcome on the key alone. It is what you write
    when the label store looks like an ordinary dimension table, and it quietly
    imports the future: outcomes that had not matured on the evaluation date, and
    outcomes that had matured but had not yet been reported to anyone.

    Never call this from a real evaluation path. ``tests/test_labels.py`` asserts
    it disagrees with ``join_labels``, so that if the two ever coincide the test
    fails rather than the guarantee silently evaporating.
    """
    return predictions.merge(
        outcomes[["application_id", "outcome_at", "observed_at", "label"]],
        on="application_id",
        how="inner",
    )
