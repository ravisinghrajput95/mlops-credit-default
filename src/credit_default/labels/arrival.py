"""How outcomes arrive -- the SYNTHETIC part of the label pipeline.

The labels themselves are real: they are the UCI ground truth. What this module
invents is *when* each one shows up, because the dataset is a static snapshot
with no timestamps at all. Everything here is therefore an explicit, documented
assumption rather than a measurement, and it is kept in one file so the boundary
between real and simulated is a file boundary.

Two assumptions do real work, and both are stated so they can be argued with:

1. **Defaults are confirmed later than non-defaults.** A "no default" is known
   the instant the performance window closes -- nothing happened. A default is
   established through missed payments, contact attempts, collections and
   charge-off, all of which take time. So the reporting lag is drawn from a
   longer distribution for positives.

   This is the assumption that makes immature cohorts *dangerous* rather than
   merely incomplete: if you evaluate before the slow tail has landed, the
   observed default rate is too low, the model looks like it is over-predicting
   risk, and a monitoring dashboard reports drift that does not exist.

2. **Outcomes exist only for applicants who were approved.** A declined
   applicant never opens an account, so there is nothing to observe. This is not
   a simulation artefact -- it is the reject-inference problem, and it is the
   single biggest reason production label metrics are not comparable to offline
   test metrics.

   The one exception is the random holdout: a small share of applicants the
   policy would decline are approved anyway, specifically to keep the rejected
   range observable. Their selection probability is known by construction, which
   is what makes an unbiased estimate recoverable later (see ``performance``).
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from credit_default.labels.store import Outcome

logger = logging.getLogger(__name__)

SYNTHETIC_WARNING = (
    "Label ARRIVAL TIMES are synthetic. The labels are real UCI ground truth and the "
    "censoring is a real consequence of the model's own threshold, but the dataset has "
    "no time dimension, so decision dates and reporting lags are generated here."
)

# Gamma shape for the reporting lag. 4 gives a right-skewed distribution with a
# realistic tail -- most files land near the mean, a few take much longer -- while
# staying strictly positive, which an outcome that "arrived before it happened"
# would not be.
LAG_SHAPE = 4.0


@dataclass(frozen=True)
class ArrivalModel:
    """Assumptions about when an outcome becomes defined, and when it is reported."""

    performance_window_days: int
    reporting_lag_days: int
    reporting_lag_default_days: int
    shape: float = LAG_SHAPE

    def matures_at(self, predicted_at: pd.Series) -> pd.Series:
        """When the outcome becomes defined. A property of the product, not of us."""
        return predicted_at + pd.Timedelta(float(self.performance_window_days), unit="D")

    def reporting_lag(self, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Days between the window closing and the outcome reaching the store."""
        means = np.where(
            labels == 1,
            float(self.reporting_lag_default_days),
            float(self.reporting_lag_days),
        )
        # Gamma parameterised by mean: scale = mean / shape.
        return rng.gamma(shape=self.shape, scale=means / self.shape)


def assign_decision_times(
    rows: int,
    origin: dt.datetime,
    span_days: int,
    rng: np.random.Generator,
) -> pd.Series:
    """Spread decisions uniformly over an origination window.

    Synthetic, and uniform rather than seasonal on purpose: a fabricated seasonal
    pattern would show up in the drift monitor and be indistinguishable from a
    real one, which is exactly the kind of thing this project refuses to do.
    """
    start = pd.Timestamp(origin)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    offsets = rng.uniform(0.0, float(span_days), rows)
    stamps = [start + pd.Timedelta(float(d), unit="D") for d in offsets]
    # Floored to microseconds, the most datetime can hold. A decision time is a
    # business event; nanosecond precision here is an artefact of doing the
    # arithmetic in floating-point days, and it only ever surfaces as a warning.
    return pd.Series(stamps).dt.floor("us").sort_values().reset_index(drop=True)


def select_for_observation(
    predictions: pd.Series,
    holdout_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Which applicants generate an observable outcome.

    Approved applicants (``prediction == 0``) always do. Declined applicants do
    only if they fall into the random holdout, which is the sole mechanism by
    which the rejected range stays measurable.
    """
    declined = predictions.to_numpy() == 1
    holdout = declined & (rng.random(len(predictions)) < holdout_fraction)
    return np.asarray(~declined | holdout)


def emit_outcomes(
    served: pd.DataFrame,
    truth: pd.Series,
    model: ArrivalModel,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[Outcome], pd.DataFrame]:
    """Generate the outcome events a servicing system would eventually report.

    ``served`` must carry ``application_id``, ``predicted_at`` and ``prediction``.
    Returns the outcomes plus a frame describing what was censored and why, which
    is what the reporting layer needs to reason about coverage.
    """
    rng = np.random.default_rng(seed)
    logger.warning(SYNTHETIC_WARNING)

    frame = served.reset_index(drop=True).copy()
    frame["label"] = truth.reset_index(drop=True).astype(int).to_numpy()
    frame["matures_at"] = model.matures_at(frame["predicted_at"])

    observable = select_for_observation(frame["prediction"], holdout_fraction, rng)
    frame["observable"] = observable
    frame["holdout"] = observable & (frame["prediction"].to_numpy() == 1)

    lags = model.reporting_lag(frame["label"].to_numpy(), rng)
    # Floored to microseconds: datetime cannot hold nanoseconds, and a reporting
    # lag is a business process measured in days -- the precision is noise.
    frame["observed_at"] = (frame["matures_at"] + pd.to_timedelta(lags, unit="D")).dt.floor("us")
    # A censored applicant has no observation date, because no observation exists.
    frame.loc[~observable, "observed_at"] = pd.NaT

    reported = frame[observable]
    outcomes = [
        Outcome(
            application_id=str(application_id),
            outcome_at=matures_at.to_pydatetime(),
            observed_at=observed_at.to_pydatetime(),
            label=int(label),
        )
        for application_id, matures_at, observed_at, label in zip(
            reported["application_id"],
            reported["matures_at"],
            reported["observed_at"],
            reported["label"],
            strict=True,
        )
    ]

    logger.info(
        "Emitted %d outcomes from %d decisions (%d censored as declined, %d holdout approvals)",
        len(outcomes),
        len(frame),
        int((~observable).sum()),
        int(frame["holdout"].sum()),
    )
    return outcomes, frame
