#!/usr/bin/env python
"""Induce SYNTHETIC drift in the current cohort, to demonstrate the monitor.

    This drift is fabricated. It is not observed in the UCI dataset.

The real dataset is a single static snapshot with no time dimension, so there is
no genuine drift to detect. Rather than pretend otherwise, this script applies an
explicit, documented shift so that the monitoring and retraining path can be
demonstrated end to end -- and so a reader can see exactly what was changed.

The scenario modelled is a credit downturn:

* repayment statuses slip later      (PAY_*     shifted up)
* issuers tighten limits             (LIMIT_BAL scaled down)
* customers pay down less each month (PAY_AMT*  scaled down)

Restore the clean cohort at any time with ``make split`` -- the split is seeded,
so it regenerates byte-identically.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from credit_default.config import TARGET, get_settings

logger = logging.getLogger(__name__)

PAY_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
PAY_AMT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]


def apply_drift(frame: pd.DataFrame, severity: float, seed: int) -> pd.DataFrame:
    """Shift the distributions. ``severity`` scales the whole effect (0-1)."""
    rng = np.random.default_rng(seed)
    drifted = frame.copy()

    # Repayment status slips later for a severity-scaled share of customers.
    slip = rng.random(len(drifted)) < severity
    for column in PAY_COLUMNS:
        drifted.loc[slip, column] = (drifted.loc[slip, column] + rng.integers(1, 3)).clip(-2, 9)

    # Issuers cut credit limits.
    drifted["LIMIT_BAL"] = (drifted["LIMIT_BAL"] * (1 - 0.35 * severity)).astype(int).clip(lower=10_000)

    # Customers pay down less.
    for column in PAY_AMT_COLUMNS:
        drifted[column] = (drifted[column] * (1 - 0.5 * severity)).astype(int).clip(lower=0)

    return drifted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--severity", type=float, default=0.6, help="0 = none, 1 = severe.")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to the current cohort.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()

    source = pd.read_parquet(settings.current_parquet)
    drifted = apply_drift(source, args.severity, settings.random_seed)
    destination = args.output or settings.current_parquet
    drifted.to_parquet(destination, index=False)

    logger.warning("This drift is SYNTHETIC -- injected deliberately, not observed in the data.")
    logger.info("Severity %.2f applied to %d rows -> %s", args.severity, len(drifted), destination)
    logger.info(
        "  LIMIT_BAL mean  %10.0f -> %10.0f",
        source["LIMIT_BAL"].mean(), drifted["LIMIT_BAL"].mean(),
    )
    logger.info(
        "  PAY_0 mean      %10.2f -> %10.2f",
        source["PAY_0"].mean(), drifted["PAY_0"].mean(),
    )
    logger.info(
        "  PAY_AMT1 mean   %10.0f -> %10.0f",
        source["PAY_AMT1"].mean(), drifted["PAY_AMT1"].mean(),
    )
    if TARGET in source:
        logger.info("  (labels left untouched; only the feature distribution is shifted)")
    logger.info("Restore the clean cohort with: make split")


if __name__ == "__main__":
    main()
