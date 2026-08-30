"""Cohort splitting.

Produces three disjoint cohorts from the raw file:

* ``reference`` -- everything the model is allowed to learn from, further split
  into ``train`` and ``test``.
* ``current``   -- held back entirely and replayed later as if it were live
  production traffic, giving the drift monitor a genuine unseen population.

All splits are stratified on the target and seeded, so ``dvc repro`` is
reproducible and a rerun produces byte-identical files.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_default.config import TARGET, Settings, get_settings
from credit_default.data.schema import clean, validate

logger = logging.getLogger(__name__)


def split_frame(frame: pd.DataFrame, settings: Settings) -> dict[str, pd.DataFrame]:
    reference, current = train_test_split(
        frame,
        test_size=settings.current_fraction,
        random_state=settings.random_seed,
        stratify=frame[TARGET],
    )
    train, test = train_test_split(
        reference,
        test_size=settings.test_fraction,
        random_state=settings.random_seed,
        stratify=reference[TARGET],
    )
    return {
        "reference": reference.reset_index(drop=True),
        "current": current.reset_index(drop=True),
        "train": train.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def run(source: Path | None = None) -> dict[str, Path]:
    settings = get_settings()
    settings.ensure_dirs()
    source = source or settings.raw_parquet

    frame = validate(clean(pd.read_parquet(source)))
    cohorts = split_frame(frame, settings)

    targets = {
        "reference": settings.reference_parquet,
        "current": settings.current_parquet,
        "train": settings.train_parquet,
        "test": settings.test_parquet,
    }
    summary = {}
    for name, path in targets.items():
        cohorts[name].to_parquet(path, index=False)
        summary[name] = {
            "rows": len(cohorts[name]),
            "positive_rate": round(float(cohorts[name][TARGET].mean()), 4),
        }
        logger.info("%-10s %6d rows -> %s", name, len(cohorts[name]), path)

    (settings.reports_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the dataset into cohorts.")
    parser.add_argument("--source", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")
    run(args.source)


if __name__ == "__main__":
    main()
