"""Download the UCI credit-default dataset and normalise it to Parquet.

The archive ships a legacy .xls whose first row is a positional header (X1, X2, ...)
and whose second row holds the real column names, so it is read with `header=1`.
"""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from credit_default.config import DATA_URL, TARGET, get_settings

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT_SECONDS = 120


def download_archive(url: str = DATA_URL) -> bytes:
    logger.info("Downloading dataset from %s", url)
    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def parse_archive(payload: bytes) -> pd.DataFrame:
    """Extract the single .xls member and load it into a DataFrame."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith((".xls", ".xlsx"))]
        if not members:
            raise ValueError(f"No spreadsheet found in archive; members={archive.namelist()}")
        with archive.open(members[0]) as handle:
            return pd.read_excel(handle, header=1)


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename to stable identifiers and drop the surrogate key.

    The source names the target "default payment next month" (with spaces) and
    carries an ID column that must never reach the model as a feature.
    """
    frame = frame.rename(columns={c: str(c).strip() for c in frame.columns})
    frame = frame.rename(columns={"default payment next month": TARGET})

    if TARGET not in frame.columns:
        raise ValueError(f"Target column missing after rename; got {list(frame.columns)}")

    frame = frame.drop(columns=["ID"], errors="ignore")
    frame[TARGET] = frame[TARGET].astype("int8")
    return frame.reset_index(drop=True)


def ingest(destination: Path | None = None) -> Path:
    settings = get_settings()
    settings.ensure_dirs()
    destination = destination or settings.raw_parquet

    frame = normalise(parse_archive(download_archive()))
    frame.to_parquet(destination, index=False)

    logger.info("Wrote %s rows x %s cols to %s", len(frame), frame.shape[1], destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the UCI credit-default dataset.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")
    path = ingest(args.output)
    frame = pd.read_parquet(path)
    print(f"rows={len(frame)} cols={frame.shape[1]} positive_rate={frame[TARGET].mean():.4f}")


if __name__ == "__main__":
    main()
