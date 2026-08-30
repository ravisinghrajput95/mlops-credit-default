"""Shared fixtures.

Tests run against a small synthetic frame rather than the real download, so the
suite stays fast, hermetic and runnable in CI with no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_default.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def make_frame(rows: int = 400, seed: int = 0) -> pd.DataFrame:
    """A frame that satisfies the cleaned contract and carries real signal.

    The target depends on PAY_0 so that a fitted model has something to learn and
    metric-based assertions are meaningful rather than coin flips.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {
        "LIMIT_BAL": rng.integers(10_000, 800_000, rows),
        "AGE": rng.integers(21, 79, rows),
        "SEX": rng.integers(1, 3, rows),
        "EDUCATION": rng.integers(1, 5, rows),
        "MARRIAGE": rng.integers(1, 4, rows),
    }
    for column in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        data[column] = rng.integers(-2, 9, rows)
    for index in range(1, 7):
        data[f"BILL_AMT{index}"] = rng.integers(-50_000, 500_000, rows)
        data[f"PAY_AMT{index}"] = rng.integers(0, 100_000, rows)

    frame = pd.DataFrame(data)
    probability = 1 / (1 + np.exp(-(frame["PAY_0"] - 1)))
    frame[TARGET] = (rng.random(rows) < probability).astype(int)
    return frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]]


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture
def application() -> dict[str, int]:
    """A single well-formed API payload."""
    payload = {
        "LIMIT_BAL": 200000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 35,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
    }
    for index in range(1, 7):
        payload[f"BILL_AMT{index}"] = 50_000 - index * 2_000
        payload[f"PAY_AMT{index}"] = 3_000
    return payload
