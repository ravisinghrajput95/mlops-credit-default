"""The data contract must reject bad data. A gate that never fires is not a gate."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from credit_default.data.schema import (
    EDUCATION_OTHER,
    MARRIAGE_OTHER,
    clean,
    validate,
)


def test_clean_folds_undocumented_education_codes():
    frame = pd.DataFrame({"EDUCATION": [0, 1, 2, 3, 4, 5, 6], "MARRIAGE": [1] * 7})
    assert sorted(clean(frame)["EDUCATION"].unique()) == [1, 2, 3, EDUCATION_OTHER]


def test_clean_folds_undocumented_marriage_codes():
    frame = pd.DataFrame({"EDUCATION": [1] * 4, "MARRIAGE": [0, 1, 2, 3]})
    assert sorted(clean(frame)["MARRIAGE"].unique()) == [1, 2, MARRIAGE_OTHER]


def test_clean_preserves_row_count(frame):
    """Undocumented codes are folded, never dropped -- dropping would bias the sample."""
    assert len(clean(frame)) == len(frame)


def test_valid_frame_passes(frame):
    assert len(validate(clean(frame))) == len(frame)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("AGE", 5),           # below the documented minimum
        ("SEX", 3),           # outside the documented codes
        ("LIMIT_BAL", -100),  # a credit limit cannot be negative
        ("PAY_AMT1", -1),     # a payment tendered cannot be negative
        ("PAY_0", 99),        # outside the repayment-status range
    ],
)
def test_out_of_range_values_are_rejected(frame, column, bad_value):
    corrupted = clean(frame).copy()
    corrupted.loc[0, column] = bad_value
    with pytest.raises(SchemaErrors):
        validate(corrupted)


def test_negative_bill_amounts_are_allowed(frame):
    """A negative bill means the customer overpaid -- valid, not corrupt."""
    valid = clean(frame).copy()
    valid.loc[0, "BILL_AMT1"] = -5_000
    assert len(validate(valid)) == len(valid)


def test_nulls_are_rejected(frame):
    corrupted = clean(frame).astype({"AGE": "float64"})
    corrupted.loc[0, "AGE"] = None
    with pytest.raises(SchemaErrors):
        validate(corrupted)
