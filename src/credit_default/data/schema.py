"""The data contract.

This module is the single source of truth for what valid data looks like. It runs
in CI and as a DVC stage, so a malformed upstream file fails the build rather than
silently producing a degraded model.

Two documented quirks of the UCI dataset are handled deliberately:

* ``EDUCATION`` contains codes 0, 5 and 6 that the data dictionary never defines
  (it documents only 1-4), and ``MARRIAGE`` contains an undocumented 0.
  ``clean()`` folds these into the existing "other" category rather than dropping
  the rows -- they are ~2% of the data and dropping them would bias the sample.
* ``BILL_AMT*`` may be negative. That is not corruption: it means the customer
  overpaid and carries a credit balance, so the schema permits it.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from credit_default.config import TARGET

# Documented category codes, after cleaning.
EDUCATION_OTHER = 4
MARRIAGE_OTHER = 3
UNDOCUMENTED_EDUCATION = (0, 5, 6)
UNDOCUMENTED_MARRIAGE = (0,)

# Repayment status: -2 = no consumption, -1 = paid in full, 0 = revolving credit,
# 1..9 = months of payment delay.
PAY_MIN, PAY_MAX = -2, 9


class CreditDefaultSchema(pa.DataFrameModel):
    """Contract for the cleaned modelling frame."""

    LIMIT_BAL: Series[int] = pa.Field(gt=0, le=2_000_000)
    SEX: Series[int] = pa.Field(isin=[1, 2])
    EDUCATION: Series[int] = pa.Field(isin=[1, 2, 3, 4])
    MARRIAGE: Series[int] = pa.Field(isin=[1, 2, 3])
    AGE: Series[int] = pa.Field(ge=18, le=100)

    PAY_0: Series[int] = pa.Field(ge=PAY_MIN, le=PAY_MAX)
    PAY_2: Series[int] = pa.Field(ge=PAY_MIN, le=PAY_MAX)
    PAY_3: Series[int] = pa.Field(ge=PAY_MIN, le=PAY_MAX)
    PAY_4: Series[int] = pa.Field(ge=PAY_MIN, le=PAY_MAX)
    PAY_5: Series[int] = pa.Field(ge=PAY_MIN, le=PAY_MAX)
    PAY_6: Series[int] = pa.Field(ge=PAY_MIN, le=PAY_MAX)

    # Bill statements may be negative (customer in credit).
    BILL_AMT1: Series[int]
    BILL_AMT2: Series[int]
    BILL_AMT3: Series[int]
    BILL_AMT4: Series[int]
    BILL_AMT5: Series[int]
    BILL_AMT6: Series[int]

    # Payments are amounts tendered, so they cannot be negative.
    PAY_AMT1: Series[int] = pa.Field(ge=0)
    PAY_AMT2: Series[int] = pa.Field(ge=0)
    PAY_AMT3: Series[int] = pa.Field(ge=0)
    PAY_AMT4: Series[int] = pa.Field(ge=0)
    PAY_AMT5: Series[int] = pa.Field(ge=0)
    PAY_AMT6: Series[int] = pa.Field(ge=0)

    class Config:
        strict = False  # the target is validated separately; see validate()
        coerce = True


class LabelledSchema(CreditDefaultSchema):
    """Contract for frames that still carry ground truth."""

    default_payment_next_month: Series[int] = pa.Field(isin=[0, 1], alias=TARGET)


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Fold undocumented category codes into their documented "other" bucket."""
    frame = frame.copy()
    frame["EDUCATION"] = frame["EDUCATION"].replace(
        dict.fromkeys(UNDOCUMENTED_EDUCATION, EDUCATION_OTHER)
    )
    frame["MARRIAGE"] = frame["MARRIAGE"].replace(
        dict.fromkeys(UNDOCUMENTED_MARRIAGE, MARRIAGE_OTHER)
    )
    return frame


def validate(frame: pd.DataFrame, *, labelled: bool = True) -> pd.DataFrame:
    """Validate against the contract, raising ``pandera.errors.SchemaError`` on breach."""
    model = LabelledSchema if labelled else CreditDefaultSchema
    return model.validate(frame, lazy=True)
