"""Unfolding the wide frame into an event table.

Six monthly observations per customer, stored as columns whose names encode a
date. The mapping is documented by UCI and is the only thing in this module that
could be got wrong, so it is stated once, in one table, and the round-trip is
asserted in the tests: unfold then refold must reproduce the original frame
exactly. A lossy unfold would quietly change the modelling data, and no
downstream test would notice.

Note the naming trap the source data sets: the repayment-status column for
September is ``PAY_0``, not ``PAY_1``, while the September bill is ``BILL_AMT1``.
Anyone deriving the month from the numeric suffix gets ``PAY_0`` wrong and
silently mis-dates a sixth of the observations.
"""

from __future__ import annotations

import pandas as pd

# (statement month, repayment-status column, bill column, payment column).
# April 2005 through September 2005, oldest first. Straight from the UCI data
# dictionary; see the module docstring for the PAY_0 trap.
MONTH_COLUMNS: list[tuple[str, str, str, str]] = [
    ("2005-04-01", "PAY_6", "BILL_AMT6", "PAY_AMT6"),
    ("2005-05-01", "PAY_5", "BILL_AMT5", "PAY_AMT5"),
    ("2005-06-01", "PAY_4", "BILL_AMT4", "PAY_AMT4"),
    ("2005-07-01", "PAY_3", "BILL_AMT3", "PAY_AMT3"),
    ("2005-08-01", "PAY_2", "BILL_AMT2", "PAY_AMT2"),
    ("2005-09-01", "PAY_0", "BILL_AMT1", "PAY_AMT1"),
]

# The label is "default in the month after the last observation": October 2005.
# Nothing may be observed at or after this date, which is the bound every
# point-in-time guarantee in this package is ultimately protecting.
OUTCOME_MONTH = pd.Timestamp("2005-10-01")

# Attributes with no time dimension in this dataset. Retained on the entity row
# rather than repeated against every month.
STATIC_COLUMNS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]

EVENT_COLUMNS = ["customer_id", "statement_month", "pay_status", "bill_amt", "pay_amt"]


def statement_months() -> list[pd.Timestamp]:
    """Every observable month, oldest first."""
    return [pd.Timestamp(month) for month, *_ in MONTH_COLUMNS]


def unfold(frame: pd.DataFrame, customer_ids: pd.Series | None = None) -> pd.DataFrame:
    """Wide frame -> one row per customer per statement month.

    ``customer_ids`` defaults to the frame's positional order. The dataset ships
    no identifier column, so one is manufactured; it must be derived the same way
    on both sides of any later join, which is why it is an explicit argument
    rather than something this function invents twice.
    """
    ids = (
        pd.Series(range(len(frame)), name="customer_id")
        if customer_ids is None
        else pd.Series(list(customer_ids), name="customer_id")
    )

    parts = []
    for month, pay_column, bill_column, amount_column in MONTH_COLUMNS:
        parts.append(
            pd.DataFrame(
                {
                    "customer_id": ids.to_numpy(),
                    "statement_month": pd.Timestamp(month),
                    "pay_status": frame[pay_column].to_numpy(),
                    "bill_amt": frame[bill_column].to_numpy(),
                    "pay_amt": frame[amount_column].to_numpy(),
                }
            )
        )

    events = pd.concat(parts, ignore_index=True)
    return events.sort_values(["customer_id", "statement_month"]).reset_index(drop=True)


def entities(
    frame: pd.DataFrame, customer_ids: pd.Series | None = None, target: str | None = None
) -> pd.DataFrame:
    """The non-time-varying side: one row per customer."""
    ids = (
        pd.Series(range(len(frame)), name="customer_id")
        if customer_ids is None
        else pd.Series(list(customer_ids), name="customer_id")
    )
    columns = [c for c in STATIC_COLUMNS if c in frame.columns]
    result = frame[columns].copy().reset_index(drop=True)
    result.insert(0, "customer_id", ids.to_numpy())
    if target and target in frame.columns:
        result[target] = frame[target].to_numpy()
    return result


def refold(events: pd.DataFrame, entity_frame: pd.DataFrame) -> pd.DataFrame:
    """Event table -> the original wide frame. Exists so the unfold can be proven lossless."""
    wide = entity_frame.set_index("customer_id")
    for month, pay_column, bill_column, amount_column in MONTH_COLUMNS:
        slice_ = events[events["statement_month"] == pd.Timestamp(month)].set_index("customer_id")
        wide[pay_column] = slice_["pay_status"]
        wide[bill_column] = slice_["bill_amt"]
        wide[amount_column] = slice_["pay_amt"]
    return wide.reset_index()
