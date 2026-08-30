"""Request and response models for the inference API.

These mirror the pandera data contract in ``credit_default.data.schema`` so that
input is rejected at the edge with a 422 rather than reaching the model and
producing a confident prediction on nonsense.

The bounds here are deliberately the *raw* documented ranges (EDUCATION 0-6,
MARRIAGE 0-3) rather than the cleaned ones. Callers send real-world codes, and
the same ``clean()`` used in training folds the undocumented values into their
"other" bucket before inference -- applying the identical transformation on both
sides is what keeps training and serving consistent.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Money = Annotated[int, Field(ge=-1_000_000, le=10_000_000)]
Payment = Annotated[int, Field(ge=0, le=10_000_000)]
# -2 no consumption, -1 paid in full, 0 revolving, 1..9 months delinquent
RepaymentStatus = Annotated[int, Field(ge=-2, le=9)]


class CreditApplication(BaseModel):
    """One customer's billing history, as the model expects it."""

    LIMIT_BAL: Annotated[int, Field(gt=0, le=2_000_000, description="Credit limit (NT$)")]
    SEX: Literal[1, 2] = Field(description="1 = male, 2 = female")
    EDUCATION: Annotated[int, Field(ge=0, le=6, description="1 = graduate school ... 4 = other")]
    MARRIAGE: Annotated[int, Field(ge=0, le=3, description="1 = married, 2 = single, 3 = other")]
    AGE: Annotated[int, Field(ge=18, le=100)]

    PAY_0: RepaymentStatus
    PAY_2: RepaymentStatus
    PAY_3: RepaymentStatus
    PAY_4: RepaymentStatus
    PAY_5: RepaymentStatus
    PAY_6: RepaymentStatus

    BILL_AMT1: Money
    BILL_AMT2: Money
    BILL_AMT3: Money
    BILL_AMT4: Money
    BILL_AMT5: Money
    BILL_AMT6: Money

    PAY_AMT1: Payment
    PAY_AMT2: Payment
    PAY_AMT3: Payment
    PAY_AMT4: Payment
    PAY_AMT5: Payment
    PAY_AMT6: Payment

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
                    "BILL_AMT1": 50000,
                    "BILL_AMT2": 48000,
                    "BILL_AMT3": 46000,
                    "BILL_AMT4": 44000,
                    "BILL_AMT5": 42000,
                    "BILL_AMT6": 40000,
                    "PAY_AMT1": 3000,
                    "PAY_AMT2": 3000,
                    "PAY_AMT3": 3000,
                    "PAY_AMT4": 3000,
                    "PAY_AMT5": 3000,
                    "PAY_AMT6": 3000,
                }
            ]
        }
    }


class PredictionRequest(BaseModel):
    applications: Annotated[list[CreditApplication], Field(min_length=1, max_length=1000)]


class Prediction(BaseModel):
    probability: Annotated[float, Field(ge=0.0, le=1.0)]
    prediction: Literal[0, 1]


class PredictionResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str | None = None
    threshold: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_source: str
    model_version: str | None
    registered_model_name: str
    alias: str
    threshold: float
    features: list[str]
