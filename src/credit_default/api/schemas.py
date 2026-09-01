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

    # The caller's own reference for this application. Optional, and echoed back
    # generated if omitted, because it is the only thing that makes a decision
    # addressable later: the outcome of a loan arrives months after the score,
    # and a prediction nobody can name is a prediction nobody can ever learn
    # from. It is deliberately NOT a model feature -- see main.predict, which
    # strips it before the frame reaches the pipeline.
    application_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Caller's reference for this application, echoed in the response and "
            "used to join the eventual outcome back to this decision. Generated "
            "if omitted."
        ),
    )

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
                    "application_id": "APP-000123",
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
    # Opt-in because it costs latency. A decline that will be communicated to the
    # applicant needs reasons; a bulk scoring job usually does not.
    explain: bool = Field(
        default=False,
        description="Return the principal reasons for each decision (adverse-action reasons).",
    )


class Reason(BaseModel):
    feature: str
    description: str = Field(description="Plain-language name the applicant can act on.")
    value: float
    contribution: float = Field(
        description="SHAP value; positive means this raised the predicted risk."
    )
    direction: Literal["increased_risk", "decreased_risk"]


class Prediction(BaseModel):
    # Echoed so the caller can report an outcome against this exact decision
    # later. Returning the score without a key would make the label pipeline
    # impossible from the outside, however well it worked internally.
    application_id: str
    probability: Annotated[float, Field(ge=0.0, le=1.0)]
    prediction: Literal[0, 1]
    reasons: list[Reason] | None = Field(
        default=None,
        description=(
            "Principal reasons, present only when explain=true. For a declined "
            "application these are the factors that increased risk, which is what "
            "an adverse-action notice must state."
        ),
    )


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
    # Reported so a caller can see what the model is not allowed to use. In a
    # regulated setting this is the sort of thing an auditor asks for.
    excluded_attributes: list[str]
    audited_attributes: list[str]
