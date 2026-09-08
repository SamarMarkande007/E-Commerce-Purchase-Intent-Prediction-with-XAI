"""Pydantic schemas for the FastAPI service.

``SessionInput`` mirrors the *raw* session columns from
``dataset/data_dictionary.csv`` — the same 17 fields a real session
would have before any feature engineering or preprocessing. The API
runs ``engineer_features()`` and the saved pipeline internally; callers
never need to know about the 5 engineered columns or the 80
post-preprocessing columns.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SessionInput(BaseModel):
    """One e-commerce session's raw behavioural data.

    Field bounds are set generously from the training data's observed
    ranges (see data_dictionary.csv) — not hard business rules, just a
    sanity check against obviously malformed input.
    """

    Administrative: int = Field(ge=0, description="Number of administrative pages visited")
    Administrative_Duration: float = Field(ge=0, description="Seconds spent on administrative pages")
    Informational: int = Field(ge=0, description="Number of informational pages visited")
    Informational_Duration: float = Field(ge=0, description="Seconds spent on informational pages")
    ProductRelated: int = Field(ge=0, description="Number of product-related pages visited")
    ProductRelated_Duration: float = Field(ge=0, description="Seconds spent on product-related pages")
    BounceRates: float = Field(ge=0, le=1, description="Average bounce rate of pages visited")
    ExitRates: float = Field(ge=0, le=1, description="Average exit rate of pages visited")
    PageValues: float = Field(
        ge=0,
        description=(
            "Average value of pages visited before a transaction. "
            "See docs/model_card.md: this model assumes it is genuinely "
            "known at scoring time (e.g. late-session / checkout)."
        ),
    )
    SpecialDay: float = Field(ge=0, le=1, description="Closeness of the visit to a special day (0=far, 1=on the day)")
    Month: Literal["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    OperatingSystems: int = Field(ge=1, le=8, description="Anonymised OS ID (categorical, not a magnitude)")
    Browser: int = Field(ge=1, le=13, description="Anonymised browser ID (categorical, not a magnitude)")
    Region: int = Field(ge=1, le=9, description="Anonymised region ID (categorical, not a magnitude)")
    TrafficType: int = Field(ge=1, le=20, description="Anonymised traffic-source ID (categorical, not a magnitude)")
    VisitorType: Literal["New_Visitor", "Returning_Visitor", "Other"]
    Weekend: bool

    model_config = {
        "json_schema_extra": {
            "example": {
                "Administrative": 2,
                "Administrative_Duration": 40.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 25,
                "ProductRelated_Duration": 620.5,
                "BounceRates": 0.01,
                "ExitRates": 0.02,
                "PageValues": 15.0,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": "Returning_Visitor",
                "Weekend": False,
            }
        }
    }


class ExplanationFactor(BaseModel):
    """One feature's contribution to a single prediction (see src/explain/shap_utils.py)."""

    feature: str
    shap_value: float = Field(description="Contribution on the model's log-odds scale; see docs/model_card.md")
    direction: Literal["increases", "decreases"]


class PredictionResponse(BaseModel):
    """The API's response for one scored session."""

    purchase_probability: float = Field(ge=0, le=1)
    will_purchase: bool = Field(description="purchase_probability >= decision_threshold")
    confidence: Literal["low", "medium", "high"] = Field(
        description="How far the probability sits from the decision threshold — a simple distance-based indicator, not a calibrated statistical confidence interval."
    )
    decision_threshold: float = Field(description="Value-based threshold from notebooks/03_modeling.ipynb, not 0.5")
    top_factors: list[ExplanationFactor] = Field(description="Top SHAP contributors for this session, largest influence first")


class HealthResponse(BaseModel):
    """The API's response for GET /health."""

    status: Literal["ok"]
    model_name: str
    decision_threshold: float
