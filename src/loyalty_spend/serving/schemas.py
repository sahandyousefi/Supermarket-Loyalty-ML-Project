"""Request and response models of the scoring service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Region = Literal["Americas", "Asia/Pacific", "Europe", "Middle East/Africa", "Unknown"]
Tenure = Literal["0-1", "1-3", "3-5", "5-10", "10+"]
Month = Literal[
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Unknown",
]  # fmt: skip


class Customer(BaseModel):
    customer_id: int = Field(ge=0)
    first_month: float | None = Field(default=None, ge=0)
    items_in_first_month: int | None = Field(default=None, ge=0)
    region: Region = "Unknown"
    loyalty_years: Tenure
    joining_month: Month = "Unknown"
    promotion: Literal["Yes", "No"]


class PredictRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1)
    alpha: float = Field(default=0.1, gt=0, lt=1)


class Prediction(BaseModel):
    customer_id: int
    spend: float
    lower: float
    upper: float


class PredictResponse(BaseModel):
    model: str
    coverage: float
    predictions: list[Prediction]


class RewardRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1)
    n_simulations: int = Field(default=2000, ge=100, le=20000)
    seed: int = 0


class CustomerReward(BaseModel):
    customer_id: int
    predicted_spend: float
    expected_reward: float
    tier_change_probability: float


class RewardResponse(BaseModel):
    expected_total: float
    p05: float
    p95: float
    expected_shortfall_95: float
    customers: list[CustomerReward]
