"""HTTP scoring service.

The application is built by a factory so that tests and deployments can pass
their own model path; ``LOYALTY_MODEL`` names the bundle otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from loyalty_spend import __version__
from loyalty_spend.business.rewards import RewardSchedule, forecast_liability
from loyalty_spend.config import load_config
from loyalty_spend.data.cleaning import clean_loyalty
from loyalty_spend.data.schema import ID
from loyalty_spend.models.persistence import ModelBundle, load_bundle
from loyalty_spend.ops.batch import score_batch
from loyalty_spend.serving.schemas import (
    Customer,
    CustomerReward,
    Prediction,
    PredictRequest,
    PredictResponse,
    RewardRequest,
    RewardResponse,
)

DEFAULT_MODEL = "artifacts/model.joblib"
MAX_BATCH = 10_000


def create_app(
    model_path: str | Path | None = None, config_path: str | Path | None = None
) -> FastAPI:
    path = Path(model_path or os.environ.get("LOYALTY_MODEL", DEFAULT_MODEL))
    bundle: ModelBundle = load_bundle(path)
    app = FastAPI(title="Loyalty spend scoring", version=__version__)
    app.state.bundle = bundle
    max_batch = int(os.environ.get("LOYALTY_MAX_BATCH", MAX_BATCH))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    def version() -> dict[str, object]:
        return {
            "package_version": __version__,
            "model": bundle.name,
            "model_package_version": bundle.package_version,
            "data_fingerprint": bundle.metadata.get("data_fingerprint", ""),
            "features": bundle.features,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        check_batch(request.customers, max_batch)
        frame = customers_frame(request.customers)
        scores = score_batch(bundle, frame, alpha=request.alpha).scores
        return PredictResponse(
            model=bundle.name,
            coverage=1 - request.alpha,
            predictions=[
                Prediction(
                    customer_id=int(r.customer_id),
                    spend=round(float(r.prediction), 2),
                    lower=round(float(r.lower), 2),
                    upper=round(float(r.upper), 2),
                )
                for r in scores.itertuples()
            ],
        )

    schedule = RewardSchedule(load_config(config_path).rewards.tiers)

    @app.post("/rewards", response_model=RewardResponse)
    def rewards(request: RewardRequest) -> RewardResponse:
        check_batch(request.customers, max_batch)
        frame = customers_frame(request.customers)
        fc = forecast_liability(
            frame[ID],
            bundle.predict(frame),
            bundle.residuals,
            schedule,
            request.n_simulations,
            request.seed,
        )
        summary = fc.summary()
        return RewardResponse(
            expected_total=round(summary["expected_total"], 2),
            p05=round(summary["p05"], 2),
            p95=round(summary["p95"], 2),
            expected_shortfall_95=round(fc.tail_risk(0.95)["expected_shortfall"], 2),
            customers=[
                CustomerReward(
                    customer_id=int(r.customer_id),
                    predicted_spend=round(float(r.predicted_spend), 2),
                    expected_reward=round(float(r.expected_reward), 4),
                    tier_change_probability=round(float(r.tier_change_probability), 4),
                )
                for r in fc.per_customer.itertuples()
            ],
        )

    return app


def check_batch(customers: list[Customer], max_batch: int) -> None:
    """Reject oversized batches and duplicated customer ids before scoring."""
    if len(customers) > max_batch:
        raise HTTPException(
            status_code=413,
            detail=f"batch of {len(customers)} exceeds the limit of {max_batch}",
        )
    ids = [c.customer_id for c in customers]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        raise HTTPException(
            status_code=422,
            detail=f"duplicated customer_id values: {duplicated[:10]}",
        )


def customers_frame(customers: list[Customer]) -> pd.DataFrame:
    """Validated request rows as a cleaned frame with the training dtypes."""
    raw = pd.DataFrame([c.model_dump() for c in customers]).astype(object)
    frame, _ = clean_loyalty(
        raw.astype(str).replace("None", ""), impute=False, require_target=False
    )
    return frame
