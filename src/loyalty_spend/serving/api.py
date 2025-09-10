"""HTTP scoring service.

The application is built by a factory so that tests and deployments can pass
their own model path; ``LOYALTY_MODEL`` names the bundle otherwise.
"""

from __future__ import annotations

import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response

from loyalty_spend import __version__
from loyalty_spend.business.rewards import RewardSchedule, forecast_liability
from loyalty_spend.config import load_config
from loyalty_spend.data.cleaning import clean_loyalty
from loyalty_spend.data.schema import ID
from loyalty_spend.explain.counterfactual import next_tier_counterfactual
from loyalty_spend.models.persistence import ModelBundle, load_bundle
from loyalty_spend.ops.batch import score_batch
from loyalty_spend.serving.metrics import RequestMetrics
from loyalty_spend.serving.schemas import (
    Customer,
    CustomerReward,
    NextTier,
    NextTierRequest,
    NextTierResponse,
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
    metrics = RequestMetrics()
    app.state.metrics = metrics

    @app.middleware("http")
    async def count_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        metrics.record(
            request.url.path, response.status_code, time.perf_counter() - start
        )
        return response

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_text() -> str:
        return metrics.render()

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
        metrics.add_customers("/predict", len(frame))
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
        metrics.add_customers("/rewards", len(frame))
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

    # Counterfactuals stay inside the range of first-month spend seen in
    # training; bundles saved before this was recorded search without a cap.
    reference_upper = float(bundle.metadata.get("first_month_q99", np.nan))

    @app.post("/next-tier", response_model=NextTierResponse)
    def next_tier(request: NextTierRequest) -> NextTierResponse:
        check_batch(request.customers, max_batch)
        frame = customers_frame(request.customers)
        out = []
        for i in range(len(frame)):
            row = frame.iloc[[i]]
            if pd.isna(row["first_month"].iloc[0]):
                raise HTTPException(422, "first_month is required for a counterfactual")
            upper = reference_upper if np.isfinite(reference_upper) else None
            cf = next_tier_counterfactual(bundle.predict, row, schedule, upper=upper)
            out.append(
                NextTier(
                    customer_id=int(row[ID].iloc[0]),
                    forecast=round(cf.prediction, 2),
                    next_threshold=cf.next_threshold,
                    gap=round(cf.gap, 2),
                    first_month_needed=(
                        None
                        if cf.required_value is None
                        else round(cf.required_value, 2)
                    ),
                    feasible=cf.feasible,
                )
            )
        return NextTierResponse(customers=out)

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
