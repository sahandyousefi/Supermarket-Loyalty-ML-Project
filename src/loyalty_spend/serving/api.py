"""HTTP scoring service.

The application is built by a factory so that tests and deployments can pass
their own model path; ``LOYALTY_MODEL`` names the bundle otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from loyalty_spend import __version__
from loyalty_spend.data.cleaning import clean_loyalty
from loyalty_spend.models.persistence import ModelBundle, load_bundle
from loyalty_spend.ops.batch import score_batch
from loyalty_spend.serving.schemas import (
    Customer,
    Prediction,
    PredictRequest,
    PredictResponse,
)

DEFAULT_MODEL = "artifacts/model.joblib"


def create_app(model_path: str | Path | None = None) -> FastAPI:
    path = Path(model_path or os.environ.get("LOYALTY_MODEL", DEFAULT_MODEL))
    bundle: ModelBundle = load_bundle(path)
    app = FastAPI(title="Loyalty spend scoring", version=__version__)
    app.state.bundle = bundle

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

    return app


def customers_frame(customers: list[Customer]) -> pd.DataFrame:
    """Validated request rows as a cleaned frame with the training dtypes."""
    raw = pd.DataFrame([c.model_dump() for c in customers]).astype(object)
    frame, _ = clean_loyalty(
        raw.astype(str).replace("None", ""), impute=False, require_target=False
    )
    return frame
