"""HTTP scoring service.

The application is built by a factory so that tests and deployments can pass
their own model path; ``LOYALTY_MODEL`` names the bundle otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from loyalty_spend import __version__
from loyalty_spend.models.persistence import ModelBundle, load_bundle

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

    return app
