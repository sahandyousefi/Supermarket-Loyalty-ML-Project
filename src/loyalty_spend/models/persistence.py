"""Saving fitted models with the metadata needed to reuse them."""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from loyalty_spend import __version__
from loyalty_spend.data.schema import FEATURES


@dataclass
class ModelBundle:
    """A fitted model, its out-of-fold residuals and provenance."""

    name: str
    model: Any
    residuals: np.ndarray
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    metadata: dict[str, Any] = field(default_factory=dict)
    package_version: str = __version__

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.features])


def data_fingerprint(df: pd.DataFrame) -> str:
    """Short hash of a frame's contents, to tie a model to its training data."""
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values)
    return digest.hexdigest()[:12]


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle.metadata.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
    bundle.metadata.setdefault("python", platform.python_version())
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str | Path) -> ModelBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path} does not contain a ModelBundle")
    return bundle
