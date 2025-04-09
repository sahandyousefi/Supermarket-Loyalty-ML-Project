"""Saving fitted models with the metadata needed to reuse them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
