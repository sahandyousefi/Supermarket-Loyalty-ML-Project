"""Repeated cross-validation and paired model comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CVResult:
    """Fold metrics and out-of-fold predictions of one model."""

    name: str
    folds: pd.DataFrame
    oof: np.ndarray
    n_train: float
    n_test: float

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {"model": self.name}  # type: ignore[dict-item]
        for col in ("rmse", "mae", "mape", "r2"):
            out[f"{col}_mean"] = float(self.folds[col].mean())
            out[f"{col}_std"] = float(self.folds[col].std(ddof=1))
        return out
