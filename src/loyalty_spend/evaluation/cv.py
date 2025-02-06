"""Repeated cross-validation and paired model comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold


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


def make_splits(
    n: int, n_splits: int = 5, n_repeats: int = 3, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed splits so that every model is evaluated on identical folds."""
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(cv.split(np.zeros(n)))
