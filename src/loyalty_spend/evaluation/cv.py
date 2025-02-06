"""Repeated cross-validation and paired model comparison."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold

from loyalty_spend.evaluation.metrics import regression_report


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


def cross_validate(
    name: str,
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    fit_hook: Callable | None = None,
) -> CVResult:
    """Fit a fresh clone on each split and score it on the held-out fold.

    Out-of-fold predictions are averaged over repeats, so ``oof`` holds one
    prediction per row.
    """
    rows = []
    oof_sum = np.zeros(len(y))
    oof_n = np.zeros(len(y))
    for i, (tr, te) in enumerate(splits):
        model = clone(estimator)
        model.fit(X.iloc[tr], y.iloc[tr])
        if fit_hook is not None:
            fit_hook(model)
        pred = model.predict(X.iloc[te])
        oof_sum[te] += pred
        oof_n[te] += 1
        rows.append({"fold": i, **regression_report(y.iloc[te], pred)})
    oof = oof_sum / np.maximum(oof_n, 1)
    n_test = float(np.mean([len(te) for _, te in splits]))
    return CVResult(name, pd.DataFrame(rows), oof, len(y) - n_test, n_test)
