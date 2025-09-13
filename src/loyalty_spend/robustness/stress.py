"""Stress tests: how does accuracy degrade when inputs get worse?

A fitted model is scored on copies of held-out data with controlled damage:
relative noise on the amounts, a share of values deleted at random, or
categories replaced by an unseen level. The increase in RMSE over the clean
data shows which failures of an upstream system would hurt and by how much.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from loyalty_spend.data.schema import CATEGORICAL, NUMERIC

Predict = Callable[[pd.DataFrame], np.ndarray]


def add_noise(
    X: pd.DataFrame, columns: list[str], scale: float, rng: np.random.Generator
) -> pd.DataFrame:
    out = X.copy()
    for c in columns:
        out[c] = out[c] * (1 + rng.normal(scale=scale, size=len(out)))
    return out


def delete_values(
    X: pd.DataFrame, columns: list[str], share: float, rng: np.random.Generator
) -> pd.DataFrame:
    out = X.copy()
    for c in columns:
        hit = rng.uniform(size=len(out)) < share
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(object)
        out.loc[hit, c] = np.nan
    return out


def unseen_levels(
    X: pd.DataFrame, columns: list[str], share: float, rng: np.random.Generator
) -> pd.DataFrame:
    out = X.copy()
    for c in columns:
        out[c] = out[c].astype(object)
        hit = rng.uniform(size=len(out)) < share
        out.loc[hit, c] = "<new level>"
    return out


def stress_test(
    predict: Predict,
    X: pd.DataFrame,
    y: ArrayLike,
    severities: tuple[float, ...] = (0.05, 0.1, 0.2),
    n_reps: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """RMSE under each kind and severity of damage, averaged over ``n_reps`` draws."""
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    base = float(np.sqrt(np.mean((predict(X) - y) ** 2)))
    scenarios = {
        "noise on amounts": lambda s: add_noise(X, list(NUMERIC), s, rng),
        "missing amounts": lambda s: delete_values(X, list(NUMERIC), s, rng),
        "missing categories": lambda s: delete_values(X, list(CATEGORICAL), s, rng),
        "unseen categories": lambda s: unseen_levels(X, list(CATEGORICAL), s, rng),
    }
    rows = []
    for name, damage in scenarios.items():
        for s in severities:
            rmse = float(
                np.mean(
                    [
                        np.sqrt(np.mean((predict(damage(s)) - y) ** 2))
                        for _ in range(n_reps)
                    ]
                )
            )
            rows.append(
                {
                    "scenario": name,
                    "severity": s,
                    "rmse": rmse,
                    "rmse_ratio": rmse / base,
                }
            )
    return pd.DataFrame(rows)
