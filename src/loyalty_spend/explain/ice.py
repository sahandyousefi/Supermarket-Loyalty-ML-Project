"""Partial dependence with individual and centred ICE curves.

An individual conditional expectation curve (Goldstein et al., 2015) traces
one member's forecast as a single feature moves over a grid, holding the
member's other attributes fixed; partial dependence is their average.
Centring every curve at the first grid point removes the level differences
between members, so remaining differences in shape reveal interactions:
parallel centred curves mean the feature acts additively.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

Predict = Callable[[pd.DataFrame], np.ndarray]


def ice_curves(
    predict: Predict,
    X: pd.DataFrame,
    feature: str,
    grid: np.ndarray | None = None,
    n_grid: int = 20,
    max_members: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid and an array of ICE curves with shape ``(members, grid)``."""
    data = X[X[feature].notna()]
    if len(data) > max_members:
        data = data.sample(max_members, random_state=seed)
    if grid is None:
        grid = np.unique(
            np.quantile(data[feature].astype(float), np.linspace(0.02, 0.98, n_grid))
        )
    curves = np.column_stack([predict(data.assign(**{feature: g})) for g in grid])
    return np.asarray(grid, float), curves


def pdp_summary(grid: np.ndarray, curves: np.ndarray) -> pd.DataFrame:
    """Partial dependence and the spread of the centred ICE curves."""
    centred = curves - curves[:, [0]]
    return pd.DataFrame(
        {
            "value": grid,
            "partial_dependence": curves.mean(axis=0),
            "centred_q10": np.quantile(centred, 0.1, axis=0),
            "centred_q90": np.quantile(centred, 0.9, axis=0),
        }
    )


def interaction_strength(curves: np.ndarray) -> float:
    """Spread of centred ICE curves relative to the range of the PD curve.

    Zero for a feature that acts additively; larger values mean the effect
    of the feature depends on the other attributes.
    """
    centred = curves - curves[:, [0]]
    pd_range = float(np.ptp(centred.mean(axis=0)))
    spread = float(np.mean(centred.std(axis=0)))
    return spread / pd_range if pd_range > 0 else float("inf") if spread > 0 else 0.0
