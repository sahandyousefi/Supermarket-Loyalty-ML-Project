r"""Stratified randomisation and the stratified effect estimator.

Members are grouped into strata by tenure band and region, and within
every stratum a share :math:`p` of them, rounded at random, is assigned to
the promotion. Arms are then balanced exactly on the stratifying
attributes, which removes their contribution to the variance of the
estimate. The stratified estimator weights the difference in means of each
stratum by its size, with the Neyman variance
:math:`\sum_s w_s^2 (s^2_{T,s}/n_{T,s} + s^2_{C,s}/n_{C,s})`
(Imbens and Rubin, 2015, chapter 9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from loyalty_spend.experiment.adjustment import EffectEstimate


def stratified_assignment(
    strata: pd.DataFrame | pd.Series, share: float = 0.5, seed: int = 0
) -> np.ndarray:
    """Boolean treatment indicator, balanced within every stratum."""
    if not 0 < share < 1:
        raise ValueError("share must be in (0, 1)")
    keys = (
        strata.astype(str).agg("|".join, axis=1)
        if isinstance(strata, pd.DataFrame)
        else strata.astype(str)
    ).to_numpy()
    rng = np.random.default_rng(seed)
    treat = np.zeros(len(keys), bool)
    for key in np.unique(keys):
        idx = np.flatnonzero(keys == key)
        exact = share * len(idx)
        k = int(np.floor(exact)) + int(rng.uniform() < exact - np.floor(exact))
        treat[rng.permutation(idx)[:k]] = True
    return treat


def standardised_differences(X: pd.DataFrame, treat: ArrayLike) -> pd.Series:
    """Standardised mean difference of every column (dummies for categories)."""
    t = np.asarray(treat, bool)
    Z = pd.get_dummies(X, dtype=float)
    a, b = Z[t], Z[~t]
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2).replace(0, np.nan)
    return ((a.mean() - b.mean()) / pooled).fillna(0.0)


def stratified_effect(
    y: ArrayLike, treat: ArrayLike, strata: pd.Series
) -> EffectEstimate:
    y, t = np.asarray(y, float), np.asarray(treat, bool)
    keys = strata.astype(str).to_numpy()
    est, var, n = 0.0, 0.0, len(y)
    for key in np.unique(keys):
        sel = keys == key
        yt, yc = y[sel & t], y[sel & ~t]
        if len(yt) < 2 or len(yc) < 2:
            raise ValueError(f"stratum '{key}' needs two members in each arm")
        w = sel.sum() / n
        est += w * (yt.mean() - yc.mean())
        var += w**2 * (yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
    return EffectEstimate(float(est), float(np.sqrt(var)), "stratified")
