r"""Proper scoring rules for probabilistic forecasts.

A scoring rule is proper when the forecaster minimises its expected score by
reporting their true belief (Gneiting and Raftery, 2007). The continuous
ranked probability score of a predictive distribution :math:`F` at outcome
:math:`y` is

.. math::

    \mathrm{CRPS}(F, y)
    = \int \big(F(z) - \mathbb{1}\{z \ge y\}\big)^2\,dz
    = \mathbb{E}_F|X - y| - \tfrac12\,\mathbb{E}_F|X - X'|,

and is measured in the units of the target. For a point forecast it reduces
to the absolute error, so CRPS values can be compared with the MAE of the
point models.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def pinball_loss(y: ArrayLike, q_pred: ArrayLike, tau: float) -> np.ndarray:
    """Quantile (pinball) loss of a ``tau`` quantile forecast, per observation."""
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0, 1)")
    diff = np.asarray(y, float) - np.asarray(q_pred, float)
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def mean_pinball(y: ArrayLike, q_pred: ArrayLike, taus: Sequence[float]) -> np.ndarray:
    """Mean pinball loss for each column of ``q_pred`` (one column per level)."""
    q = np.asarray(q_pred, float)
    if q.ndim != 2 or q.shape[1] != len(taus):
        raise ValueError("q_pred must have one column per quantile level")
    return np.array([pinball_loss(y, q[:, j], t).mean() for j, t in enumerate(taus)])


def crps_gaussian(y: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> np.ndarray:
    """Closed-form CRPS of a normal forecast (Gneiting et al., 2005)."""
    sigma = np.asarray(sigma, float)
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive")
    z = (np.asarray(y, float) - np.asarray(mu, float)) / sigma
    return sigma * (
        z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)
    )


def crps_sample(y: ArrayLike, samples: ArrayLike) -> np.ndarray:
    """CRPS of an ensemble or Monte Carlo forecast, one row of draws per case.

    Uses the identity ``E|X - X'| = 2 / m^2 * sum_i (2i - m - 1) x_(i)`` for
    sorted draws, which avoids the quadratic pairwise sum.
    """
    x = np.sort(np.atleast_2d(np.asarray(samples, float)), axis=1)
    y = np.asarray(y, float).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("need one row of samples per observation")
    m = x.shape[1]
    first = np.abs(x - y[:, None]).mean(axis=1)
    weights = 2 * np.arange(1, m + 1) - m - 1
    spread = 2 * (x * weights).sum(axis=1) / m**2
    return first - 0.5 * spread
