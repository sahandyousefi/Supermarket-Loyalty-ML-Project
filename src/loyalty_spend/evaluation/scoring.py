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


def interval_score(
    y: ArrayLike, lower: ArrayLike, upper: ArrayLike, alpha: float
) -> np.ndarray:
    """Interval score of a central ``1 - alpha`` interval (Gneiting and Raftery, 2007).

    The width plus a penalty of ``2 / alpha`` times the distance by which the
    outcome falls outside. It rewards narrow intervals only as long as they
    keep their coverage.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    y = np.asarray(y, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    if np.any(hi < lo):
        raise ValueError("upper bounds must not be below lower bounds")
    below = np.clip(lo - y, 0.0, None)
    above = np.clip(y - hi, 0.0, None)
    return (hi - lo) + 2.0 / alpha * (below + above)


def interval_summary(
    y: ArrayLike, lower: ArrayLike, upper: ArrayLike, alpha: float
) -> dict[str, float]:
    """Coverage with its binomial standard error, width and interval score."""
    y = np.asarray(y, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    inside = (y >= lo) & (y <= hi)
    n = len(y)
    cov = float(inside.mean())
    return {
        "target": 1.0 - alpha,
        "coverage": cov,
        "coverage_se": float(np.sqrt(cov * (1 - cov) / n)) if n else float("nan"),
        "mean_width": float(np.mean(hi - lo)),
        "median_width": float(np.median(hi - lo)),
        "interval_score": float(interval_score(y, lo, hi, alpha).mean()),
    }


def pit_gaussian(y: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> np.ndarray:
    """Probability integral transform values of normal forecasts."""
    return stats.norm.cdf(np.asarray(y, float), np.asarray(mu, float), sigma)


def pit_sample(y: ArrayLike, samples: ArrayLike, seed: int = 0) -> np.ndarray:
    """Randomised PIT of sample forecasts.

    Ties between the outcome and the draws are broken uniformly at random, so
    a calibrated forecast gives exactly uniform values even when the draws
    are discrete.
    """
    rng = np.random.default_rng(seed)
    x = np.atleast_2d(np.asarray(samples, float))
    y = np.asarray(y, float).reshape(-1, 1)
    below = (x < y).mean(axis=1)
    equal = (x == y).mean(axis=1)
    return below + rng.uniform(size=len(below)) * equal


def pit_quantiles(y: ArrayLike, q_pred: ArrayLike, taus: Sequence[float]) -> np.ndarray:
    """PIT values from quantile forecasts by linear interpolation of the CDF.

    Outcomes beyond the outermost quantiles get the outermost levels, which
    is conservative: those cases still show up in the tail bins.
    """
    q = np.sort(np.asarray(q_pred, float), axis=1)
    t = np.asarray(taus, float)
    y = np.asarray(y, float)
    return np.array([np.interp(v, row, t) for v, row in zip(y, q)])


def calibration_test(pit: ArrayLike, n_bins: int = 10) -> dict[str, float]:
    """Tests of uniformity of PIT values.

    A calibrated forecast has uniform PIT values. A U-shaped histogram means
    the forecast is too narrow, a hump means it is too wide, and a slope
    means it is biased. Returns the chi-square statistic over equal bins,
    its p-value and the KS p-value.
    """
    u = np.asarray(pit, float)
    counts = np.histogram(u, bins=n_bins, range=(0.0, 1.0))[0]
    chi2 = stats.chisquare(counts)
    ks = stats.kstest(u, "uniform")
    return {
        "chi2": float(chi2.statistic),
        "chi2_p_value": float(chi2.pvalue),
        "ks_p_value": float(ks.pvalue),
        "tail_share": float(np.mean((u < 0.05) | (u > 0.95))),
    }


def crps_quantiles(
    y: ArrayLike, q_pred: ArrayLike, taus: Sequence[float]
) -> np.ndarray:
    """CRPS approximated from quantile forecasts on an equally spaced grid.

    The CRPS equals twice the pinball loss integrated over all levels, so
    the average over an evenly spaced grid of levels, times two, converges
    to it as the grid becomes finer. Every model scored on the same grid is
    compared on the same footing.
    """
    q = np.asarray(q_pred, float)
    losses = np.column_stack([pinball_loss(y, q[:, j], t) for j, t in enumerate(taus)])
    return 2.0 * losses.mean(axis=1)
