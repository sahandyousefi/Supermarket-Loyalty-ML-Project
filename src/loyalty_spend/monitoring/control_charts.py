r"""Control charts for the standardised residuals of scored members.

Once the spend of scored members is observed, each batch gives a mean
standardised residual :math:`x_t`, which is :math:`N(0, 1)` in control
when residuals are scaled by their in-control standard error. A shift in
the relation between features and spend moves the mean of :math:`x_t`.
Shewhart-type checks of single batches react slowly to small shifts;
charts with memory accumulate evidence over batches.

The exponentially weighted moving average (Roberts, 1959) is
:math:`z_t = \lambda x_t + (1 - \lambda) z_{t-1}` with :math:`z_0 = \mu_0`
and exact variance
:math:`\sigma^2 \frac{\lambda}{2 - \lambda}[1 - (1 - \lambda)^{2t}]`, so the
limits :math:`\mu_0 \pm L \sigma_{z_t}` widen over the first batches
instead of starting at their asymptotic width (Lucas and Saccucci, 1990).

The tabular CUSUM (Page, 1954) accumulates deviations beyond a reference
value :math:`k`: :math:`C^+_t = \max(0, x_t - \mu_0 - k\sigma + C^+_{t-1})`
and :math:`C^-_t = \max(0, \mu_0 - k\sigma - x_t + C^-_{t-1})`, with a
signal when either exceeds :math:`h\sigma`. With :math:`k` half the shift
to be detected it is optimal for that shift. The decision interval
:math:`h` for a target in-control run length follows from Siegmund's (1985)
approximation of the one-sided run length,
:math:`(e^{-2\Delta b} + 2\Delta b - 1) / (2\Delta^2)` with
:math:`b = h + 1.166` and :math:`\Delta` the drift beyond :math:`k`,
combined over both sides as :math:`1/\mathrm{ARL} = 1/\mathrm{ARL}^+ +
1/\mathrm{ARL}^-`.

The homogeneously weighted moving average (Abbas, 2018) gives the current
point weight :math:`\lambda` and spreads the rest evenly over all earlier
points: :math:`H_t = \lambda x_t + (1 - \lambda)\bar x_{t-1}` with
:math:`\bar x_0 = \mu_0`, so
:math:`\mathrm{Var}(H_t) = \sigma^2[\lambda^2 + (1 - \lambda)^2/(t - 1)]`
for :math:`t > 1` and :math:`\lambda^2\sigma^2` for :math:`t = 1`. Its
multivariate form (MHWMA, Adegoke et al., 2019) applies the same weights
to a vector of means and charts
:math:`T^2_t = H_t^\top \Sigma_{H_t}^{-1} H_t`, with
:math:`\Sigma_{H_t}` the covariance :math:`\Sigma` scaled by the same
factor. Here the vector holds the mean standardised residual of each tenure
band, so a shift confined to one band is not averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq


@dataclass
class ChartResult:
    statistic: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @property
    def signals(self) -> np.ndarray:
        return (self.statistic > self.upper) | (self.statistic < self.lower)

    @property
    def first_signal(self) -> int | None:
        """One-based index of the first out-of-control point, if any."""
        hits = np.flatnonzero(self.signals)
        return int(hits[0]) + 1 if hits.size else None


def ewma_chart(
    x: ArrayLike,
    lam: float = 0.1,
    L: float = 2.814,
    mu: float = 0.0,
    sigma: float = 1.0,
    exact: bool = True,
) -> ChartResult:
    """EWMA statistic with exact (or asymptotic) control limits."""
    if not 0.0 < lam <= 1.0:
        raise ValueError("lam must be in (0, 1]")
    x = np.asarray(x, float)
    z = np.empty_like(x)
    prev = mu
    for i, value in enumerate(x):
        prev = lam * value + (1 - lam) * prev
        z[i] = prev
    steps = np.arange(1, len(x) + 1)
    factor = 1 - (1 - lam) ** (2 * steps) if exact else np.ones(len(x))
    half = L * sigma * np.sqrt(lam / (2 - lam) * factor)
    return ChartResult(z, mu - half, mu + half)


@dataclass
class CusumResult(ChartResult):
    plus: np.ndarray | None = None
    minus: np.ndarray | None = None


def cusum_chart(
    x: ArrayLike,
    k: float = 0.5,
    h: float = 4.77,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> CusumResult:
    """Two-sided tabular CUSUM; the statistic is the larger side, signed."""
    x = (np.asarray(x, float) - mu) / sigma
    plus, minus = np.empty_like(x), np.empty_like(x)
    cp = cm = 0.0
    for i, value in enumerate(x):
        cp = max(0.0, value - k + cp)
        cm = max(0.0, -k - value + cm)
        plus[i], minus[i] = cp, cm
    stat = np.where(plus >= minus, plus, -minus)
    bound = np.full(len(x), h)
    return CusumResult(stat, -bound, bound, plus, minus)


def siegmund_arl(h: float, k: float, shift: float = 0.0) -> float:
    """Two-sided CUSUM run length for a mean shift, Siegmund's approximation."""
    b = h + 1.166

    def one_sided(delta: float) -> float:
        if abs(delta) < 1e-9:
            return b**2
        return (np.exp(-2 * delta * b) + 2 * delta * b - 1) / (2 * delta**2)

    return float(1 / (1 / one_sided(shift - k) + 1 / one_sided(-shift - k)))


def cusum_interval(k: float, arl0: float) -> float:
    """Decision interval ``h`` that gives in-control run length ``arl0``."""
    return float(brentq(lambda h: siegmund_arl(h, k) - arl0, 0.01, 50.0))


def _hwma_factor(n: int, lam: float) -> np.ndarray:
    t = np.arange(1, n + 1)
    tail = np.where(t > 1, (1 - lam) ** 2 / np.maximum(t - 1, 1), 0.0)
    return lam**2 + tail


def hwma_chart(
    x: ArrayLike,
    lam: float = 0.1,
    L: float = 2.8,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> ChartResult:
    """HWMA statistic with time-varying control limits."""
    if not 0.0 < lam <= 1.0:
        raise ValueError("lam must be in (0, 1]")
    x = np.asarray(x, float)
    past = np.r_[mu, np.cumsum(x)[:-1] / np.arange(1, len(x))] if len(x) else x
    h = lam * x + (1 - lam) * past
    half = L * sigma * np.sqrt(_hwma_factor(len(x), lam))
    return ChartResult(h, mu - half, mu + half)


def mhwma_chart(
    x: ArrayLike,
    lam: float = 0.1,
    h: float = 15.0,
    mu: ArrayLike | None = None,
    cov: ArrayLike | None = None,
) -> ChartResult:
    """Multivariate HWMA: Hotelling statistic of the weighted mean vector.

    ``x`` has one row per batch and one column per monitored mean. The
    statistic is non-negative, so the lower limit is zero and never crossed.
    """
    if not 0.0 < lam <= 1.0:
        raise ValueError("lam must be in (0, 1]")
    x = np.atleast_2d(np.asarray(x, float))
    n, p = x.shape
    mu_ = np.zeros(p) if mu is None else np.asarray(mu, float)
    cov_ = np.eye(p) if cov is None else np.asarray(cov, float)
    d = x - mu_
    past = np.vstack([np.zeros(p), np.cumsum(d, axis=0)[:-1]])
    past = past / np.maximum(np.arange(n), 1)[:, None]
    hv = lam * d + (1 - lam) * past
    inv = np.linalg.inv(cov_)
    t2 = np.einsum("ij,jk,ik->i", hv, inv, hv) / _hwma_factor(n, lam)
    return ChartResult(t2, np.full(n, -np.inf), np.full(n, h))
