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
