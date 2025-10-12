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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


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
