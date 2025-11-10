r"""Reward totals for many schedules on the same draws of spend.

Applying a schedule to an :math:`m \times n` matrix of draws costs
:math:`O(mn)` work, and a threshold search repeats that for every
candidate. Both kinds of schedule depend on spend only through sums over
the members inside each band. After sorting every draw once and taking
prefix sums, the number of members below a threshold and their total spend
follow from one binary search per draw, so a candidate with :math:`K`
tiers costs :math:`O(mK \log n)` instead of :math:`O(mn)`.

With :math:`C(t)` the number of members with spend below :math:`t` and
:math:`S(t)` their total spend, a whole-spend schedule pays
:math:`\sum_k r_k\,[S(t_{k+1}) - S(t_k)]` and a marginal schedule pays
:math:`\sum_k r_k\,\{[S(t_{k+1}) - S(t_k)] - t_k[C(t_{k+1}) - C(t_k)]
+ (t_{k+1} - t_k)[n - C(t_{k+1})]\}`, with :math:`t_{K+1} = \infty`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from loyalty_spend.arrays import FloatArray, IntArray
from loyalty_spend.business.rewards import (
    MarginalSchedule,
    RewardSchedule,
)


class SortedDraws:
    def __init__(self, spend: ArrayLike) -> None:
        self.sorted = np.sort(np.clip(np.asarray(spend, float), 0.0, None), axis=1)
        self.m, self.n = self.sorted.shape
        self.rows = np.arange(self.m)
        prefix = np.cumsum(self.sorted, axis=1)
        self.prefix = np.concatenate([np.zeros((self.m, 1)), prefix], axis=1)
        self.total = prefix[:, -1] if self.n else np.zeros(self.m)

    def below(self, threshold: float) -> tuple[IntArray, FloatArray]:
        r"""Count and total spend of members strictly below ``threshold``.

        A binary search runs in every draw at once: each step compares one
        element per draw, so the work is :math:`O(m \log n)` and every
        comparison is exact. (Searching one flattened array with each draw
        shifted by an offset is faster but loses thresholds that differ
        from a spend value by less than the rounding error of the offset.)
        """
        if np.isinf(threshold):
            return np.full(self.m, self.n), self.total
        lo: IntArray = np.zeros(self.m, int)
        hi: IntArray = np.full(self.m, self.n)
        while np.any(lo < hi):
            mid = (lo + hi) // 2
            active = lo < hi
            probe = self.sorted[self.rows, np.minimum(mid, self.n - 1)]
            smaller = active & (probe < threshold)
            lo = np.where(smaller, mid + 1, lo)
            hi = np.where(active & ~smaller, mid, hi)
        return lo, self.prefix[self.rows, lo]

    def reach(self, threshold: float) -> float:
        """Expected share of members with spend at or above ``threshold``."""
        count, _ = self.below(threshold)
        return float(1.0 - count.mean() / self.n)

    def totals(self, schedule: RewardSchedule | MarginalSchedule) -> FloatArray:
        """Total reward in every draw."""
        thresholds = [t for t, _ in schedule.tiers]
        rates = [r for _, r in schedule.tiers]
        uppers = [*thresholds[1:], np.inf]
        marginal = isinstance(schedule, MarginalSchedule)
        out = np.zeros(self.m)
        for lo, hi, rate in zip(thresholds, uppers, rates, strict=True):
            if rate == 0:
                continue
            c_lo, s_lo = self.below(lo)
            c_hi, s_hi = self.below(hi)
            band = s_hi - s_lo
            if marginal:
                band = band - lo * (c_hi - c_lo)
                if np.isfinite(hi):
                    band = band + (hi - lo) * (self.n - c_hi)
            out += rate * band
        return out
