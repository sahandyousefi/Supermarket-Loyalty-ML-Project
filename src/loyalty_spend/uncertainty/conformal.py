"""Distribution-free prediction intervals.

A point forecast of spend is not enough to budget for rewards; the finance
team needs a range with a stated probability of containing the outcome.
Conformal methods give intervals with finite-sample marginal coverage of at
least ``1 - alpha`` under exchangeability, for any underlying model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile of the scores.

    Returns infinity when the calibration set is too small for the level.
    """
    scores = np.sort(np.asarray(scores, float))
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float("inf") if k > n else float(scores[k - 1])


@dataclass
class Intervals:
    lower: np.ndarray
    upper: np.ndarray

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def coverage(self, y) -> float:
        y = np.asarray(y, float)
        return float(np.mean((y >= self.lower) & (y <= self.upper)))
