r"""Overlap diagnostics and propensity trimming.

Inverse-probability weights :math:`1/e(x)` and :math:`1/(1-e(x))` explode
where the propensity approaches 0 or 1, and a few units then decide the
estimate. The effective sample size of Kish (1965),
:math:`(\sum_i w_i)^2 / \sum_i w_i^2`, shows how many units the weighted
sample is worth. Crump et al. (2009) derive the trimming threshold
:math:`\alpha` that minimises the asymptotic variance of the effect on the
retained population: no trimming if
:math:`\max g \le 2\,\mathbb E[g]` with :math:`g = 1/(e(1-e))`, otherwise the
:math:`\alpha` that solves
:math:`1/(\alpha(1-\alpha)) = 2\,\mathbb E[g \mid \alpha \le e \le 1-\alpha]`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from loyalty_spend.arrays import BoolArray


def effective_sample_size(weights: ArrayLike) -> float:
    w = np.asarray(weights, float)
    return float(w.sum() ** 2 / np.sum(w**2)) if np.any(w) else 0.0


def overlap_summary(
    propensity: ArrayLike, t: ArrayLike, bound: float = 0.05
) -> dict[str, float]:
    """Range of the scores by group, extreme shares and effective sample sizes."""
    e = np.asarray(propensity, float)
    t = np.asarray(t, int)
    treated, control = e[t == 1], e[t == 0]
    return {
        "treated_min": float(treated.min()),
        "treated_max": float(treated.max()),
        "control_min": float(control.min()),
        "control_max": float(control.max()),
        "share_extreme": float(np.mean((e < bound) | (e > 1 - bound))),
        "ess_treated": effective_sample_size(1 / treated),
        "ess_control": effective_sample_size(1 / (1 - control)),
        "n_treated": float(len(treated)),
        "n_control": float(len(control)),
    }


def crump_threshold(propensity: ArrayLike) -> float:
    """Optimal symmetric trimming threshold of Crump et al. (2009)."""
    e = np.asarray(propensity, float)
    g = 1.0 / (e * (1.0 - e))
    if np.max(g) <= 2.0 * np.mean(g):
        return 0.0
    # The retained set only changes at the observed values of min(e, 1 - e).
    # Between two of them the condition has the closed-form solution
    # alpha (1 - alpha) = 1 / (2 m), with m the mean of g over retained units.
    v = np.minimum(e, 1 - e)
    order = np.argsort(v)
    v_sorted, g_sorted = v[order], g[order]
    tail_mean = np.cumsum(g_sorted[::-1])[::-1] / np.arange(len(g), 0, -1)
    lower = 0.0
    for k in range(len(v_sorted)):
        m = tail_mean[k]
        alpha = 0.5 * (1.0 - np.sqrt(max(1.0 - 2.0 / m, 0.0)))
        if lower < alpha <= v_sorted[k]:
            return float(alpha)
        lower = v_sorted[k]
    return 0.1


def trim_mask(propensity: ArrayLike, alpha: float) -> BoolArray:
    """Units whose propensity lies in ``[alpha, 1 - alpha]``."""
    if not 0.0 <= alpha < 0.5:
        raise ValueError("alpha must be in [0, 0.5)")
    e = np.asarray(propensity, float)
    return np.asarray((e >= alpha) & (e <= 1 - alpha), bool)
