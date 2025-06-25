r"""Evaluation of effect rankings: Qini and gain curves.

Members are sorted by their predicted effect and the curve tracks the
incremental spend gained by targeting the top share :math:`\phi` of them.
The Qini curve (Radcliffe, 2007) of a randomised campaign is

.. math::

    Q(\phi) = Y_T(\phi) - Y_C(\phi)\,\frac{N_T(\phi)}{N_C(\phi)},

with cumulative outcomes :math:`Y` and counts :math:`N` of treated and
control members among the top :math:`\phi`. In observational data the
treated and control members of a top slice are not comparable, so the
doubly robust gain curve sums the AIPW scores of the top slice instead,
which estimates the same quantity without assuming randomisation. Both are
compared with the straight line of random targeting; the area between the
curve and that line, divided by the number of members, is the Qini
coefficient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def _order(tau_hat: np.ndarray, seed: int) -> np.ndarray:
    # Random tie-breaking so that constant predictions give the random line.
    # The extra entropy keeps the stream apart from a caller's data generator.
    rng = np.random.default_rng((seed, 17))
    return np.lexsort((rng.uniform(size=len(tau_hat)), -tau_hat))


def qini_curve(
    tau_hat: ArrayLike, t: ArrayLike, y: ArrayLike, n_points: int = 100, seed: int = 0
) -> pd.DataFrame:
    """Qini curve of a ranking on randomised data."""
    tau_hat, t, y = np.asarray(tau_hat, float), np.asarray(t, int), np.asarray(y, float)
    order = _order(tau_hat, seed)
    t, y = t[order], y[order]
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(n_c > 0, y_t - y_c * n_t / n_c, 0.0)
    idx = np.unique(np.linspace(0, len(y) - 1, n_points).round().astype(int))
    frac = (idx + 1) / len(y)
    return pd.DataFrame({"fraction": frac, "gain": q[idx], "random": frac * q[-1]})


def dr_gain_curve(
    tau_hat: ArrayLike, scores: ArrayLike, n_points: int = 100, seed: int = 0
) -> pd.DataFrame:
    """Cumulative doubly robust effect of treating the top-ranked members."""
    tau_hat, psi = np.asarray(tau_hat, float), np.asarray(scores, float)
    gain = np.cumsum(psi[_order(tau_hat, seed)])
    idx = np.unique(np.linspace(0, len(psi) - 1, n_points).round().astype(int))
    frac = (idx + 1) / len(psi)
    return pd.DataFrame(
        {"fraction": frac, "gain": gain[idx], "random": frac * gain[-1]}
    )


def qini_coefficient(curve: pd.DataFrame, n: int) -> float:
    """Area between a gain curve and random targeting, per member."""
    diff = (curve["gain"] - curve["random"]).to_numpy()
    return float(np.trapezoid(diff, curve["fraction"].to_numpy()) / n)
