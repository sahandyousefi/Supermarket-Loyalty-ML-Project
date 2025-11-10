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

from loyalty_spend.arrays import FloatArray, IntArray


def _order(tau_hat: FloatArray, seed: int) -> IntArray:
    # Random tie-breaking so that constant predictions give the random line.
    # The extra entropy keeps the stream apart from a caller's data generator.
    rng = np.random.default_rng((seed, 17))
    return np.asarray(np.lexsort((rng.uniform(size=len(tau_hat)), -tau_hat)))


def qini_curve(
    tau_hat: ArrayLike, t: ArrayLike, y: ArrayLike, n_points: int = 100, seed: int = 0
) -> pd.DataFrame:
    """Qini curve of a ranking on randomised data."""
    order = _order(np.asarray(tau_hat, float), seed)
    treated = np.asarray(t, int)[order]
    outcome = np.asarray(y, float)[order]
    n_t, n_c = np.cumsum(treated), np.cumsum(1 - treated)
    y_t, y_c = np.cumsum(outcome * treated), np.cumsum(outcome * (1 - treated))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(n_c > 0, y_t - y_c * n_t / n_c, 0.0)
    idx = np.unique(np.linspace(0, len(outcome) - 1, n_points).round().astype(int))
    frac = (idx + 1) / len(outcome)
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


def effect_by_group(
    tau_hat: ArrayLike, scores: ArrayLike, n_groups: int = 10
) -> pd.DataFrame:
    """Predicted and doubly robust estimated effect by quantile group of ``tau_hat``.

    Group 1 holds the members with the largest predicted effect. If the
    ranking carries information, the estimated effect falls from the first
    group to the last; the standard errors come from the spread of the
    scores within each group.
    """
    tau_hat, psi = np.asarray(tau_hat, float), np.asarray(scores, float)
    ranks = pd.Series(-tau_hat).rank(method="first").to_numpy()
    groups = np.ceil(ranks * n_groups / len(tau_hat)).astype(int)
    frame = pd.DataFrame({"group": groups, "predicted": tau_hat, "score": psi})
    out = frame.groupby("group").agg(
        n=("score", "size"),
        predicted=("predicted", "mean"),
        estimated=("score", "mean"),
        sd=("score", "std"),
    )
    out["se"] = out["sd"] / np.sqrt(out["n"])
    return out.drop(columns="sd").reset_index()
