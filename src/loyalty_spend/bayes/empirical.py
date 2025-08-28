r"""Empirical Bayes shrinkage of group means.

Under the normal-normal model :math:`\bar y_g \mid \theta_g \sim
\mathcal N(\theta_g, \sigma^2 / n_g)` and :math:`\theta_g \sim
\mathcal N(\mu, \tau^2)`, the posterior mean of a group effect is

.. math::

    \hat\theta_g = w_g\,\bar y_g + (1 - w_g)\,\hat\mu, \qquad
    w_g = \frac{\hat\tau^2}{\hat\tau^2 + \hat\sigma^2 / n_g},

so small or noisy groups are pulled towards the overall mean (Efron and
Morris, 1975). :math:`\sigma^2` is the pooled within-group variance and
:math:`\tau^2` is estimated by the method of moments, truncated at zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def shrink_group_means(y: ArrayLike, groups: ArrayLike) -> pd.DataFrame:
    """Raw and shrunken means per group with the shrinkage weight."""
    frame = pd.DataFrame({"y": np.asarray(y, float), "g": np.asarray(groups)})
    stats = frame.groupby("g", observed=True)["y"].agg(["size", "mean", "var"])
    stats = stats.rename(columns={"size": "n", "mean": "raw_mean"})
    dof = (stats["n"] - 1).clip(lower=0)
    sigma2 = float((stats["var"].fillna(0) * dof).sum() / max(dof.sum(), 1))
    se2 = sigma2 / stats["n"]
    weights = stats["n"] / stats["n"].sum()
    mu = float((weights * stats["raw_mean"]).sum())
    between = float(np.average((stats["raw_mean"] - mu) ** 2, weights=weights))
    tau2 = max(0.0, between - float(np.average(se2, weights=weights)))
    w = tau2 / (tau2 + se2) if tau2 > 0 else se2 * 0.0
    stats["weight"] = w
    stats["shrunk_mean"] = w * stats["raw_mean"] + (1 - w) * mu
    stats.attrs.update(mu=mu, tau2=tau2, sigma2=sigma2)
    return stats.drop(columns="var").rename_axis("group").reset_index()
