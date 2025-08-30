"""Prior sensitivity of the hierarchical model.

With a thousand members the fixed effects and the residual scale are
determined by the data, but the between-group scale rests on only as many
observations as there are groups (five regions). Refitting under several
priors shows which conclusions depend on the prior (Gelman, 2006).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from loyalty_spend.bayes.hierarchical import Priors, fit_hierarchical

PRIOR_SETS: dict[str, Priors] = {
    "default IG(1, 1)": Priors(a0=1.0, b0=1.0),
    "vague IG(0.01, 0.01)": Priors(a0=0.01, b0=0.01),
    "informative IG(5, 0.5)": Priors(a0=5.0, b0=0.5),
}


def prior_sensitivity(
    X: pd.DataFrame,
    y: ArrayLike,
    priors: dict[str, Priors] | None = None,
    seed: int = 0,
    **fit_kwargs: object,
) -> pd.DataFrame:
    """Posterior means and 90% intervals of sigma, tau and the group effects."""
    rows = []
    for name, prior in (priors or PRIOR_SETS).items():
        post = fit_hierarchical(X, y, priors=prior, seed=seed, **fit_kwargs)  # type: ignore[arg-type]
        sigma = np.sqrt(post.flat("sigma2"))
        tau = np.sqrt(post.flat("tau2"))
        u = post.flat("u")
        rows.append(
            {
                "prior": name,
                "sigma_mean": float(sigma.mean()),
                "tau_mean": float(tau.mean()),
                "tau_q05": float(np.quantile(tau, 0.05)),
                "tau_q95": float(np.quantile(tau, 0.95)),
                "max_abs_group_effect": float(np.abs(u.mean(axis=0)).max()),
            }
        )
    return pd.DataFrame(rows)
