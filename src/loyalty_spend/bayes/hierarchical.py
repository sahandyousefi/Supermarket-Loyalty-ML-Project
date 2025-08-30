r"""Hierarchical normal model fitted by Gibbs sampling.

.. math::

    y_i = x_i^\top \beta + u_{g(i)} + \varepsilon_i,\qquad
    \varepsilon_i \sim \mathcal N(0, \sigma^2),\qquad
    u_g \sim \mathcal N(0, \tau^2),

with :math:`\beta \sim \mathcal N(0, v_0 I)`, and inverse-gamma priors
:math:`\sigma^2, \tau^2 \sim \mathrm{IG}(a_0, b_0)`. All full conditionals
are conjugate (Gelman et al., 2013, ch. 15), so each sweep draws

* :math:`(\beta, u)` jointly from a multivariate normal, which avoids the
  slow mixing between the intercept and the group effects,
* :math:`\sigma^2` and :math:`\tau^2` from inverse-gamma distributions.

Fixed effects enter through the usual design (standardized numerics, one-hot
categories); the grouping factor (region by default) gets partially pooled
effects. Several chains from dispersed starting points allow convergence
checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_FIXED = [
    "first_month",
    "items_in_first_month",
    "loyalty_years",
    "joining_month",
    "promotion",
]


@dataclass
class Priors:
    beta_variance: float = 1e4
    a0: float = 1.0
    b0: float = 1.0


@dataclass
class Posterior:
    """Draws with a leading (chain, iteration) shape."""

    beta: np.ndarray
    u: np.ndarray
    sigma2: np.ndarray
    tau2: np.ndarray
    groups: list[str]
    feature_names: list[str]
    design: ColumnTransformer = field(repr=False)
    group_column: str = "region"

    @property
    def n_chains(self) -> int:
        return self.beta.shape[0]

    def summary(self) -> pd.DataFrame:
        """Posterior summary with convergence diagnostics for every scalar."""
        from loyalty_spend.bayes.diagnostics import summarise_draws

        draws = {name: self.beta[:, :, j] for j, name in enumerate(self.feature_names)}
        draws |= {f"u[{g}]": self.u[:, :, k] for k, g in enumerate(self.groups)}
        draws |= {"sigma": np.sqrt(self.sigma2), "tau": np.sqrt(self.tau2)}
        return summarise_draws(draws)

    def flat(self, name: str) -> np.ndarray:
        x = getattr(self, name)
        return x.reshape(-1, *x.shape[2:])


def _design(X: pd.DataFrame, fixed: list[str]) -> ColumnTransformer:
    numeric = [c for c in fixed if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in fixed if c not in numeric]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                categorical,
            ),
        ]
    )


def fit_hierarchical(
    X: pd.DataFrame,
    y: ArrayLike,
    group: str = "region",
    fixed: list[str] | None = None,
    priors: Priors | None = None,
    n_chains: int = 4,
    n_iter: int = 1500,
    n_warmup: int = 500,
    seed: int = 0,
) -> Posterior:
    priors = priors or Priors()
    fixed = list(DEFAULT_FIXED if fixed is None else fixed)
    data = X.copy()
    numeric = [c for c in fixed if pd.api.types.is_numeric_dtype(data[c])]
    data[numeric] = data[numeric].fillna(data[numeric].median())
    design = _design(data, fixed).fit(data)
    Z = np.column_stack([np.ones(len(data)), design.transform(data)])
    names = ["intercept", *design.get_feature_names_out()]
    y = np.asarray(y, float)
    labels = data[group].astype(str).to_numpy()
    groups = sorted(set(labels))
    gi = np.searchsorted(groups, labels)
    n, p, G = len(y), Z.shape[1], len(groups)
    D = np.zeros((n, G))
    D[np.arange(n), gi] = 1.0
    W = np.column_stack([Z, D])
    WtW = W.T @ W
    Wty = W.T @ y
    prior_prec = np.eye(p) / priors.beta_variance
    prior_prec[0, 0] = 1e-8

    keep = n_iter - n_warmup
    out = {
        "beta": np.empty((n_chains, keep, p)),
        "u": np.empty((n_chains, keep, G)),
        "sigma2": np.empty((n_chains, keep)),
        "tau2": np.empty((n_chains, keep)),
    }
    root = np.random.default_rng(seed)
    for c in range(n_chains):
        rng = np.random.default_rng(root.integers(2**32))
        sigma2 = float(np.var(y)) * rng.uniform(0.1, 2.0)
        tau2 = float(np.var(y)) * rng.uniform(0.01, 1.0)
        u = rng.normal(scale=np.sqrt(tau2), size=G)
        for it in range(n_iter):
            # Joint draw of fixed and group effects: drawing them one after
            # the other mixes badly because the intercept and the mean of the
            # group effects are almost unidentified separately.
            prior_block = np.diag(np.r_[np.diag(prior_prec), np.full(G, 1 / tau2)])
            prec = WtW / sigma2 + prior_block
            chol = np.linalg.cholesky(prec)
            mean = np.linalg.solve(prec, Wty / sigma2)
            theta = mean + np.linalg.solve(chol.T, rng.standard_normal(p + G))
            beta, u = theta[:p], theta[p:]
            resid = y - Z @ beta
            e = resid - u[gi]
            sigma2 = 1 / float(
                rng.gamma(priors.a0 + n / 2, 1 / (priors.b0 + e @ e / 2))
            )
            tau2 = 1 / float(rng.gamma(priors.a0 + G / 2, 1 / (priors.b0 + u @ u / 2)))
            if it >= n_warmup:
                k = it - n_warmup
                out["beta"][c, k] = beta
                out["u"][c, k] = u
                out["sigma2"][c, k] = sigma2
                out["tau2"][c, k] = tau2
    return Posterior(
        out["beta"], out["u"], out["sigma2"], out["tau2"], groups, names, design, group
    )
