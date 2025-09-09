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

With ``nu`` set, the errors are Student-t with ``nu`` degrees of freedom,
written as :math:`\varepsilon_i \mid \lambda_i \sim \mathcal N(0,
\sigma^2/\lambda_i)` with :math:`\lambda_i \sim \mathrm{Gamma}(\nu/2,
\nu/2)`; the weights :math:`\lambda_i` are drawn in each sweep and
down-weight outlying members (Geweke, 1993).

With ``variance_by`` set, each level of that column has its own residual
variance with an inverse-gamma prior, so tenure bands with more erratic
spend get wider predictive distributions.

Fixed effects enter through the usual design (standardized numerics, one-hot
categories); the grouping factor (region by default) gets partially pooled
effects. Several chains from dispersed starting points allow convergence
checks.
"""

from __future__ import annotations

from collections.abc import Callable
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
    fill: dict[str, float] = field(default_factory=dict)
    nu: float | None = None
    variance_column: str | None = None
    variance_levels: list[str] = field(default_factory=lambda: ["all"])

    @property
    def n_chains(self) -> int:
        return self.beta.shape[0]

    def summary(self) -> pd.DataFrame:
        """Posterior summary with convergence diagnostics for every scalar."""
        from loyalty_spend.bayes.diagnostics import summarise_draws

        draws = {name: self.beta[:, :, j] for j, name in enumerate(self.feature_names)}
        draws |= {f"u[{g}]": self.u[:, :, k] for k, g in enumerate(self.groups)}
        if self.sigma2.ndim == 2:
            draws["sigma"] = np.sqrt(self.sigma2)
        else:
            for k, level in enumerate(self.variance_levels):
                draws[f"sigma[{level}]"] = np.sqrt(self.sigma2[:, :, k])
        draws["tau"] = np.sqrt(self.tau2)
        return summarise_draws(draws)

    def _design_matrix(self, X: pd.DataFrame) -> np.ndarray:
        data = X.copy().fillna(self.fill)
        return np.column_stack([np.ones(len(data)), self.design.transform(data)])

    def predictive(
        self, X: pd.DataFrame, n_draws: int = 1000, seed: int = 0
    ) -> np.ndarray:
        """Posterior predictive draws of spend, shape ``(len(X), n_draws)``.

        Each draw uses one posterior sample of every parameter, so the
        spread includes parameter uncertainty as well as noise. Members of a
        group not seen in training get a fresh effect from N(0, tau^2).
        """
        rng = np.random.default_rng(seed)
        Z = self._design_matrix(X)
        beta, u = self.flat("beta"), self.flat("u")
        sigma, tau = np.sqrt(self.flat("sigma2")), np.sqrt(self.flat("tau2"))
        pick = rng.integers(0, len(tau), size=n_draws)
        if sigma.ndim == 1:
            member_sigma = np.broadcast_to(sigma[pick][None, :], (len(X), n_draws))
        else:
            levels = X[self.variance_column].astype(str).to_numpy()
            vi = np.searchsorted(self.variance_levels, levels).clip(
                max=len(self.variance_levels) - 1
            )
            member_sigma = sigma[pick][:, vi].T
        labels = X[self.group_column].astype(str).to_numpy()
        known = np.isin(labels, self.groups)
        gi = np.searchsorted(self.groups, labels).clip(max=len(self.groups) - 1)
        mean = Z @ beta[pick].T
        effect = np.where(
            known[:, None],
            u[pick][:, gi].T,
            rng.normal(size=(len(X), n_draws)) * tau[pick][None, :],
        )
        shocks = (
            rng.normal(size=(len(X), n_draws))
            if self.nu is None
            else rng.standard_t(self.nu, size=(len(X), n_draws))
        )
        noise = shocks * member_sigma
        return mean + effect + noise

    def log_likelihood(
        self, X: pd.DataFrame, y: ArrayLike, max_draws: int = 1000, seed: int = 0
    ) -> np.ndarray:
        """Pointwise log density of ``y`` under each posterior draw, ``(S, n)``."""
        from scipy import stats

        rng = np.random.default_rng(seed)
        y = np.asarray(y, float)
        beta, u = self.flat("beta"), self.flat("u")
        sigma2, tau2 = self.flat("sigma2"), self.flat("tau2")
        pick = rng.choice(len(tau2), size=min(max_draws, len(tau2)), replace=False)
        Z = self._design_matrix(X)
        labels = X[self.group_column].astype(str).to_numpy()
        known = np.isin(labels, self.groups)
        gi = np.searchsorted(self.groups, labels).clip(max=len(self.groups) - 1)
        mean = (Z @ beta[pick].T).T + np.where(known, u[pick][:, gi], 0.0)
        if sigma2.ndim == 1:
            var = np.broadcast_to(sigma2[pick][:, None], mean.shape)
        else:
            levels = X[self.variance_column].astype(str).to_numpy()
            vi = np.searchsorted(self.variance_levels, levels).clip(
                max=len(self.variance_levels) - 1
            )
            var = sigma2[pick][:, vi]
        # A group never seen in training adds its effect variance.
        var = var + np.where(known, 0.0, tau2[pick][:, None])
        scale = np.sqrt(var)
        if self.nu is None:
            return stats.norm.logpdf(y[None, :], mean, scale)
        return stats.t.logpdf(y[None, :], self.nu, mean, scale)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Posterior mean of expected spend for known groups."""
        Z = self._design_matrix(X)
        labels = X[self.group_column].astype(str).to_numpy()
        gi = np.searchsorted(self.groups, labels).clip(max=len(self.groups) - 1)
        u_mean = self.flat("u").mean(axis=0)
        effect = np.where(np.isin(labels, self.groups), u_mean[gi], 0.0)
        return Z @ self.flat("beta").mean(axis=0) + effect

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
    nu: float | None = None,
    variance_by: str | None = None,
    n_chains: int = 4,
    n_iter: int = 1500,
    n_warmup: int = 500,
    seed: int = 0,
) -> Posterior:
    priors = priors or Priors()
    fixed = list(DEFAULT_FIXED if fixed is None else fixed)
    data = X.copy()
    numeric = [c for c in fixed if pd.api.types.is_numeric_dtype(data[c])]
    fill = {c: float(data[c].median()) for c in numeric}
    data[numeric] = data[numeric].fillna(fill)
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
    lam = np.ones(n)
    if variance_by is None:
        levels, vi = ["all"], np.zeros(n, dtype=int)
    else:
        v_labels = data[variance_by].astype(str).to_numpy()
        levels = sorted(set(v_labels))
        vi = np.searchsorted(levels, v_labels)
    K = len(levels)
    n_v = np.bincount(vi, minlength=K)
    prior_prec = np.eye(p) / priors.beta_variance
    prior_prec[0, 0] = 1e-8

    keep = n_iter - n_warmup
    out = {
        "beta": np.empty((n_chains, keep, p)),
        "u": np.empty((n_chains, keep, G)),
        "sigma2": np.empty((n_chains, keep, K)),
        "tau2": np.empty((n_chains, keep)),
    }
    root = np.random.default_rng(seed)
    for c in range(n_chains):
        rng = np.random.default_rng(root.integers(2**32))
        sigma2: np.ndarray = np.full(K, float(np.var(y)) * rng.uniform(0.1, 2.0))
        tau2 = float(np.var(y)) * rng.uniform(0.01, 1.0)
        u = rng.normal(scale=np.sqrt(tau2), size=G)
        for it in range(n_iter):
            # Joint draw of fixed and group effects: drawing them one after
            # the other mixes badly because the intercept and the mean of the
            # group effects are almost unidentified separately.
            prior_block = np.diag(np.r_[np.diag(prior_prec), np.full(G, 1 / tau2)])
            # Per-member precision: Student-t weight over the member's variance.
            Wl = W * (lam / sigma2[vi])[:, None]
            prec = W.T @ Wl + prior_block
            chol = np.linalg.cholesky(prec)
            mean = np.linalg.solve(prec, Wl.T @ y)
            theta = mean + np.linalg.solve(chol.T, rng.standard_normal(p + G))
            beta, u = theta[:p], theta[p:]
            resid = y - Z @ beta
            e = resid - u[gi]
            if nu is not None:
                # Student-t errors as a scale mixture: precision weights per member.
                lam = rng.gamma((nu + 1) / 2, 2 / (nu + e**2 / sigma2[vi]))
            ssr = np.bincount(vi, weights=lam * e**2, minlength=K)
            sigma2 = 1 / rng.gamma(priors.a0 + n_v / 2, 1 / (priors.b0 + ssr / 2))
            tau2 = 1 / float(rng.gamma(priors.a0 + G / 2, 1 / (priors.b0 + u @ u / 2)))
            if it >= n_warmup:
                k = it - n_warmup
                out["beta"][c, k] = beta
                out["u"][c, k] = u
                out["sigma2"][c, k] = sigma2
                out["tau2"][c, k] = tau2
    if variance_by is None:
        out["sigma2"] = out["sigma2"][:, :, 0]
    return Posterior(
        out["beta"],
        out["u"],
        out["sigma2"],
        out["tau2"],
        groups,
        names,
        design,
        group,
        fill,
        nu,
        variance_by,
        levels,
    )


def posterior_sampler(
    posterior: Posterior, X: pd.DataFrame
) -> Callable[[np.random.Generator, int], np.ndarray]:
    """Sampler of joint spend draws for the reward simulation.

    Every row of a batch uses one posterior draw of the parameters for all
    customers at once, so uncertainty about the parameters moves everyone
    together, as it would in reality; noise is drawn independently per
    customer. This makes the spread of total cost wider than a plug-in
    forecast that fixes the parameters.
    """
    Z = posterior._design_matrix(X)
    labels = X[posterior.group_column].astype(str).to_numpy()
    known = np.isin(labels, posterior.groups)
    gi = np.searchsorted(posterior.groups, labels).clip(max=len(posterior.groups) - 1)
    beta, u = posterior.flat("beta"), posterior.flat("u")
    sigma2, tau2 = posterior.flat("sigma2"), posterior.flat("tau2")
    if sigma2.ndim > 1:
        levels = X[posterior.variance_column].astype(str).to_numpy()
        vi = np.searchsorted(posterior.variance_levels, levels).clip(
            max=len(posterior.variance_levels) - 1
        )

    def draw(rng: np.random.Generator, m: int) -> np.ndarray:
        pick = rng.integers(0, len(tau2), size=m)
        mean = beta[pick] @ Z.T
        effect = np.where(
            known[None, :],
            u[pick][:, gi],
            rng.normal(size=(m, len(X))) * np.sqrt(tau2[pick])[:, None],
        )
        sd = (
            np.sqrt(sigma2[pick])[:, None]
            if sigma2.ndim == 1
            else np.sqrt(sigma2[pick][:, vi])
        )
        shocks = (
            rng.normal(size=(m, len(X)))
            if posterior.nu is None
            else rng.standard_t(posterior.nu, size=(m, len(X)))
        )
        return mean + effect + shocks * sd

    return draw
