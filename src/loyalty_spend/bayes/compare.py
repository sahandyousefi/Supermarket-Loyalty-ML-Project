r"""Predictive comparison of the Bayesian models with OLS.

WAIC estimates out-of-sample predictive accuracy from the pointwise
log-likelihood matrix :math:`\ell_{si} = \log p(y_i \mid \theta_s)`:

.. math::

    \mathrm{lppd} = \sum_i \log \frac1S \sum_s e^{\ell_{si}}, \qquad
    p_{\mathrm{WAIC}} = \sum_i \mathrm{Var}_s(\ell_{si}), \qquad
    \mathrm{WAIC} = -2(\mathrm{lppd} - p_{\mathrm{WAIC}})

(Watanabe, 2010; Gelman, Hwang and Vehtari, 2014). The cross-validated log
score evaluates the same quantity on held-out folds directly and is also
defined for OLS, whose predictive density is normal with the residual
standard deviation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats
from scipy.special import logsumexp

from loyalty_spend.bayes.hierarchical import Posterior


def waic(log_lik: np.ndarray) -> dict[str, float]:
    S = log_lik.shape[0]
    lppd_i = logsumexp(log_lik, axis=0) - np.log(S)
    p_i = log_lik.var(axis=0, ddof=1)
    elpd_i = lppd_i - p_i
    return {
        "elpd_waic": float(elpd_i.sum()),
        "p_waic": float(p_i.sum()),
        "waic": float(-2 * elpd_i.sum()),
        "se": float(np.sqrt(len(elpd_i) * elpd_i.var(ddof=1)) * 2),
    }


def ols_log_score(
    X_train: pd.DataFrame,
    y_train: ArrayLike,
    X_test: pd.DataFrame,
    y_test: ArrayLike,
    columns: list[str],
) -> np.ndarray:
    """Held-out log density of OLS with every column as a fixed effect."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    from loyalty_spend.bayes.hierarchical import _design

    y_train, y_test = np.asarray(y_train, float), np.asarray(y_test, float)
    fill = X_train[columns].median(numeric_only=True)
    model = make_pipeline(_design(X_train, columns), LinearRegression())
    model.fit(X_train[columns].fillna(fill), y_train)
    resid = y_train - model.predict(X_train[columns].fillna(fill))
    sd = float(np.sqrt(resid @ resid / max(len(resid) - 1, 1)))
    return stats.norm.logpdf(y_test, model.predict(X_test[columns].fillna(fill)), sd)


def cv_log_scores(
    X: pd.DataFrame,
    y: ArrayLike,
    models: dict[str, Callable[[pd.DataFrame, np.ndarray], Posterior]],
    splits: list[tuple[np.ndarray, np.ndarray]],
    ols_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Mean held-out log predictive density per member, by model and fold."""
    y = np.asarray(y, float)
    rows = []
    for k, (tr, te) in enumerate(splits):
        if ols_columns is not None:
            score = ols_log_score(X.iloc[tr], y[tr], X.iloc[te], y[te], ols_columns)
            rows.append({"fold": k, "model": "ols", "log_score": float(score.mean())})
        for name, fit in models.items():
            post = fit(X.iloc[tr], y[tr])
            ll = post.log_likelihood(X.iloc[te], y[te])
            lpd = logsumexp(ll, axis=0) - np.log(ll.shape[0])
            rows.append({"fold": k, "model": name, "log_score": float(lpd.mean())})
    return pd.DataFrame(rows)
