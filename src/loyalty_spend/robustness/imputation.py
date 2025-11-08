"""Multiple imputation of missing amounts by chained equations.

Single imputation by the median treats an imputed value as if it had been
observed and understates uncertainty. Multiple imputation (Rubin, 1987)
creates ``m`` completed data sets in which each missing amount is drawn
from its predictive distribution given the other attributes, using
chained regressions with posterior sampling (van Buuren and
Groothuis-Oudshoorn, 2011). Analyses are run on each data set and pooled.
Missing categories keep the explicit "Unknown" level used everywhere else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from loyalty_spend.arrays import FloatArray
from loyalty_spend.data.schema import CATEGORICAL, NUMERIC


def multiple_imputations(
    X: pd.DataFrame, m: int = 5, max_iter: int = 10, seed: int = 0
) -> list[pd.DataFrame]:
    """``m`` completed copies of ``X`` with numeric gaps drawn, not filled."""
    numeric = [c for c in NUMERIC if c in X]
    categorical = [c for c in CATEGORICAL if c in X]
    cats = pd.get_dummies(
        X[categorical].astype("string").fillna("Unknown"), dtype=float
    )
    design = pd.concat(
        [X[numeric].astype(float).reset_index(drop=True), cats.reset_index(drop=True)],
        axis=1,
    )
    out = []
    for k in range(m):
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            sample_posterior=True,
            max_iter=max_iter,
            random_state=seed + k,
        )
        filled = imputer.fit_transform(design)[:, : len(numeric)]
        completed = X.copy().reset_index(drop=True)
        for j, c in enumerate(numeric):
            values = np.clip(filled[:, j], 0.0, None)
            if c == "items_in_first_month":
                values = np.round(values)
            completed[c] = values
        out.append(completed)
    return out


def pool_rubin(estimates: FloatArray, variances: FloatArray) -> pd.DataFrame:
    r"""Combine ``m`` sets of estimates and their sampling variances.

    With :math:`\bar Q` the mean estimate, :math:`W` the mean within-imputation
    variance and :math:`B` the between-imputation variance, the total variance
    is :math:`T = W + (1 + 1/m) B` and the reference distribution is t with
    :math:`\nu = (m - 1)(1 + 1/r)^2` degrees of freedom, where
    :math:`r = (1 + 1/m) B / W` (Rubin, 1987). The fraction of missing
    information :math:`(1 + 1/m) B / T` shows how much the gaps matter.
    """
    from scipy import stats

    q = np.asarray(estimates, float)
    u = np.asarray(variances, float)
    m = q.shape[0]
    q_bar = q.mean(axis=0)
    w = u.mean(axis=0)
    b = q.var(axis=0, ddof=1)
    t = w + (1 + 1 / m) * b
    r = (1 + 1 / m) * b / np.where(w > 0, w, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (m - 1) * (1 + 1 / r) ** 2
    df = np.where(np.isfinite(df), df, np.inf)
    se = np.sqrt(t)
    crit = stats.t.ppf(0.975, df)
    return pd.DataFrame(
        {
            "estimate": q_bar,
            "se": se,
            "ci_lower": q_bar - crit * se,
            "ci_upper": q_bar + crit * se,
            "df": df,
            "missing_information": (1 + 1 / m) * b / t,
        }
    )


def pooled_ols(
    completed: list[pd.DataFrame], y: FloatArray, columns: list[str]
) -> pd.DataFrame:
    """OLS on every completed data set, pooled with Rubin's rules."""
    estimates, variances, names = [], [], None
    for d in completed:
        Z = pd.get_dummies(d[columns], drop_first=True, dtype=float)
        Z.insert(0, "intercept", 1.0)
        names = list(Z.columns)
        A = Z.to_numpy()
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        sigma2 = resid @ resid / (len(y) - A.shape[1])
        cov = sigma2 * np.linalg.pinv(A.T @ A)
        estimates.append(beta)
        variances.append(np.diag(cov))
    table = pool_rubin(np.array(estimates), np.array(variances))
    table.insert(0, "term", names)
    return table
