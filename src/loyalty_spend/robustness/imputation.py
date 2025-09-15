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
