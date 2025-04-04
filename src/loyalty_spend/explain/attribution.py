"""Model explanations aggregated to the original features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from loyalty_spend.features.preprocess import original_feature


def grouped_permutation_importance(
    model, X: pd.DataFrame, y, n_repeats: int = 10, seed: int = 0
) -> pd.DataFrame:
    """Increase in RMSE when one raw feature is shuffled.

    Permuting the raw column moves every encoded column derived from it
    together, so one-hot blocks are treated as a single feature.
    """
    res = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="neg_root_mean_squared_error",
    )
    out = pd.DataFrame(
        {
            "feature": X.columns,
            "rmse_increase": res.importances_mean,
            "std": res.importances_std,
        }
    )
    return out.sort_values("rmse_increase", ascending=False).reset_index(drop=True)


def _group(values: np.ndarray, names: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=names)
    return frame.T.groupby([original_feature(n) for n in names]).sum().T


def linear_contributions(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Exact additive contributions of a linear pipeline.

    For ``f(x) = b + w'z(x)`` the contribution of feature ``j`` is
    ``w_j (z_j(x) - mean z_j)``, which equals its Shapley value when the
    features are treated as independent. Encoded columns are summed back to
    their raw feature.
    """
    prep, model = pipeline[:-1], pipeline[-1]
    Z = np.asarray(prep.transform(X), float)
    names = list(prep.get_feature_names_out())
    values = (Z - Z.mean(axis=0)) * np.ravel(model.coef_)
    return _group(values, names).set_index(X.index)
