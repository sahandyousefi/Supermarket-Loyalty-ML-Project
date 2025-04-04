"""Model explanations aggregated to the original features."""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance


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
