"""Model comparison across simulated scenarios."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from loyalty_spend.data.simulate import SCENARIOS, simulate
from loyalty_spend.evaluation.cv import cross_validate, make_splits
from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.models.registry import available_models, build_model, model_family


def run_study(
    train: pd.DataFrame,
    models: Sequence[str],
    scenarios: Sequence[str] = tuple(SCENARIOS),
    n: int = 2000,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """Cross-validated RMSE of every model in every scenario.

    ``excess_rmse`` is the RMSE above the Bayes error of the scenario and
    ``relative_excess`` divides it by the Bayes error, so scenarios with
    different noise levels can be read on one scale.
    """
    models = [m for m in models if m in available_models()]
    rows = []
    for s_name in scenarios:
        data = simulate(train, SCENARIOS[s_name], n=n, seed=seed)
        X, y = split_xy(data.frame)
        splits = make_splits(len(y), n_splits, n_repeats, seed)
        for m in models:
            res = cross_validate(m, build_model(m), X, y, splits)
            rmse = float(res.folds["rmse"].mean())
            rows.append(
                {
                    "scenario": s_name,
                    "model": m,
                    "family": model_family(m),
                    "rmse": rmse,
                    "rmse_std": float(res.folds["rmse"].std(ddof=1)),
                    "bayes_rmse": data.bayes_rmse,
                    "excess_rmse": rmse - data.bayes_rmse,
                    "relative_excess": (rmse - data.bayes_rmse) / data.bayes_rmse,
                }
            )
    return pd.DataFrame(rows)
