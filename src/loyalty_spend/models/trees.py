"""Tree ensembles."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline

from loyalty_spend.features.preprocess import make_preprocessor
from loyalty_spend.models.registry import register

SEED = 42
# The ordinal preprocessor outputs pandas, so histogram boosting can find the
# categorical columns by name and split on category sets instead of codes.
CATEGORICAL_NAMES = ["region", "loyalty_years", "joining_month", "promotion"]


def _pipeline(model) -> Pipeline:
    prep = make_preprocessor("ordinal").set_output(transform="pandas")
    return Pipeline([("prep", prep), ("model", model)])


@register("random_forest", family="tree")
def random_forest(**params: Any) -> Pipeline:
    defaults = dict(n_estimators=500, min_samples_leaf=2, max_features=0.6, n_jobs=-1)
    return _pipeline(RandomForestRegressor(random_state=SEED, **{**defaults, **params}))


@register("hist_gbm", family="tree")
def hist_gbm(**params: Any) -> Pipeline:
    defaults = dict(
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        l2_regularization=1.0,
        categorical_features=CATEGORICAL_NAMES,
    )
    return _pipeline(
        HistGradientBoostingRegressor(random_state=SEED, **{**defaults, **params})
    )


@register("xgboost", family="tree", requires="xgboost")
def xgboost(**params: Any) -> Pipeline:
    from xgboost import XGBRegressor

    defaults = dict(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        n_jobs=1,
    )
    return _pipeline(XGBRegressor(random_state=SEED, **{**defaults, **params}))


@register("lightgbm", family="tree", requires="lightgbm")
def lightgbm(**params: Any) -> Pipeline:
    from lightgbm import LGBMRegressor

    defaults = dict(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=15,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        n_jobs=1,
        verbose=-1,
    )
    return _pipeline(LGBMRegressor(random_state=SEED, **{**defaults, **params}))
