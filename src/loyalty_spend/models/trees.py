"""Tree ensembles."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor
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
