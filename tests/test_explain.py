import numpy as np
import pytest

from loyalty_spend.explain.attribution import (
    grouped_permutation_importance,
    linear_contributions,
    mean_absolute_attribution,
    tree_shap,
)
from loyalty_spend.models.registry import build_model


def test_linear_contributions_add_up(xy):
    X, y = xy
    model = build_model("ols").fit(X, y)
    contrib = linear_contributions(model, X)
    assert set(contrib.columns) <= set(X.columns)
    centred = model.predict(X) - model.predict(X).mean()
    assert np.allclose(contrib.sum(axis=1), centred, atol=1e-6)


def test_tree_shap_adds_up(xy):
    pytest.importorskip("shap")
    X, y = xy
    model = build_model("lightgbm", n_estimators=100).fit(X, y)
    values = tree_shap(model, X)
    pred = model.predict(X)
    assert np.allclose(
        values.sum(axis=1) + (pred - values.sum(axis=1)).mean(), pred, atol=1e-3
    )


def test_permutation_importance_ranks_tenure_first(xy):
    X, y = xy
    model = build_model("ols").fit(X, y)
    table = grouped_permutation_importance(model, X, y, n_repeats=3)
    assert table["feature"].iloc[0] == "loyalty_years"
    assert (
        mean_absolute_attribution(linear_contributions(model, X)).index[0]
        == "loyalty_years"
    )
