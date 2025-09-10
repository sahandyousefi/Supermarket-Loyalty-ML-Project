import numpy as np
import pandas as pd
import pytest

from loyalty_spend.explain.ale import ale_1d
from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.models.registry import build_model


def test_ale_of_a_known_function_with_correlated_features():
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0, 1, 3000)
    x2 = x1 + rng.normal(scale=0.05, size=3000)
    X = pd.DataFrame({"x1": x1, "x2": x2})

    def f(Z):
        return Z["x1"].to_numpy() ** 2 + 5 * Z["x2"].to_numpy()

    curve = ale_1d(f, X, "x1", n_bins=25)
    grid = curve["value"].to_numpy()
    expected = grid**2 - np.mean(x1**2)
    np.testing.assert_allclose(curve["effect"], expected, atol=0.02)


def test_ale_slope_matches_the_linear_effect(train_df):
    X, y = split_xy(train_df)
    model = build_model("ols").fit(X, y)
    curve = ale_1d(model.predict, X, "first_month", n_bins=10)
    slope = np.polyfit(curve["value"], curve["effect"], 1)[0]
    # For a linear model the effect of one more unit is the same everywhere.
    unit = np.mean(
        model.predict(X.assign(first_month=X["first_month"] + 1)) - model.predict(X)
    )
    assert slope == pytest.approx(unit, rel=1e-6)
    assert curve["n_in_bin"].sum() == X["first_month"].notna().sum()
    with pytest.raises(ValueError):
        ale_1d(model.predict, X.assign(first_month=1.0), "first_month")
