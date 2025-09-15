import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from loyalty_spend.robustness.influence import influence_table


def _pipe():
    return Pipeline([("prep", FunctionTransformer()), ("model", LinearRegression())])


def test_planted_influential_point_is_flagged():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 2 * x + rng.normal(size=200)
    x[0], y[0] = 6.0, -10.0
    X = pd.DataFrame({"x": x})
    table = influence_table(_pipe().fit(X, y), X, y)
    assert table["cooks_distance"].idxmax() == 0
    assert table.loc[0, ["flag_hat", "flag_cooks", "flag_dfbetas"]].all()
    assert table["hat"].sum() == pytest.approx(2.0)


def test_dfbetas_match_leave_one_out_refits():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=60)})
    y = 1 + X["x"].to_numpy() + rng.normal(size=60)
    table = influence_table(_pipe().fit(X, y), X, y)
    full = LinearRegression().fit(X, y).coef_[0]
    loo = LinearRegression().fit(X.drop(index=5), np.delete(y, 5)).coef_[0]
    assert np.sign(full - loo) != 0
    assert table.loc[5, "max_abs_dfbetas"] > 0
