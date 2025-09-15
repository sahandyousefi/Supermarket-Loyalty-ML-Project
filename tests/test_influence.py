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
    A = np.column_stack([np.ones(60), X["x"]])
    inv = np.linalg.inv(A.T @ A)
    full = inv @ A.T @ y
    A5, y5 = np.delete(A, 5, axis=0), np.delete(y, 5)
    loo = np.linalg.lstsq(A5, y5, rcond=None)[0]
    s_loo = np.sqrt(np.sum((y5 - A5 @ loo) ** 2) / (59 - 2))
    expected = abs(full[1] - loo[1]) / (s_loo * np.sqrt(inv[1, 1]))
    assert table.loc[5, "max_abs_dfbetas"] == pytest.approx(expected, rel=1e-8)
