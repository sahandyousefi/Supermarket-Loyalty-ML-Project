import numpy as np
import pandas as pd
import pytest

from loyalty_spend.data.simulate import SCENARIOS, simulate
from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.models.registry import build_model
from loyalty_spend.uncertainty.conformal import (
    ConformalizedQuantileRegressor,
    Intervals,
    SplitConformalRegressor,
    conformal_quantile,
    coverage_by_group,
    cv_plus_intervals,
)


@pytest.fixture(scope="module")
def data(train_df):
    fit = simulate(train_df, SCENARIOS["noisy"], n=1500, seed=0).frame
    test = simulate(train_df, SCENARIOS["noisy"], n=3000, seed=1).frame
    return split_xy(fit), split_xy(test)


def test_conformal_quantile_rank():
    scores = np.arange(1, 20, dtype=float)
    assert conformal_quantile(scores, 0.1) == 18.0
    assert conformal_quantile(scores[:5], 0.05) == float("inf")


def test_split_conformal_coverage(data):
    (X, y), (Xt, yt) = data
    model = SplitConformalRegressor(build_model("ols"), alpha=0.1).fit(X, y)
    iv = model.predict_interval(Xt)
    assert 0.87 <= iv.coverage(yt) <= 0.94
    assert np.allclose(iv.width, iv.width[0])


def test_cqr_coverage_and_adaptivity(data):
    (X, y), (Xt, yt) = data
    iv = ConformalizedQuantileRegressor(alpha=0.1).fit(X, y).predict_interval(Xt)
    assert 0.86 <= iv.coverage(yt) <= 0.95
    assert iv.width.std() > 0


def test_cv_plus(data):
    (X, y), (Xt, yt) = data
    iv = cv_plus_intervals(
        build_model("ols"), X.iloc[:600], y.iloc[:600], Xt.iloc[:500], 0.1, 5
    )
    assert 0.83 <= iv.coverage(yt.iloc[:500]) <= 0.97


def test_coverage_by_group():
    iv = Intervals(np.zeros(4), np.ones(4))
    out = coverage_by_group(iv, [0.5, 2.0, 0.2, 0.9], pd.Series(["a", "a", "b", "b"]))
    assert out.set_index("group")["coverage"].to_dict() == {"a": 0.5, "b": 1.0}
