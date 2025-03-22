import numpy as np
import pytest

from loyalty_spend.data.simulate import SCENARIOS, simulate
from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.models.registry import build_model
from loyalty_spend.uncertainty.conformal import (
    SplitConformalRegressor,
    conformal_quantile,
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
