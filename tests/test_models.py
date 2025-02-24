import numpy as np
import pytest

from loyalty_spend.evaluation.metrics import rmse
from loyalty_spend.models.registry import build_model

CLASSICAL = [
    "ols",
    "ridge",
    "elastic_net",
    "huber",
    "gamma_glm",
    "spline_ridge",
    "random_forest",
    "hist_gbm",
]


@pytest.mark.parametrize("name", CLASSICAL)
def test_classical_models_fit_and_beat_the_mean(xy, name):
    X, y = xy
    model = build_model(name).fit(X.iloc[:450], y.iloc[:450])
    pred = model.predict(X.iloc[450:])
    assert pred.shape == (150,)
    assert rmse(y.iloc[450:], pred) < y.std()


@pytest.mark.parametrize("name", ["xgboost", "lightgbm"])
def test_optional_boosters(xy, name):
    pytest.importorskip(name)
    X, y = xy
    pred = build_model(name, n_estimators=100).fit(X, y).predict(X)
    assert np.isfinite(pred).all()
