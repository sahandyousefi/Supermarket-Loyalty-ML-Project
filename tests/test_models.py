import numpy as np
import pytest

from loyalty_spend.evaluation.metrics import rmse
from loyalty_spend.models import registry
from loyalty_spend.models.registry import available_models, build_model, register

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


def test_registry_lists_and_rejects():
    assert set(CLASSICAL) <= set(available_models())
    with pytest.raises(KeyError):
        build_model("svm")


def test_duplicate_registration_is_refused():
    with pytest.raises(ValueError):
        register("ols", family="linear")(lambda: None)


def test_missing_optional_dependency(monkeypatch):
    registry._load_all()
    spec = registry._REGISTRY["ols"]
    monkeypatch.setitem(
        registry._REGISTRY,
        "ols",
        registry.ModelSpec("ols", spec.factory, "not_a_pkg", "linear"),
    )
    assert "ols" not in available_models()
    with pytest.raises(ImportError):
        build_model("ols")
