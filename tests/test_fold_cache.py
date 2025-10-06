import numpy as np
import pytest

from loyalty_spend.evaluation.cv import cross_validate, make_splits
from loyalty_spend.evaluation.fold_cache import FoldCache, split_pipeline
from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.models.registry import build_model


@pytest.fixture(scope="module")
def data(train_df):
    X, y = split_xy(train_df)
    return X, y, make_splits(len(y), 3, 2, 0)


def test_cached_results_equal_uncached(data):
    X, y, splits = data
    cache = FoldCache(X, y, splits)
    for name in ("ols", "ridge", "hist_gbm"):
        plain = cross_validate(name, build_model(name), X, y, splits)
        cached = cross_validate(name, build_model(name), X, y, splits, cache=cache)
        assert np.allclose(plain.oof, cached.oof, rtol=0, atol=1e-10)


def test_models_with_the_same_encoding_share_preprocessing(data):
    X, y, splits = data
    cache = FoldCache(X, y, splits)
    cross_validate("ols", build_model("ols"), X, y, splits, cache=cache)
    cross_validate("ridge", build_model("ridge"), X, y, splits, cache=cache)
    assert cache.misses == len(splits) and cache.hits == len(splits)
    cross_validate("hist_gbm", build_model("hist_gbm"), X, y, splits, cache=cache)
    assert cache.misses == 2 * len(splits)


def test_cache_rejects_other_splits(data):
    X, y, splits = data
    cache = FoldCache(X, y, splits)
    with pytest.raises(ValueError):
        cross_validate("ols", build_model("ols"), X, y, list(splits), cache=cache)


def test_non_pipeline_estimators_are_fitted_whole(data):
    from sklearn.dummy import DummyRegressor

    X, y, splits = data
    assert split_pipeline(DummyRegressor()) is None
    model, pred = FoldCache(X, y, splits).fit_predict(DummyRegressor(), 0)
    assert np.allclose(pred, y.iloc[splits[0][0]].mean())
