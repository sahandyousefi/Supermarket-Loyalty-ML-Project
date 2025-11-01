import numpy as np
import pandas as pd
import pytest

from loyalty_spend.uncertainty.conformal import SplitConformalRegressor
from loyalty_spend.uncertainty.localized import LocalizedConformalRegressor


def _data(rng, n):
    band = rng.choice(["quiet", "noisy"], n, p=[0.7, 0.3])
    x = rng.uniform(0, 10, n)
    y = 2 * x + rng.normal(0, np.where(band == "noisy", 3.0, 0.5))
    return pd.DataFrame({"band": pd.Categorical(band), "x": x}), y


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    X, y = _data(rng, 2000)
    X_new, y_new = _data(rng, 1500)
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    model = make_pipeline(
        make_column_transformer((OneHotEncoder(), ["band"]), remainder="passthrough"),
        LinearRegression(),
    )
    local = LocalizedConformalRegressor(model, 0.1, bandwidth=0.05).fit(X, y)
    split = SplitConformalRegressor(model, 0.1).fit(X, y)
    return X_new, y_new, local, split


def _coverage(iv, y, mask):
    return float(((y >= iv.lower) & (y <= iv.upper))[mask].mean())


def test_localised_intervals_cover_each_band(fitted):
    X_new, y_new, local, split = fitted
    noisy = (X_new["band"] == "noisy").to_numpy()
    iv_l, iv_s = local.predict_interval(X_new), split.predict_interval(X_new)
    assert _coverage(iv_s, y_new, noisy) < 0.8
    assert _coverage(iv_l, y_new, noisy) == pytest.approx(0.9, abs=0.04)
    assert _coverage(iv_l, y_new, ~noisy) == pytest.approx(0.9, abs=0.04)
    assert _coverage(iv_l, y_new, np.ones(len(y_new), bool)) == pytest.approx(
        0.9, abs=0.03
    )


def test_large_bandwidth_recovers_split_conformal(fitted):
    X_new, _, local, split = fitted
    wide = LocalizedConformalRegressor(local.estimator, 0.1, bandwidth=1e6)
    wide.model_, wide.X_cal_, wide.scores_ = local.model_, local.X_cal_, local.scores_
    iv = wide.predict_interval(X_new.iloc[:50])
    assert np.allclose(
        iv.width, split.predict_interval(X_new.iloc[:50]).width, rtol=0.02
    )
    assert (local.effective_size(X_new.iloc[:20]) < len(local.scores_)).all()
