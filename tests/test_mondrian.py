import numpy as np
import pandas as pd
import pytest

from loyalty_spend.data.schema import FEATURES
from loyalty_spend.data.simulate import SCENARIOS, simulate
from loyalty_spend.models.registry import build_model
from loyalty_spend.uncertainty.conformal import SplitConformalRegressor
from loyalty_spend.uncertainty.mondrian import MondrianConformalRegressor


@pytest.fixture(scope="module")
def noisy(train_df):
    df = simulate(train_df, SCENARIOS["noisy"], n=6000, seed=21).frame
    return (
        df[FEATURES].iloc[:3000],
        df["spend"].iloc[:3000],
        df[FEATURES].iloc[3000:],
        df["spend"].iloc[3000:],
    )


def _worst_bin_gap(iv, y, pred, target=0.9):
    bins = pd.qcut(pred, 5, labels=False)
    covered = (y >= iv.lower) & (y <= iv.upper)
    by_bin = pd.Series(covered).groupby(bins).mean()
    return float(np.max(np.abs(by_bin - target)))


def test_prediction_bins_improve_conditional_coverage(noisy):
    X, y, X_new, y_new = noisy
    y_new = y_new.to_numpy()
    split = SplitConformalRegressor(build_model("ols")).fit(X, y)
    mondrian = MondrianConformalRegressor(build_model("ols"), n_bins=5).fit(X, y)
    pred = split.predict(X_new)
    gap_split = _worst_bin_gap(split.predict_interval(X_new), y_new, pred)
    gap_mondrian = _worst_bin_gap(mondrian.predict_interval(X_new), y_new, pred)
    assert gap_mondrian < gap_split
    assert mondrian.predict_interval(X_new).coverage(y_new) == pytest.approx(
        0.9, abs=0.025
    )


def test_feature_groups_and_small_group_fallback(noisy):
    X, y, X_new, _ = noisy
    model = MondrianConformalRegressor(
        build_model("ols"), by="loyalty_years", min_group_size=200
    ).fit(X, y)
    small = [g for g, n in model.group_sizes_.items() if n < 200]
    assert set(model.q_) == set(model.group_sizes_) - set(small)
    iv = model.predict_interval(X_new)
    half = (iv.upper - iv.lower) / 2
    in_small = X_new["loyalty_years"].astype(str).isin(small).to_numpy()
    if in_small.any():
        np.testing.assert_allclose(half[in_small], model.pooled_q_)
