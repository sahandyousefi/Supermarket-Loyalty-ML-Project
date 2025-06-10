import numpy as np
import pytest

from loyalty_spend.data.schema import FEATURES
from loyalty_spend.data.simulate import SCENARIOS, simulate
from loyalty_spend.models.registry import build_model
from loyalty_spend.uncertainty.conformal import (
    SplitConformalRegressor,
    conformal_quantile,
)
from loyalty_spend.uncertainty.weighted import (
    WeightedConformalRegressor,
    weighted_conformal_quantile,
)


def test_equal_weights_reproduce_split_conformal():
    rng = np.random.default_rng(0)
    for n in (9, 50, 333):
        s = rng.exponential(size=n)
        for alpha in (0.05, 0.1, 0.2):
            q = weighted_conformal_quantile(s, np.ones(n), [1.0], alpha)[0]
            assert q == pytest.approx(conformal_quantile(s, alpha))


def test_large_test_weight_gives_an_infinite_interval():
    q = weighted_conformal_quantile([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [10.0], 0.1)
    assert np.isinf(q[0])


def test_weights_move_the_quantile_towards_heavy_scores():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    light = weighted_conformal_quantile(s, [1, 1, 1, 0.1], [0.1], 0.3)[0]
    heavy = weighted_conformal_quantile(s, [0.1, 1, 1, 1], [0.1], 0.3)[0]
    assert light < heavy


def test_invalid_weights():
    with pytest.raises(ValueError):
        weighted_conformal_quantile([1.0, 2.0], [1.0], [1.0], 0.1)
    with pytest.raises(ValueError):
        weighted_conformal_quantile([1.0], [-1.0], [1.0], 0.1)


@pytest.fixture(scope="module")
def shifted(train_df):
    # Noise grows with spend, and the new batch over-represents customers
    # with high expected spend, so the training residuals understate its
    # errors. Selection depends on the features only: a pure covariate shift.
    df = simulate(train_df, SCENARIOS["noisy"], n=8000, seed=41).frame
    train, pool = df.iloc[:3000], df.iloc[3000:].reset_index(drop=True)
    expected = build_model("ols").fit(train[FEATURES], train["spend"])
    pred = expected.predict(pool[FEATURES])
    rng = np.random.default_rng(0)
    high = pred > np.quantile(pred, 0.6)
    new = pool[rng.uniform(size=len(pool)) < np.where(high, 0.9, 0.1)]
    return train[FEATURES], train["spend"], new[FEATURES], new["spend"].to_numpy()


def test_weighting_restores_coverage_under_shift(shifted):
    X, y, X_new, y_new = shifted
    split = SplitConformalRegressor(build_model("ols")).fit(X, y)
    weighted = WeightedConformalRegressor(build_model("ols")).fit(X, y)
    iv = weighted.predict_interval(X_new)
    assert np.all(np.isfinite(iv.upper))
    assert split.predict_interval(X_new).coverage(y_new) < 0.89
    # Estimated weights are noisy, which errs on the side of wider intervals.
    assert 0.89 <= iv.coverage(y_new) <= 0.95
