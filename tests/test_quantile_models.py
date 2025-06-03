import numpy as np
import pytest

from loyalty_spend.data.schema import FEATURES
from loyalty_spend.data.simulate import SCENARIOS, simulate
from loyalty_spend.evaluation.scoring import mean_pinball
from loyalty_spend.models.quantile import (
    MultiQuantileRegressor,
    crossing_rate,
    gbm_quantile_model,
    rearrange,
)


@pytest.fixture(scope="module")
def hetero(train_df):
    sim = simulate(train_df, SCENARIOS["noisy"], n=1600, seed=3)
    df = sim.frame
    return (
        df[FEATURES].iloc[:1000],
        df["spend"].iloc[:1000],
        df[FEATURES].iloc[1000:],
        df["spend"].iloc[1000:],
    )


def test_linear_quantiles_cover_their_levels(hetero):
    X, y, X_new, y_new = hetero
    model = MultiQuantileRegressor(taus=(0.1, 0.5, 0.9)).fit(X, y)
    q = model.predict_quantiles(X_new)
    assert q.shape == (len(X_new), 3)
    shares = (y_new.to_numpy()[:, None] <= q).mean(axis=0)
    np.testing.assert_allclose(shares, [0.1, 0.5, 0.9], atol=0.05)
    np.testing.assert_allclose(model.predict(X_new), q[:, 1])


def test_quantile_models_beat_a_constant_spread(hetero):
    X, y, X_new, y_new = hetero
    taus = (0.1, 0.5, 0.9)
    q = MultiQuantileRegressor(taus=taus).fit(X, y).predict_quantiles(X_new)
    constant = np.tile(np.quantile(y, taus), (len(X_new), 1))
    assert (
        mean_pinball(y_new, q, taus).sum() < mean_pinball(y_new, constant, taus).sum()
    )


@pytest.mark.parametrize("taus", [(0.1, 0.9), (0.5, 0.2), (0.0, 0.5), ()])
def test_invalid_levels_are_rejected(hetero, taus):
    X, y, _, _ = hetero
    with pytest.raises(ValueError):
        MultiQuantileRegressor(taus=taus).fit(X.iloc[:50], y.iloc[:50])


def test_gradient_boosting_quantiles_adapt_to_heteroscedasticity(hetero):
    X, y, X_new, y_new = hetero
    taus = (0.1, 0.5, 0.9)
    gbm = MultiQuantileRegressor(gbm_quantile_model, taus).fit(X, y)
    q = gbm.predict_quantiles(X_new)
    shares = (y_new.to_numpy()[:, None] <= q).mean(axis=0)
    np.testing.assert_allclose(shares, taus, atol=0.07)
    # In the heteroscedastic scenario, high spenders get wider intervals.
    width = q[:, 2] - q[:, 0]
    high = q[:, 1] > np.median(q[:, 1])
    assert width[high].mean() > width[~high].mean()


def test_rearrangement_removes_crossings_and_reduces_loss():
    rng = np.random.default_rng(0)
    y = rng.normal(size=2000)
    taus = (0.1, 0.3, 0.5, 0.7, 0.9)
    true_q = np.tile(np.quantile(rng.normal(size=100000), taus), (2000, 1))
    noisy = true_q + rng.normal(scale=0.5, size=true_q.shape)
    assert crossing_rate(noisy) > 0.3
    fixed = rearrange(noisy)
    assert crossing_rate(fixed) == 0.0
    assert mean_pinball(y, fixed, taus).sum() <= mean_pinball(y, noisy, taus).sum()


def test_estimator_returns_monotone_quantiles(hetero):
    X, y, X_new, _ = hetero
    taus = (0.05, 0.1, 0.5, 0.9, 0.95)
    raw = MultiQuantileRegressor(gbm_quantile_model, taus, monotone=False).fit(X, y)
    fixed = MultiQuantileRegressor(gbm_quantile_model, taus).fit(X, y)
    assert crossing_rate(fixed.predict_quantiles(X_new)) == 0.0
    np.testing.assert_allclose(
        np.sort(raw.predict_quantiles(X_new), axis=1), fixed.predict_quantiles(X_new)
    )
