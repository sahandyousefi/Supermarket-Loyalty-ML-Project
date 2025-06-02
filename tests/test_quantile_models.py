import numpy as np
import pytest

from loyalty_spend.data.schema import FEATURES
from loyalty_spend.data.simulate import SCENARIOS, simulate
from loyalty_spend.evaluation.scoring import mean_pinball
from loyalty_spend.models.quantile import MultiQuantileRegressor


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
