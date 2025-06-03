import numpy as np
import pytest

from loyalty_spend.data.schema import FEATURES
from loyalty_spend.data.simulate import SCENARIOS, simulate
from loyalty_spend.evaluation.scoring import (
    calibration_test,
    crps_gaussian,
    pit_gaussian,
)
from loyalty_spend.models.distributional import HeteroscedasticGaussianRegressor


@pytest.fixture(scope="module")
def noisy(train_df):
    sim = simulate(train_df, SCENARIOS["noisy"], n=3000, seed=11)
    df = sim.frame
    return df[FEATURES], df["spend"].to_numpy(), sim.noise_sd


def test_sigma_tracks_the_true_noise_level(noisy):
    X, y, sd = noisy
    model = HeteroscedasticGaussianRegressor().fit(X.iloc[:2000], y[:2000])
    sigma = model.predict_sigma(X.iloc[2000:])
    assert np.corrcoef(sigma, sd[2000:])[0, 1] > 0.8
    assert np.median(sigma / sd[2000:]) == pytest.approx(1.0, rel=0.25)


def test_heteroscedastic_model_beats_a_constant_variance(noisy):
    X, y, _ = noisy
    model = HeteroscedasticGaussianRegressor().fit(X.iloc[:2000], y[:2000])
    mu = model.predict(X.iloc[2000:])
    resid_sd = np.std(y[:2000] - model.predict(X.iloc[:2000]))
    crps_het = crps_gaussian(y[2000:], mu, model.predict_sigma(X.iloc[2000:])).mean()
    crps_const = crps_gaussian(y[2000:], mu, resid_sd).mean()
    assert crps_het < crps_const


def test_quantiles_and_samples_are_consistent(noisy):
    X, y, _ = noisy
    model = HeteroscedasticGaussianRegressor().fit(X.iloc[:2000], y[:2000])
    X_new = X.iloc[2000:2100]
    q = model.predict_quantiles(X_new, [0.1, 0.5, 0.9])
    np.testing.assert_allclose(q[:, 1], model.predict(X_new))
    draws = model.sample(X_new, 4000, seed=0)
    np.testing.assert_allclose(np.quantile(draws, 0.9, axis=1), q[:, 2], rtol=0.02)
    pit = pit_gaussian(
        y[2000:], model.predict(X.iloc[2000:]), model.predict_sigma(X.iloc[2000:])
    )
    assert calibration_test(pit)["tail_share"] == pytest.approx(0.1, abs=0.04)
