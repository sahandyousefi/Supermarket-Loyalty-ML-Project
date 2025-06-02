import numpy as np
import pytest
from scipy import integrate, stats

from loyalty_spend.evaluation.scoring import (
    crps_gaussian,
    crps_sample,
    mean_pinball,
    pinball_loss,
)


def test_pinball_loss_values():
    np.testing.assert_allclose(pinball_loss([1.0, 1.0], [0.0, 2.0], 0.9), [0.9, 0.1])
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], 1.0)


def test_pinball_is_minimised_by_the_true_quantile():
    rng = np.random.default_rng(0)
    y = rng.normal(size=20000)
    grid = np.linspace(-2, 2, 81)
    losses = [pinball_loss(y, q, 0.8).mean() for q in grid]
    assert grid[int(np.argmin(losses))] == pytest.approx(stats.norm.ppf(0.8), abs=0.06)


def test_mean_pinball_checks_shape():
    with pytest.raises(ValueError):
        mean_pinball([1.0], [[1.0, 2.0]], [0.5])
    assert mean_pinball([1.0], [[1.0, 2.0]], [0.5, 0.9]).shape == (2,)


@pytest.mark.parametrize("y,mu,sigma", [(0.3, 0.0, 1.0), (5.0, 2.0, 0.5), (-1, 1, 3)])
def test_gaussian_crps_matches_the_integral_definition(y, mu, sigma):
    def integrand(z):
        return (stats.norm.cdf(z, mu, sigma) - (z >= y)) ** 2

    lo, hi = mu - 12 * sigma, mu + 12 * sigma
    value = integrate.quad(integrand, lo, y)[0] + integrate.quad(integrand, y, hi)[0]
    assert crps_gaussian(y, mu, sigma) == pytest.approx(value, rel=1e-6)


def test_sample_crps_converges_to_the_gaussian_value():
    rng = np.random.default_rng(1)
    y = np.array([0.0, 1.5, -2.0])
    draws = rng.normal(size=(3, 40000))
    np.testing.assert_allclose(
        crps_sample(y, draws), crps_gaussian(y, 0.0, 1.0), rtol=0.02
    )


def test_sample_crps_of_a_point_forecast_is_the_absolute_error():
    np.testing.assert_allclose(crps_sample([1.0, 4.0], [[3.0], [1.0]]), [2.0, 3.0])


def test_sample_crps_matches_the_pairwise_formula():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=25), 0.4
    pairwise = np.abs(x - y).mean() - 0.5 * np.abs(x[:, None] - x[None, :]).mean()
    assert crps_sample([y], x[None, :])[0] == pytest.approx(pairwise)
