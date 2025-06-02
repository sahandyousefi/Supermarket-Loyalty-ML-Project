import numpy as np
import pytest
from scipy import integrate, stats

from loyalty_spend.evaluation.scoring import (
    calibration_test,
    crps_gaussian,
    crps_sample,
    interval_score,
    interval_summary,
    mean_pinball,
    pinball_loss,
    pit_gaussian,
    pit_quantiles,
    pit_sample,
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


def test_interval_score_penalises_misses():
    score = interval_score([5.0, 0.0, 12.0], [2.0, 2.0, 2.0], [10.0, 10.0, 10.0], 0.2)
    np.testing.assert_allclose(score, [8.0, 8.0 + 10 * 2.0, 8.0 + 10 * 2.0])
    with pytest.raises(ValueError):
        interval_score([1.0], [2.0], [1.0], 0.1)


def test_interval_score_is_minimised_by_the_true_quantiles():
    rng = np.random.default_rng(3)
    y = rng.normal(size=40000)
    halfwidths = np.linspace(1.0, 2.4, 57)
    scores = [interval_score(y, -h, h, 0.1).mean() for h in halfwidths]
    best = halfwidths[int(np.argmin(scores))]
    assert best == pytest.approx(stats.norm.ppf(0.95), abs=0.05)


def test_interval_summary():
    out = interval_summary([1.0, 2.0, 3.0, 9.0], [0.0] * 4, [4.0] * 4, 0.2)
    assert out["coverage"] == pytest.approx(0.75)
    assert out["coverage_se"] == pytest.approx(np.sqrt(0.75 * 0.25 / 4))
    assert out["mean_width"] == pytest.approx(4.0)
    assert out["interval_score"] == pytest.approx(4.0 + 10 * 5.0 / 4)


def test_calibrated_forecasts_have_uniform_pit():
    rng = np.random.default_rng(4)
    mu = rng.normal(size=4000)
    y = mu + rng.normal(size=4000)
    out = calibration_test(pit_gaussian(y, mu, 1.0))
    assert out["chi2_p_value"] > 0.01
    assert out["tail_share"] == pytest.approx(0.10, abs=0.02)


def test_overconfident_forecasts_fail_the_test():
    rng = np.random.default_rng(5)
    mu = rng.normal(size=4000)
    y = mu + rng.normal(size=4000)
    out = calibration_test(pit_gaussian(y, mu, 0.5))
    assert out["chi2_p_value"] < 1e-6
    assert out["tail_share"] > 0.3


def test_sample_pit_breaks_ties_at_random():
    draws = np.repeat([[1.0, 2.0, 3.0, 4.0]], 4000, axis=0)
    u = pit_sample(np.full(4000, 2.0), draws, seed=1)
    assert u.min() >= 0.25 and u.max() <= 0.5
    assert np.mean(u) == pytest.approx(0.375, abs=0.01)


def test_quantile_pit_interpolates():
    taus = [0.1, 0.5, 0.9]
    q = [[0.0, 1.0, 2.0]] * 3
    np.testing.assert_allclose(pit_quantiles([0.5, 1.5, 5.0], q, taus), [0.3, 0.7, 0.9])
