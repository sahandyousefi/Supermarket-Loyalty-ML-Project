import numpy as np
import pytest
from scipy import stats

from loyalty_spend.evaluation.scoring import crps_gaussian, crps_quantiles
from loyalty_spend.models.forecasters import FORECASTERS, TAU_GRID, build_forecaster


def test_quantile_crps_approximates_the_exact_value():
    rng = np.random.default_rng(0)
    y = rng.normal(size=5000)
    fine = np.linspace(0.005, 0.995, 199)
    q = np.tile(stats.norm.ppf(fine), (5000, 1))
    approx = crps_quantiles(y, q, fine).mean()
    assert approx == pytest.approx(crps_gaussian(y, 0.0, 1.0).mean(), rel=0.02)


def test_grid_is_symmetric_and_contains_the_median():
    assert len(TAU_GRID) == 19
    assert TAU_GRID[0] == 0.05 and TAU_GRID[-1] == 0.95 and 0.5 in TAU_GRID


@pytest.mark.parametrize("name", sorted(FORECASTERS))
def test_every_forecaster_returns_sorted_grid_quantiles(name, xy):
    X, y = xy
    q = build_forecaster(name).fit(X.iloc[:400], y.iloc[:400]).quantiles(X.iloc[400:])
    assert q.shape == (len(X) - 400, len(TAU_GRID))
    assert np.all(np.diff(q, axis=1) >= -1e-9)


def test_unknown_forecaster():
    with pytest.raises(KeyError):
        build_forecaster("nope")
