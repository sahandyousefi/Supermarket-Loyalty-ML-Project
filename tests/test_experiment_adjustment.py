import numpy as np
import pandas as pd
import pytest

from loyalty_spend.experiment.adjustment import (
    cuped,
    difference_in_means,
    lin_adjusted,
)


def _experiment(rng, n=400, effect=1.0):
    x = rng.normal(20, 4, n)
    band = rng.choice(["a", "b", "c"], n)
    t = rng.permutation(np.arange(n) < n // 2)
    y = 50 + 2.0 * x + np.select([band == "b", band == "c"], [10, -5], 0)
    y = y + effect * t + rng.normal(0, 3, n)
    return y, x, band, t


def test_estimators_are_unbiased_and_adjustment_is_more_precise():
    rng = np.random.default_rng(0)
    est = {"difference": [], "cuped": [], "lin": []}
    for _ in range(400):
        y, x, band, t = _experiment(rng)
        est["difference"].append(difference_in_means(y, t).estimate)
        est["cuped"].append(cuped(y, x, t).estimate)
        frame = pd.DataFrame({"x": x, "band": band})
        est["lin"].append(lin_adjusted(y, frame, t).estimate)
    sd = {k: np.std(v) for k, v in est.items()}
    for v in est.values():
        assert np.mean(v) == pytest.approx(1.0, abs=0.1)
    assert sd["lin"] < sd["cuped"] < sd["difference"]


def test_standard_errors_match_the_spread():
    rng = np.random.default_rng(1)
    estimates, ses = [], []
    for _ in range(400):
        y, x, band, t = _experiment(rng)
        e = lin_adjusted(y, pd.DataFrame({"x": x, "band": band}), t)
        estimates.append(e.estimate)
        ses.append(e.se)
    assert np.mean(ses) == pytest.approx(np.std(estimates), rel=0.12)


def test_cuped_reduction_is_one_minus_rho_squared():
    rng = np.random.default_rng(2)
    n = 20000
    x = rng.normal(size=n)
    y = 0.6 * x + 0.8 * rng.normal(size=n)
    t = rng.permutation(np.arange(n) < n // 2)
    ratio = (cuped(y, x, t).se / difference_in_means(y, t).se) ** 2
    assert ratio == pytest.approx(1 - 0.36, abs=0.03)
