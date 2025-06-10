import numpy as np
import pytest

from loyalty_spend.evaluation.scoring import calibration_test
from loyalty_spend.models.registry import build_model
from loyalty_spend.uncertainty.predictive import (
    ConformalPredictiveDistribution,
    residual_quantiles,
)


def test_residual_quantiles_use_the_conformal_rank():
    r = np.arange(1.0, 10.0)  # n = 9, so n + 1 = 10
    np.testing.assert_allclose(residual_quantiles(r, [0.1, 0.5, 0.95]), [1.0, 5.0, 9.0])


def test_pit_of_new_outcomes_is_uniform(xy):
    X, y = xy
    cpd = ConformalPredictiveDistribution(build_model("ols")).fit(
        X.iloc[:300], y.iloc[:300]
    )
    u = cpd.cdf(X.iloc[300:], y.iloc[300:])
    assert u.min() >= 0 and u.max() <= 1
    assert calibration_test(u, n_bins=5)["chi2_p_value"] > 0.001


def test_quantiles_cdf_and_samples_agree(xy):
    X, y = xy
    cpd = ConformalPredictiveDistribution(build_model("ols")).fit(
        X.iloc[:300], y.iloc[:300]
    )
    X_new = X.iloc[300:320]
    q = cpd.quantiles(X_new, [0.25, 0.5, 0.75])
    assert np.all(np.diff(q, axis=1) >= 0)
    np.testing.assert_array_less(cpd.cdf(X_new, q[:, 1]) - 1e-12, 0.5 + 1 / 76 + 1e-9)
    draws = cpd.sample(X_new, 5000, seed=1)
    assert np.median(draws, axis=1) == pytest.approx(
        q[:, 1], abs=np.ptp(cpd.residuals_) / 10
    )
