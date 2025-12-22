import numpy as np
import pandas as pd
import pytest

from loyalty_spend.models.gaussian_process import MixedGaussianProcess, mixed_kernel
from loyalty_spend.models.registry import available_models, build_model


def test_kernel_has_one_length_scale_per_column():
    k = mixed_kernel(4)
    assert k.k1.k2.k2.length_scale.shape == (4,)


def test_gp_is_registered():
    assert "gp" in available_models()


def test_gp_captures_an_interaction_that_a_linear_model_misses():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(-2, 2, n)
    g = rng.choice([0.0, 1.0], n)
    y = x + 2 * np.sin(2 * x) * g + rng.normal(0, 0.1, n)
    X = np.column_stack([x, g])
    gp = MixedGaussianProcess(n_restarts_optimizer=0).fit(X[:200], y[:200])
    lin = np.polyfit(x[:200], y[:200], 1)
    gp_rmse = np.sqrt(np.mean((gp.predict(X[200:]) - y[200:]) ** 2))
    lin_rmse = np.sqrt(np.mean((np.polyval(lin, x[200:]) - y[200:]) ** 2))
    assert gp_rmse < lin_rmse / 4
    noise = gp.kernel_.k2.noise_level * gp._y_train_std**2
    assert np.sqrt(noise) == pytest.approx(0.1, rel=0.3)


def test_pipeline_fits_the_member_data(train_df):
    from loyalty_spend.features.preprocess import split_xy

    X, y = split_xy(train_df.iloc[:300])
    model = build_model("gp", n_restarts_optimizer=0).fit(X, y)
    assert np.isfinite(model.predict(X)).all()


def test_restarts_reach_the_same_optimum_on_a_simple_problem():
    from loyalty_spend.models.gaussian_process import restart_table

    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, (120, 2))
    y = np.sin(2 * X[:, 0]) + rng.normal(0, 0.1, 120)
    table = restart_table(X, y, n_starts=4)
    assert len(table) == 4
    assert (table["parameters_at_bounds"] >= 0).all()
    best = table["log_marginal_likelihood"].iloc[0]
    assert (table["log_marginal_likelihood"] <= best + 1e-6).all()
    assert table["noise_sd"].iloc[0] == pytest.approx(0.1, rel=0.3)


def test_predictive_sd_includes_member_noise(train_df):
    from loyalty_spend.features.preprocess import split_xy
    from loyalty_spend.models.gaussian_process import predictive

    X, y = split_xy(train_df.iloc[:300])
    model = build_model("gp", n_restarts_optimizer=0).fit(X, y)
    _, sd = predictive(model, X.iloc[:20])
    resid_sd = np.std(y - model.predict(X))
    assert (sd > 0.5 * resid_sd).all()


def test_cross_fitted_intervals_are_calibrated():
    from loyalty_spend.models.gaussian_process import cross_fitted_calibration

    rng = np.random.default_rng(2)
    n = 400
    X = pd.DataFrame(
        {
            "first_month": rng.uniform(10, 50, n),
            "items_in_first_month": rng.integers(5, 40, n).astype(float),
            "region": pd.Categorical(rng.choice(["Americas", "Europe"], n)),
            "loyalty_years": pd.Categorical(rng.choice(["0-1", "1-3"], n)),
            "joining_month": pd.Categorical(rng.choice(["Jan", "Feb"], n)),
            "promotion": pd.Categorical(rng.choice(["Yes", "No"], n)),
        }
    )
    y = 100 + 0.5 * X["first_month"].to_numpy() + rng.normal(0, 2, n)
    out = cross_fitted_calibration(X, y, n_splits=3)
    assert out["coverage"] == pytest.approx(0.9, abs=0.04)
