import numpy as np
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
