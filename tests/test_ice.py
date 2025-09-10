import numpy as np
import pandas as pd
import pytest

from loyalty_spend.explain.ice import ice_curves, interaction_strength, pdp_summary


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.uniform(0, 1, 500), "b": rng.choice([0.0, 1.0], 500)})


def test_additive_model_has_parallel_curves(frame):
    grid, curves = ice_curves(
        lambda Z: 2 * Z["a"].to_numpy() + Z["b"].to_numpy(), frame, "a"
    )
    assert interaction_strength(curves) == pytest.approx(0.0, abs=1e-12)
    summary = pdp_summary(grid, curves)
    slope = np.polyfit(summary["value"], summary["partial_dependence"], 1)[0]
    assert slope == pytest.approx(2.0)


def test_interaction_spreads_the_centred_curves(frame):
    grid, curves = ice_curves(
        lambda Z: Z["a"].to_numpy() * (2 * Z["b"].to_numpy() - 1) * 3, frame, "a"
    )
    assert interaction_strength(curves) > 1.0
    assert curves.shape == (200, len(grid))


def test_explanation_figures(frame, tmp_path):
    from loyalty_spend.explain.ale import ale_1d
    from loyalty_spend.reporting.figures import plot_ale, plot_ice

    def f(Z):
        return Z["a"].to_numpy() ** 2

    curve = ale_1d(f, frame, "a")
    assert plot_ale([curve, curve], tmp_path / "ale.png").stat().st_size > 1000
    grid, curves = ice_curves(f, frame, "a")
    assert plot_ice(grid, curves, "a", tmp_path / "ice.png").stat().st_size > 1000
