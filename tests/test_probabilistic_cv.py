import numpy as np
import pytest

from loyalty_spend.data.schema import FEATURES
from loyalty_spend.data.simulate import SCENARIOS, simulate
from loyalty_spend.evaluation.cv import make_splits
from loyalty_spend.evaluation.probabilistic import (
    compare_distributions,
    cross_validate_distribution,
)
from loyalty_spend.models.forecasters import FORECASTERS


@pytest.fixture(scope="module")
def comparison(train_df):
    df = simulate(train_df, SCENARIOS["noisy"], n=1500, seed=5).frame
    X, y = df[FEATURES], df["spend"]
    splits = make_splits(len(y), n_splits=3, n_repeats=1, seed=0)
    names = ["gaussian", "gaussian_heteroscedastic", "linear_quantile"]
    results = [
        cross_validate_distribution(n, FORECASTERS[n], X, y, splits) for n in names
    ]
    return compare_distributions(results, "gaussian").set_index("model")


def test_heteroscedastic_models_beat_constant_variance(comparison):
    assert comparison.loc["gaussian_heteroscedastic", "crps_diff"] < 0
    assert (
        comparison.loc["linear_quantile", "interval_score_mean"]
        < (comparison.loc["gaussian", "interval_score_mean"])
    )


def test_coverage_is_close_to_nominal(comparison):
    assert comparison["coverage_mean"].between(0.85, 0.95).all()
    # The reference is not tested against itself.
    assert np.isnan(comparison.loc["gaussian", "p_value"])


def test_unknown_reference(comparison):
    with pytest.raises(KeyError):
        compare_distributions([], "gaussian")
