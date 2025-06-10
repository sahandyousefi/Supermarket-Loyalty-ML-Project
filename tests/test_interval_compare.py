import pytest

from loyalty_spend.data.schema import FEATURES
from loyalty_spend.data.simulate import SCENARIOS, simulate
from loyalty_spend.uncertainty.compare import compare_intervals, interval_methods


@pytest.fixture(scope="module")
def table(train_df):
    df = simulate(train_df, SCENARIOS["noisy"], n=4000, seed=51).frame
    fit, new = df.iloc[:2000], df.iloc[2000:]
    return compare_intervals(
        fit[FEATURES], fit["spend"], new[FEATURES], new["spend"]
    ).set_index("method")


def test_every_method_is_evaluated(table):
    assert list(table.index) == list(interval_methods())
    assert table["coverage"].between(0.86, 0.94).all()
    assert (table["worst_group_coverage"] <= table["coverage"]).all()
    assert (table["best_group_coverage"] >= table["coverage"]).all()


def test_adaptive_methods_balance_coverage_across_groups(table):
    spread = table["best_group_coverage"] - table["worst_group_coverage"]
    assert spread["normalized conformal"] < spread["split conformal"]
    assert spread["Mondrian (prediction bins)"] < spread["split conformal"]
