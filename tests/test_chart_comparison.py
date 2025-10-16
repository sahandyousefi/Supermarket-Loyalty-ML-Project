import pytest

from loyalty_spend.monitoring.chart_comparison import Design, compare_charts


@pytest.fixture(scope="module")
def table():
    designs = (
        Design("Shewhart", "ewma", {"lam": 1.0}, (1.5, 4.0)),
        Design("EWMA", "ewma", {"lam": 0.1}, (1.0, 3.5)),
        Design("MHWMA", "mhwma", {"lam": 0.1}, (2.0, 40.0), multivariate=True),
    )
    return compare_charts(designs, deltas=(1.0,), target=100.0, n_reps=300, tau=20)


def test_all_charts_share_the_in_control_run_length(table):
    arl0 = table.groupby("chart")["arl0"].first()
    assert ((arl0 - 100.0).abs() < 8).all()


def test_memory_charts_beat_shewhart_on_small_shifts(table):
    zs = table[
        (table["state"] == "zero state") & table["scenario"].str.startswith("shift")
    ]
    arl = zs.set_index("chart")["arl"]
    assert arl["EWMA"] < arl["Shewhart"]
    assert arl["MHWMA"] < arl["Shewhart"]


def test_every_scenario_is_reported_in_both_states(table):
    assert set(table["state"]) == {"zero state", "steady state"}
    assert len(table) == 3 * 2 * 2
