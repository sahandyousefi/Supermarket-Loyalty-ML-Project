import pytest

from loyalty_spend.business.programme import design_programme
from loyalty_spend.config import RewardConfig
from loyalty_spend.data.io import load_modeling_data


@pytest.fixture(scope="module")
def design(raw_dir):
    train, test = load_modeling_data(raw_dir / "train.csv", raw_dir / "test.csv")
    return design_programme(
        train, test, RewardConfig().tiers, offsets=(-5.0, 0.0, 5.0), n_simulations=300
    )


def test_design_summarises_cost_and_alternatives(design):
    schedules = design.schedules.set_index("schedule")
    assert (
        schedules.loc["marginal", "expected_cost"]
        < schedules.loc["whole spend", "expected_cost"]
    )
    assert design.tail_risk["expected_shortfall"] >= design.tail_risk["var"]
    assert design.budget == pytest.approx(design.tail_risk["var"])
    assert design.search.iloc[0]["feasible"]
    assert 0 < design.cost_to_spend < 0.05
    behaviour = design.behaviour.set_index("schedule")
    assert behaviour.loc["marginal", "cost_increase"] == 0.0
