import pytest

from loyalty_spend.evaluation.cv import make_splits
from loyalty_spend.synthetic.conditional import ConditionalCopulaSynthesizer
from loyalty_spend.synthetic.copula import GaussianCopulaSynthesizer
from loyalty_spend.synthetic.utility import (
    train_synthetic_test_real,
    utility_summary,
)


@pytest.fixture(scope="module")
def table(train_df):
    data = train_df.drop(columns="customer_id")
    splits = make_splits(len(data), 3, 1, 0)
    synths = {
        "copula": lambda s: GaussianCopulaSynthesizer(seed=s),
        "conditional": lambda s: ConditionalCopulaSynthesizer("loyalty_years", seed=s),
    }
    return train_synthetic_test_real(data, synths, splits)


def test_real_rows_have_unit_ratio(table):
    real = table[table["source"] == "real"]
    assert (real["ratio_to_real"] == 1.0).all()
    assert len(table) == 3 * 3


def test_conditional_synthesiser_keeps_most_predictive_information(table):
    summary = utility_summary(table).set_index("source")
    assert summary.loc["conditional", "rmse"] < summary.loc["copula", "rmse"] / 3
    # A plain copula loses the tenure effect, so its model is far worse.
    assert summary.loc["copula", "ratio_to_real"] > 10
