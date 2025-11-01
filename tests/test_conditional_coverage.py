import pytest

from loyalty_spend.features.preprocess import split_xy
from loyalty_spend.uncertainty.compare import conditional_coverage_comparison


@pytest.fixture(scope="module")
def result(train_df):
    X, y = split_xy(train_df)
    return conditional_coverage_comparison(X, y, n_splits=3)


def test_every_method_is_audited_on_every_member(result, train_df):
    summary, audit = result
    assert len(summary) == 7
    counts = audit.groupby(["method", "attribute"])["n"].sum()
    assert (counts == len(train_df)).all()


def test_mondrian_by_month_removes_month_level_flags(result):
    summary, audit = result
    by_month = audit[audit["attribute"] == "joining_month"]
    flagged = by_month.groupby("method")["flagged"].sum()
    assert flagged["Mondrian (joining_month)"] < flagged["split conformal"]
    assert summary["coverage"].between(0.85, 0.95).all()
