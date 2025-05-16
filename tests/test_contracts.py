import pytest

from loyalty_spend.data.contracts import ColumnContract, TableContract, default_contract
from loyalty_spend.data.schema import COLUMNS, FEATURES, ID


def test_default_contract_covers_the_schema():
    assert default_contract().names == COLUMNS
    assert default_contract(require_target=False).names == [ID, *FEATURES]
    assert default_contract().column(ID).unique


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "text"},
        {"kind": "numeric", "max_missing": 1.5},
        {"kind": "numeric", "minimum": 2.0, "maximum": 1.0},
        {"kind": "category"},
        {"kind": "category", "levels": ("a",), "minimum": 0.0},
        {"kind": "numeric", "levels": ("a",)},
    ],
)
def test_inconsistent_column_rules_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ColumnContract("x", **kwargs)


def test_duplicated_columns_are_rejected():
    col = ColumnContract("x", "numeric")
    with pytest.raises(ValueError):
        TableContract("t", (col, col))
    with pytest.raises(KeyError):
        TableContract("t", (col,)).column("y")


def _frame(**overrides):
    import pandas as pd

    base = {
        "customer_id": [1, 2, 3],
        "spend": [120.5, 99.0, 140.2],
        "first_month": [10.0, 12.5, 15.0],
        "items_in_first_month": [3, 4, 5],
        "region": ["Europe", "Americas", "Asia/Pacific"],
        "loyalty_years": ["0-1", "1-3", "10+"],
        "joining_month": ["Jan", "Unknown", "Dec"],
        "promotion": ["Yes", "No", "No"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_supplied_training_data_satisfy_the_contract(train_df):
    from loyalty_spend.data.contracts import check_contract

    report = check_contract(train_df, default_contract())
    assert report.ok, report.to_frame()


def test_range_and_type_violations_are_counted():
    from loyalty_spend.data.contracts import check_contract

    df = _frame(
        spend=[-1.0, 99.0, "abc"],
        items_in_first_month=[3, 4.5, -2],
    )
    report = check_contract(df, default_contract())
    found = {(v.column, v.rule): v.n for v in report.errors}
    assert found == {
        ("spend", "type"): 1,
        ("spend", "range"): 1,
        ("items_in_first_month", "integer"): 1,
        ("items_in_first_month", "range"): 1,
    }
    with pytest.raises(ValueError, match="contract violated"):
        report.raise_for_errors()


def test_missing_required_column_and_row_minimum():
    from loyalty_spend.data.contracts import check_contract

    contract = TableContract(
        "t",
        (
            ColumnContract("x", "numeric"),
            ColumnContract("y", "numeric", required=False),
        ),
        min_rows=5,
    )
    report = check_contract(_frame().rename(columns={"spend": "z"}), contract)
    assert {(v.column, v.rule) for v in report.errors} == {
        ("*", "rows"),
        ("x", "presence"),
    }
