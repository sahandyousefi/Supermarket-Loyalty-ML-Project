import numpy as np
import pandas as pd
import pytest

from loyalty_spend.data.contracts import (
    ColumnContract,
    TableContract,
    check_contract,
    contract_from_dict,
    default_contract,
    load_contracts,
)
from loyalty_spend.data.schema import COLUMNS, FEATURES, ID


def _frame(**overrides):
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


def test_supplied_training_data_satisfy_the_contract(train_df):
    report = check_contract(train_df, default_contract())
    assert report.ok, report.to_frame()


def test_range_and_type_violations_are_counted():
    df = _frame(spend=[-1.0, 99.0, "abc"], items_in_first_month=[3, 4.5, -2])
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
    contract = TableContract(
        "t",
        (
            ColumnContract("x", "numeric"),
            ColumnContract("y", "numeric", required=False),
        ),
        min_rows=5,
    )
    report = check_contract(_frame(), contract)
    assert {(v.column, v.rule) for v in report.errors} == {
        ("*", "rows"),
        ("x", "presence"),
    }


def test_levels_uniqueness_and_missing_budget():
    df = _frame(
        customer_id=[1, 1, 2],
        region=["Europe", "europe", "Mars"],
        promotion=[np.nan, np.nan, "No"],
    )
    report = check_contract(df, default_contract())
    found = {(v.column, v.rule): v.n for v in report.errors}
    assert found == {
        ("customer_id", "unique"): 1,
        ("region", "levels"): 2,
        ("promotion", "missing"): 2,
    }


def test_unexpected_columns_are_a_warning_only():
    report = check_contract(_frame(note=["a", "b", "c"]), default_contract())
    assert report.ok
    assert [(v.rule, v.severity) for v in report.violations] == [
        ("unexpected", "warning")
    ]


def test_yaml_contracts_match_the_data_dictionary():
    contracts = load_contracts()
    training, scoring = contracts["training"], contracts["scoring"]
    reference = default_contract()
    assert training.names == reference.names
    assert scoring.names == default_contract(require_target=False).names
    for name in FEATURES:
        assert training.column(name).levels == reference.column(name).levels
        assert training.column(name).kind == reference.column(name).kind


def test_yaml_contracts_accept_the_supplied_files(train_df):
    report = check_contract(train_df, load_contracts()["training"])
    assert report.ok, report.to_frame()


def test_unknown_contract_keys_are_rejected():
    with pytest.raises(ValueError):
        contract_from_dict("t", {"columns": [], "rows": 3})
    with pytest.raises(TypeError):
        contract_from_dict(
            "t", {"columns": [{"name": "x", "kind": "numeric", "lo": 0}]}
        )
