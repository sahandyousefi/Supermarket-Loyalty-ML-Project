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
