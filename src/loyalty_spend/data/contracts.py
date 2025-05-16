"""Declarative data contracts.

The schema module fixes the column names of the brief. A contract states
what a valid extract looks like column by column: its kind, the allowed
range or levels, whether values must be unique and how many may be missing.
Contracts are data, so a monitoring job can check a new extract against the
same rules that the training data were checked against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from loyalty_spend.data.schema import (
    CATEGORICAL,
    ID,
    TARGET,
    allowed_values,
)

Kind = Literal["id", "numeric", "integer", "category"]
KINDS: tuple[str, ...] = ("id", "numeric", "integer", "category")


@dataclass(frozen=True)
class ColumnContract:
    """Rules for one column.

    Parameters
    ----------
    name : str
    kind : {"id", "numeric", "integer", "category"}
    required : bool
        The column must be present.
    minimum, maximum : float, optional
        Inclusive bounds for numeric and integer columns.
    levels : tuple of str, optional
        Allowed values of a category column.
    unique : bool
        Values must not repeat.
    max_missing : float
        Largest acceptable share of missing values, in [0, 1].
    """

    name: str
    kind: Kind
    required: bool = True
    minimum: float | None = None
    maximum: float | None = None
    levels: tuple[str, ...] | None = None
    unique: bool = False
    max_missing: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind must be one of {KINDS}")
        if not 0.0 <= self.max_missing <= 1.0:
            raise ValueError(f"{self.name}: max_missing must be in [0, 1]")
        has_bounds = self.minimum is not None or self.maximum is not None
        if has_bounds and self.kind not in ("numeric", "integer"):
            raise ValueError(f"{self.name}: bounds apply to numeric columns only")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(f"{self.name}: minimum exceeds maximum")
        if self.kind == "category" and not self.levels:
            raise ValueError(f"{self.name}: a category column needs levels")
        if self.levels is not None and self.kind != "category":
            raise ValueError(f"{self.name}: levels apply to category columns only")


@dataclass(frozen=True)
class TableContract:
    """Rules for a whole extract."""

    name: str
    columns: tuple[ColumnContract, ...]
    min_rows: int = 1

    def __post_init__(self) -> None:
        names = [c.name for c in self.columns]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name}: duplicated column contracts")
        if self.min_rows < 0:
            raise ValueError(f"{self.name}: min_rows must be non-negative")

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> ColumnContract:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"{self.name}: no contract for column '{name}'")


def default_contract(require_target: bool = True) -> TableContract:
    """Contract implied by the data dictionary of the brief.

    Amounts are non-negative and items are whole numbers; categories must use
    the canonical levels after cleaning. Missing feature values are allowed
    up to a fifth of the rows, which is well above the 10% observed in
    ``loyalty.csv`` but low enough to catch a broken extract.
    """
    columns = [
        ColumnContract(ID, "id", unique=True, max_missing=0.0),
        ColumnContract("first_month", "numeric", minimum=0.0, max_missing=0.2),
        ColumnContract("items_in_first_month", "integer", minimum=0.0, max_missing=0.2),
        *(
            ColumnContract(
                c, "category", levels=tuple(allowed_values(c)), max_missing=0.2
            )
            for c in CATEGORICAL
        ),
    ]
    if require_target:
        columns.insert(1, ColumnContract(TARGET, "numeric", minimum=0.0))
    return TableContract("loyalty", tuple(columns))
