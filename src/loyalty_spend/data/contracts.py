"""Declarative data contracts.

The schema module fixes the column names of the brief. A contract states
what a valid extract looks like column by column: its kind, the allowed
range or levels, whether values must be unique and how many may be missing.
Contracts are data, so a monitoring job can check a new extract against the
same rules that the training data were checked against.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import pandas as pd

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


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Violation:
    column: str
    rule: str
    n: int
    detail: str
    severity: Severity = "error"


@dataclass
class ContractReport:
    """Outcome of checking one frame against a contract."""

    contract: str
    n_rows: int
    violations: list[Violation] = field(default_factory=list)

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_frame(self) -> pd.DataFrame:
        cols = ["column", "rule", "n", "detail", "severity"]
        return pd.DataFrame([asdict(v) for v in self.violations], columns=cols)

    def raise_for_errors(self) -> None:
        if self.errors:
            lines = [f"{v.column}: {v.rule}: {v.detail}" for v in self.errors]
            raise ValueError(
                f"{self.contract}: contract violated:\n- " + "\n- ".join(lines)
            )


def _numeric_rules(series: pd.Series, rule: ColumnContract) -> list[Violation]:
    out: list[Violation] = []
    values = pd.to_numeric(series, errors="coerce")
    n_type = int((values.isna() & series.notna()).sum())
    if n_type:
        out.append(Violation(rule.name, "type", n_type, "values are not numeric"))
    if rule.kind == "integer":
        finite = values.dropna()
        n_frac = int((finite != finite.round()).sum())
        if n_frac:
            out.append(
                Violation(rule.name, "integer", n_frac, "values are not whole numbers")
            )
    if rule.minimum is not None:
        n_low = int((values < rule.minimum).sum())
        if n_low:
            out.append(
                Violation(rule.name, "range", n_low, f"values below {rule.minimum}")
            )
    if rule.maximum is not None:
        n_high = int((values > rule.maximum).sum())
        if n_high:
            out.append(
                Violation(rule.name, "range", n_high, f"values above {rule.maximum}")
            )
    return out


def check_contract(df: pd.DataFrame, contract: TableContract) -> ContractReport:
    """Check ``df`` against ``contract`` and collect every violation.

    All rules are evaluated, so one run lists every problem of an extract
    instead of stopping at the first.
    """
    report = ContractReport(contract.name, len(df))
    if len(df) < contract.min_rows:
        report.violations.append(
            Violation("*", "rows", len(df), f"fewer than {contract.min_rows} rows")
        )
    for rule in contract.columns:
        if rule.name not in df.columns:
            if rule.required:
                report.violations.append(
                    Violation(rule.name, "presence", 1, "required column is missing")
                )
            continue
        if rule.kind in ("numeric", "integer"):
            report.violations.extend(_numeric_rules(df[rule.name], rule))
    return report
