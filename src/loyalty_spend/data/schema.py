"""Column specification of the loyalty data and validation against it."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

ID = "customer_id"
TARGET = "spend"
NUMERIC = ["first_month", "items_in_first_month"]
REGIONS = ["Americas", "Asia/Pacific", "Europe", "Middle East/Africa"]
LOYALTY_ORDER = ["0-1", "1-3", "3-5", "5-10", "10+"]
MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
PROMOTION = ["Yes", "No"]
CATEGORIES: dict[str, list[str]] = {
    "region": REGIONS,
    "loyalty_years": LOYALTY_ORDER,
    "joining_month": MONTHS,
    "promotion": PROMOTION,
}
CATEGORICAL = list(CATEGORIES)
FEATURES = NUMERIC + CATEGORICAL
COLUMNS = [ID, TARGET, *FEATURES]

# Replacement values for missing entries, as stated in the data dictionary.
FILL_VALUES: dict[str, object] = {
    "spend": 0.0,
    "first_month": 0.0,
    "items_in_first_month": 0,
    "region": "Unknown",
    "loyalty_years": "0-1",
    "joining_month": "Unknown",
    "promotion": "No",
}


def allowed_values(column: str) -> list[str]:
    """Valid levels of a categorical column after cleaning."""
    levels = list(CATEGORIES[column])
    if FILL_VALUES.get(column) == "Unknown":
        levels.append("Unknown")
    return levels


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("validation failed:\n- " + "\n- ".join(self.errors))


def validate(df: pd.DataFrame, require_target: bool = True) -> ValidationReport:
    """Check a cleaned frame against the data dictionary.

    Missing columns, missing or duplicated ids, negative amounts and unknown
    category levels are errors. Missing feature values are warnings, since the
    modeling pipeline can impute them.
    """
    report = ValidationReport()
    expected = [c for c in COLUMNS if require_target or c != TARGET]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        report.errors.append(f"missing columns: {missing}")
        return report

    if df[ID].isna().any():
        report.errors.append(f"{ID}: missing values")
    n_dup = int(df[ID].duplicated().sum())
    if n_dup:
        report.errors.append(f"{ID}: {n_dup} duplicated ids")

    for col in NUMERIC + ([TARGET] if require_target else []):
        values = pd.to_numeric(df[col], errors="coerce")
        n_bad = int((values < 0).sum())
        if n_bad:
            report.errors.append(f"{col}: {n_bad} negative values")
        n_na = int(values.isna().sum())
        if n_na:
            report.warnings.append(f"{col}: {n_na} missing or non-numeric values")

    for col in CATEGORICAL:
        observed = set(df[col].dropna().astype(str))
        unknown = sorted(observed - set(allowed_values(col)))
        if unknown:
            report.errors.append(f"{col}: unexpected levels {unknown}")
        n_na = int(df[col].isna().sum())
        if n_na:
            report.warnings.append(f"{col}: {n_na} missing values")
    return report
