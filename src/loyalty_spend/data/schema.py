"""Column specification of the loyalty data and validation against it."""

from __future__ import annotations

from dataclasses import dataclass, field

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
