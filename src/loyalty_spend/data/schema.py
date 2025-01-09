"""Column specification of the loyalty data and validation against it."""

from __future__ import annotations

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
