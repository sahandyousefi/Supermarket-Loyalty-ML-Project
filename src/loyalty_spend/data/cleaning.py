"""Cleaning of the raw loyalty extract according to the data dictionary."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

PLACEHOLDERS = {"", ".", "-", "na", "n/a", "nan", "null", "none", "?"}


@dataclass
class CleaningReport:
    """Number of values changed per column and rule."""

    counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def add(self, column: str, rule: str, n: int) -> None:
        if n:
            key = (column, rule)
            self.counts[key] = self.counts.get(key, 0) + int(n)

    def to_frame(self) -> pd.DataFrame:
        rows = [{"column": c, "rule": r, "n": n} for (c, r), n in self.counts.items()]
        return pd.DataFrame(rows, columns=["column", "rule", "n"])


def coerce_numeric(series: pd.Series) -> tuple[pd.Series, int, int]:
    """Convert to float, treating placeholder strings such as '.' as missing.

    Returns the converted series, the number of placeholders and the number
    of other non-empty entries that could not be parsed.
    """
    text = series.astype("string").str.strip()
    is_placeholder = text.str.lower().isin(PLACEHOLDERS).fillna(False)
    text = text.mask(is_placeholder)
    values = pd.to_numeric(text, errors="coerce")
    n_invalid = int((values.isna() & text.notna()).sum())
    return values.astype(float), int(is_placeholder.sum()), n_invalid
