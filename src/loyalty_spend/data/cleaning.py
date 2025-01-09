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
