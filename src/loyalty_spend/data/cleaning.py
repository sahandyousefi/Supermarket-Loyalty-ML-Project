"""Cleaning of the raw loyalty extract according to the data dictionary."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from loyalty_spend.data.schema import (
    CATEGORICAL,
    CATEGORIES,
    COLUMNS,
    FILL_VALUES,
    ID,
    LOYALTY_ORDER,
    TARGET,
    allowed_values,
)

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


def normalize_category(series: pd.Series, levels: list[str]) -> tuple[pd.Series, int]:
    """Map case and whitespace variants (for example 'YES') to the canonical level.

    Values that match no level after normalization become missing. Returns the
    cleaned series and the number of values that were rewritten.
    """
    lookup = {level.lower(): level for level in levels}
    text = series.astype("string").str.strip()
    text = text.mask(text.str.lower().isin(PLACEHOLDERS))
    mapped = text.str.lower().map(lookup)
    changed = int((mapped.notna() & (mapped != text)).sum())
    return mapped.astype(object).where(mapped.notna(), np.nan), changed


def clean_loyalty(
    df: pd.DataFrame, impute: bool = True, require_target: bool = True
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a raw extract so that it matches the data dictionary.

    Parameters
    ----------
    df : DataFrame
        Raw data, ideally read with ``dtype=str`` so that placeholders survive.
    impute : bool
        Apply the replacement values of the data dictionary. With ``False``,
        missing values stay missing, which is what the modeling code expects.
    require_target : bool
        Scoring files have no ``spend`` column.

    Returns
    -------
    clean : DataFrame
        Columns in dictionary order, amounts rounded to two decimals,
        ``loyalty_years`` as an ordered categorical.
    report : CleaningReport
        Counts of every change, for the data quality log.
    """
    report = CleaningReport()
    columns = [c for c in COLUMNS if require_target or c != TARGET]
    out = pd.DataFrame(index=df.index)
    out[ID] = pd.to_numeric(df[ID], errors="raise").astype(int)

    for col in [c for c in columns if c not in (ID, *CATEGORICAL)]:
        values, n_placeholder, n_invalid = coerce_numeric(df[col])
        report.add(col, "placeholder to missing", n_placeholder)
        report.add(col, "unparseable to missing", n_invalid)
        n_neg = int((values < 0).sum())
        report.add(col, "negative to missing", n_neg)
        values = values.mask(values < 0)
        if impute:
            report.add(col, "missing filled", int(values.isna().sum()))
            values = values.fillna(FILL_VALUES[col])
        out[col] = values.round(2)
    if "items_in_first_month" in out and impute:
        out["items_in_first_month"] = out["items_in_first_month"].astype(int)

    for col in CATEGORICAL:
        values, n_changed = normalize_category(df[col], CATEGORIES[col])
        report.add(col, "case or whitespace normalized", n_changed)
        if impute:
            report.add(col, "missing filled", int(values.isna().sum()))
            values = values.fillna(FILL_VALUES[col])
        levels = LOYALTY_ORDER if col == "loyalty_years" else allowed_values(col)
        out[col] = pd.Categorical(
            values, categories=levels, ordered=col == "loyalty_years"
        )
    return out[columns], report
