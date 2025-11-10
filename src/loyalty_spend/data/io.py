"""Reading raw files and returning validated, cleaned frames."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from loyalty_spend.data.cleaning import CleaningReport, clean_loyalty
from loyalty_spend.data.schema import validate

logger = logging.getLogger(__name__)


PARQUET_SUFFIXES = {".parquet", ".pq"}


def as_text(df: pd.DataFrame) -> pd.DataFrame:
    """Columns as strings with missing values as NaN, as read from a CSV.

    Cleaning works on text so that placeholders such as ``"-"`` are seen
    before any type conversion; a typed source is brought to the same form.
    """
    return df.astype(object).apply(lambda col: col.map(str).where(col.notna()))


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet file as strings so that placeholders are not lost."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; see data/README.md")
    if path.suffix.lower() in PARQUET_SUFFIXES:
        df = as_text(pd.read_parquet(path))
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    """Write CSV or, by suffix, Parquet."""
    path = Path(path)
    if path.suffix.lower() in PARQUET_SUFFIXES:
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def load_clean(
    path: str | Path, impute: bool = True, require_target: bool = True
) -> tuple[pd.DataFrame, CleaningReport]:
    """Read, clean and validate one file.

    Validation errors raise; warnings are logged.
    """
    clean, report = clean_loyalty(
        read_raw(path), impute=impute, require_target=require_target
    )
    result = validate(clean, require_target=require_target)
    for warning in result.warnings:
        logger.warning("%s: %s", Path(path).name, warning)
    result.raise_for_errors()
    return clean, report


def load_modeling_data(
    train_path: str | Path, test_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training frame with target and scoring frame without, both cleaned."""
    train, _ = load_clean(train_path, impute=False)
    test, _ = load_clean(test_path, impute=False, require_target=False)
    return train, test
