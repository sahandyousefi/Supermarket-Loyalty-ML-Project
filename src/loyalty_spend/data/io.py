"""Reading raw files and returning validated, cleaned frames."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from loyalty_spend.data.cleaning import CleaningReport, clean_loyalty
from loyalty_spend.data.schema import validate

logger = logging.getLogger(__name__)


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read a CSV as strings so that placeholder values are not lost."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; see data/README.md")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


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
