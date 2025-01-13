"""Reading raw files and returning validated, cleaned frames."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read a CSV as strings so that placeholder values are not lost."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; see data/README.md")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]
