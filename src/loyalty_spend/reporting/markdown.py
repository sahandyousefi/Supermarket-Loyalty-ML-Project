"""Plain Markdown tables without an extra dependency."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def format_value(value: Any, digits: int = 4) -> str:
    """Floats to a fixed number of decimals, missing values to blanks."""
    if value is None:
        return ""
    if isinstance(value, float | np.floating):
        return "" if np.isnan(value) else f"{value:.{digits}f}"
    return str(value).replace("|", "\\|")


def markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    """Render ``df`` as a GitHub-flavoured Markdown table, ignoring the index."""
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(format_value(v, digits) for v in r) + " |"
        for r in df.itertuples(index=False)
    ]
    return "\n".join([head, sep, *rows])
