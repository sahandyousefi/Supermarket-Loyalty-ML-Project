"""Column profiles of an extract.

A profile is the first thing to look at when a new extract arrives: how much
is missing, how many distinct values there are, where the bulk of each
numeric column lies and which levels dominate each category.
"""

from __future__ import annotations

import pandas as pd

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _numeric_summary(values: pd.Series) -> dict[str, float]:
    x = values.dropna()
    if x.empty:
        return {}
    out = {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "max": float(x.max()),
        "zeros": int((x == 0).sum()),
    }
    for q, v in zip(QUANTILES, x.quantile(list(QUANTILES))):
        out[f"q{int(round(100 * q)):02d}"] = float(v)
    return out


def top_levels(values: pd.Series, k: int = 3) -> str:
    """The ``k`` most frequent levels with their shares, as one string."""
    shares = values.dropna().astype(str).value_counts(normalize=True).head(k)
    return ", ".join(f"{lvl} ({share:.0%})" for lvl, share in shares.items())


def profile_frame(df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """One row per column with missingness, cardinality and distribution.

    Columns that are numeric, or that parse as numeric without loss, get
    quantiles and moments; all other columns get their most frequent levels.
    """
    rows = []
    n = len(df)
    for col in df.columns:
        series = df[col]
        parsed = pd.to_numeric(series, errors="coerce")
        numeric = bool(series.notna().any()) and bool(
            (parsed.notna() == series.notna()).all()
        )
        n_missing = int(series.isna().sum())
        row: dict[str, object] = {
            "column": col,
            "kind": "numeric" if numeric else "category",
            "n": n,
            "missing": n_missing,
            "missing_share": n_missing / n if n else 0.0,
            "distinct": int(series.nunique(dropna=True)),
        }
        if numeric:
            row.update(_numeric_summary(parsed))
        else:
            row["top"] = top_levels(series, top_k)
        rows.append(row)
    return pd.DataFrame(rows).set_index("column")
