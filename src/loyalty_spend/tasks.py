"""The four deliverables of the original brief, produced by the package.

Task 1  ``clean_data``: loyalty.csv cleaned to the data dictionary.
Task 2  ``spend_by_years``: mean and variance of spend by tenure band.
Task 3  ``base_result``: baseline predictions for test.csv.
Task 4  ``compare_result``: comparison-model predictions for test.csv.
"""

from __future__ import annotations

import pandas as pd

from loyalty_spend.config import Config
from loyalty_spend.data.io import load_clean
from loyalty_spend.data.schema import TARGET


def task1_clean_data(cfg: Config) -> pd.DataFrame:
    clean, _ = load_clean(cfg.paths.raw(cfg.paths.loyalty_file))
    return clean


def task2_spend_by_years(clean: pd.DataFrame) -> pd.DataFrame:
    out = (
        clean.groupby("loyalty_years", observed=True)[TARGET]
        .agg(avg_spend="mean", var_spend="var")
        .reset_index()
    )
    out[["avg_spend", "var_spend"]] = out[["avg_spend", "var_spend"]].round(2)
    out["loyalty_years"] = out["loyalty_years"].astype(str)
    return out
