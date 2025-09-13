"""Slice finding: where does the model do worst?

Every level of one attribute and every combination of levels of two
attributes defines a slice; numeric attributes are cut into quartiles. A
slice is reported when its squared errors are larger than those of the rest
of the data by a Welch t-test that survives Benjamini-Hochberg adjustment
over all slices tested (in the spirit of Chung et al., 2019).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

from loyalty_spend.monitoring.drift import benjamini_hochberg


def _labels(frame: pd.DataFrame, col: str) -> pd.Series:
    s = frame[col]
    if pd.api.types.is_numeric_dtype(s):
        s = pd.qcut(s, 4, duplicates="drop").astype("string")
    return s.astype("string").fillna("<missing>").reset_index(drop=True)


def find_slices(
    y: ArrayLike,
    pred: ArrayLike,
    frame: pd.DataFrame,
    columns: list[str],
    min_size: int = 20,
    fdr: float = 0.05,
) -> pd.DataFrame:
    sq = (np.asarray(pred, float) - np.asarray(y, float)) ** 2
    overall = float(np.sqrt(sq.mean()))
    labels = {c: _labels(frame, c) for c in columns}
    rows = []
    groups = [(c,) for c in columns] + list(itertools.combinations(columns, 2))
    for cols in groups:
        keys = (
            labels[cols[0]]
            if len(cols) == 1
            else labels[cols[0]] + " & " + labels[cols[1]]
        )
        for key in keys.unique():
            mask = (keys == key).to_numpy()
            n = int(mask.sum())
            if n < min_size or n > len(sq) - min_size:
                continue
            test = stats.ttest_ind(
                sq[mask], sq[~mask], equal_var=False, alternative="greater"
            )
            rows.append(
                {
                    "slice": " | ".join(cols) + " = " + str(key),
                    "n": n,
                    "rmse": float(np.sqrt(sq[mask].mean())),
                    "rmse_ratio": float(np.sqrt(sq[mask].mean())) / overall,
                    "p_value": float(test.pvalue),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_adjusted"] = benjamini_hochberg(table["p_value"])
    table["flagged"] = table["p_adjusted"] < fdr
    return table.sort_values(
        ["flagged", "rmse_ratio"], ascending=[False, False]
    ).reset_index(drop=True)
