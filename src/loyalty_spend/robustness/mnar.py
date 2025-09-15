"""Sensitivity to values missing not at random.

Multiple imputation assumes that, given the observed attributes, members
with a missing first-month spend resemble members without. If the gaps are
there because spend was unusual, that fails and cannot be detected from the
data. The delta adjustment of the pattern-mixture approach (van Buuren,
2018, section 3.8) shifts every imputed value by a stated amount and
reports how the quantity of interest moves; a conclusion that holds over a
plausible range of shifts does not depend on the missing-at-random
assumption.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from loyalty_spend.robustness.imputation import multiple_imputations

Quantity = Callable[[pd.DataFrame], float]


def delta_sensitivity(
    X: pd.DataFrame,
    quantity: Quantity,
    deltas: tuple[float, ...] = (-6.0, -3.0, 0.0, 3.0, 6.0),
    column: str = "first_month",
    m: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Mean and spread of ``quantity`` over imputations, for each shift."""
    gaps = X[column].isna().to_numpy()
    completed = multiple_imputations(X, m=m, seed=seed)
    rows = []
    for delta in deltas:
        values = []
        for d in completed:
            shifted = d.copy()
            shifted.loc[gaps, column] = np.clip(
                shifted.loc[gaps, column] + delta, 0.0, None
            )
            values.append(quantity(shifted))
        rows.append(
            {
                "delta": delta,
                "estimate": float(np.mean(values)),
                "between_imputation_sd": float(np.std(values, ddof=1)),
                "n_missing": int(gaps.sum()),
            }
        )
    return pd.DataFrame(rows)
