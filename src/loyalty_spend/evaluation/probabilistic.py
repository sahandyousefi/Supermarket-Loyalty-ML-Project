"""Cross-validated comparison of distributional forecasters.

Each forecaster is fitted on the same folds as the point models and scored
with the quantile-grid CRPS, the 90% interval score and the coverage of the
90% interval. Differences in CRPS are tested with the corrected resampled
t-test, as for the point models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from loyalty_spend.arrays import FloatArray, Split
from loyalty_spend.evaluation.cv import corrected_ttest
from loyalty_spend.evaluation.scoring import (
    calibration_test,
    crps_quantiles,
    interval_score,
    pit_quantiles,
)
from loyalty_spend.models.forecasters import TAU_GRID, QuantileForecaster


@dataclass
class DistributionCVResult:
    name: str
    folds: pd.DataFrame
    pit: FloatArray
    n_train: float
    n_test: float

    def summary(self) -> dict[str, float | str]:
        out: dict[str, float | str] = {"model": self.name}
        for col in ("crps", "interval_score", "coverage", "width"):
            out[f"{col}_mean"] = float(self.folds[col].mean())
        out["pit_chi2_p_value"] = calibration_test(self.pit)["chi2_p_value"]
        return out


def cross_validate_distribution(
    name: str,
    factory: Callable[[], QuantileForecaster],
    X: pd.DataFrame,
    y: ArrayLike,
    splits: list[Split],
) -> DistributionCVResult:
    y = np.asarray(y, float)
    lo_idx, hi_idx = TAU_GRID.index(0.05), TAU_GRID.index(0.95)
    rows, pits = [], []
    for i, (tr, te) in enumerate(splits):
        q = factory().fit(X.iloc[tr], y[tr]).quantiles(X.iloc[te])
        lo, hi = q[:, lo_idx], q[:, hi_idx]
        rows.append(
            {
                "fold": i,
                "crps": float(crps_quantiles(y[te], q, TAU_GRID).mean()),
                "interval_score": float(interval_score(y[te], lo, hi, 0.1).mean()),
                "coverage": float(np.mean((y[te] >= lo) & (y[te] <= hi))),
                "width": float(np.mean(hi - lo)),
            }
        )
        pits.append(pit_quantiles(y[te], q, TAU_GRID))
    n_test = float(np.mean([len(te) for _, te in splits]))
    return DistributionCVResult(
        name, pd.DataFrame(rows), np.concatenate(pits), len(y) - n_test, n_test
    )


def compare_distributions(
    results: list[DistributionCVResult], reference: str
) -> pd.DataFrame:
    """Summary table with the CRPS difference to ``reference`` and its p-value."""
    by_name = {r.name: r for r in results}
    if reference not in by_name:
        raise KeyError(f"reference '{reference}' not in results")
    base = by_name[reference]
    ref = base.folds["crps"].to_numpy()
    rows = []
    for r in results:
        d = r.folds["crps"].to_numpy() - ref
        _, p = (
            (np.nan, np.nan)
            if r.name == reference
            else corrected_ttest(d, base.n_train, base.n_test)
        )
        rows.append({**r.summary(), "crps_diff": float(d.mean()), "p_value": p})
    return pd.DataFrame(rows).sort_values("crps_mean").reset_index(drop=True)
