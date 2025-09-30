"""Distance-based privacy checks for synthetic data.

The distance to closest record (DCR) of a synthetic row is its Gower
distance to the nearest member of the data the synthesiser was fitted on.
Copies and near copies have a DCR close to zero. Because members of a
small population with few attributes lie close to each other anyway, the
synthetic DCRs are compared with those of real members held out from
fitting: synthetic rows should not be closer to the training members than
new real members are. The nearest-neighbour distance ratio (NNDR), the
nearest over the second-nearest distance, is small when a row sits next
to one particular member rather than in a dense region.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loyalty_spend.segments.gower import ORDINAL


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not isinstance(
        s.dtype, pd.CategoricalDtype
    )


def gower_cross(
    rows: pd.DataFrame,
    reference: pd.DataFrame,
    ordinal: dict[str, list[str]] | None = None,
) -> np.ndarray:
    """Gower distances from ``rows`` to ``reference`` with reference ranges."""
    ordinal = ORDINAL if ordinal is None else ordinal
    total = np.zeros((len(rows), len(reference)))
    norm = np.zeros_like(total)
    for col in reference.columns:
        a, b = rows[col], reference[col]
        if col in ordinal:
            levels = ordinal[col]
            ca = pd.Categorical(a.astype("string"), categories=levels).codes
            cb = pd.Categorical(b.astype("string"), categories=levels).codes
            x = np.where(ca >= 0, ca / max(len(levels) - 1, 1), np.nan)
            y = np.where(cb >= 0, cb / max(len(levels) - 1, 1), np.nan)
            d = np.abs(x[:, None] - y[None, :])
        elif _is_numeric(b):
            x, y = a.to_numpy(float), b.to_numpy(float)
            span = np.nanmax(y) - np.nanmin(y)
            d = np.abs(x[:, None] - y[None, :]) / (span if span > 0 else 1.0)
        else:
            x = a.astype("string").to_numpy(dtype=object)
            y = b.astype("string").to_numpy(dtype=object)
            d = (x[:, None] != y[None, :]).astype(float)
            d[pd.isna(x)] = np.nan
            d[:, pd.isna(y)] = np.nan
        valid = ~np.isnan(d)
        total += np.where(valid, d, 0.0)
        norm += valid
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 0, total / norm, np.nan)


def nearest_distances(rows: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    D = gower_cross(rows, reference)
    two = np.sort(D, axis=1)[:, :2]
    with np.errstate(invalid="ignore", divide="ignore"):
        nndr = np.where(two[:, 1] > 0, two[:, 0] / two[:, 1], 1.0)
    return pd.DataFrame({"dcr": two[:, 0], "nndr": nndr})


def privacy_report(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    synth: pd.DataFrame,
    quantile: float = 0.05,
) -> dict[str, float]:
    """Compare synthetic and held-out real rows by their closeness to train."""
    s = nearest_distances(synth, train)
    h = nearest_distances(holdout, train)
    return {
        "exact_copies": float((s["dcr"] == 0).mean()),
        "holdout_exact": float((h["dcr"] == 0).mean()),
        "dcr_q05_synthetic": float(s["dcr"].quantile(quantile)),
        "dcr_q05_holdout": float(h["dcr"].quantile(quantile)),
        "dcr_median_synthetic": float(s["dcr"].median()),
        "dcr_median_holdout": float(h["dcr"].median()),
        "nndr_q05_synthetic": float(s["nndr"].quantile(quantile)),
        "nndr_q05_holdout": float(h["nndr"].quantile(quantile)),
        # Share of synthetic rows closer to train than the 5% quantile of holdout.
        "share_closer_than_holdout_q05": float(
            (s["dcr"] < h["dcr"].quantile(quantile)).mean()
        ),
    }
