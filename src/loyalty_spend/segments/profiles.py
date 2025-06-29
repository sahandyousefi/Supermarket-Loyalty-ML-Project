"""Plain-language profiles of segments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from loyalty_spend.data.schema import CATEGORICAL, NUMERIC


def profile_segments(
    X: pd.DataFrame, labels: ArrayLike, spend: ArrayLike | None = None
) -> pd.DataFrame:
    """Size, typical numeric values and dominant levels of every segment.

    Numeric columns are summarised by their median, categorical columns by
    their most common level and its share within the segment, and the lift
    of that level over its share in the whole base, so that a profile reads
    as "mostly Europe (2.1 times the base rate)".
    """
    labels = np.asarray(labels, int)
    frame = X.reset_index(drop=True).copy()
    if spend is not None:
        frame["spend"] = np.asarray(spend, float)
    rows = []
    for seg in np.unique(labels):
        part = frame[labels == seg]
        row: dict[str, object] = {
            "segment": int(seg),
            "size": len(part),
            "share": len(part) / len(frame),
        }
        for col in [c for c in NUMERIC if c in part]:
            row[f"{col}_median"] = float(part[col].median())
        if "spend" in part:
            row["spend_mean"] = float(part["spend"].mean())
        for col in [c for c in CATEGORICAL if c in part]:
            counts = part[col].astype("string").value_counts(normalize=True)
            if counts.empty:
                continue
            level, share = counts.index[0], float(counts.iloc[0])
            base = float((frame[col].astype("string") == level).mean())
            row[f"{col}_mode"] = level
            row[f"{col}_share"] = share
            row[f"{col}_lift"] = share / base if base else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
