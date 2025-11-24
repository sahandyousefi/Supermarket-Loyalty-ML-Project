"""Checks on a member file at intake, before cleaning decisions are final.

Repeated customer ids and repeated records usually come from an export run
twice or a join that multiplied rows; both inflate the weight of some
members in every estimate. Near duplicates, pairs of records that agree
on every category and differ by at most a cent in every amount, can be the
same member entered twice with a rounding difference. They are reported
for review, not removed, because two genuine members can look alike when
there are few attributes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loyalty_spend.data.schema import ID


def duplicate_ids(frame: pd.DataFrame, id_column: str = ID) -> pd.DataFrame:
    """Rows whose customer id occurs more than once."""
    dup = frame[id_column].duplicated(keep=False) & frame[id_column].notna()
    return frame[dup].sort_values(id_column)


def exact_duplicates(frame: pd.DataFrame, id_column: str = ID) -> pd.DataFrame:
    """Rows identical to another row in everything but the id."""
    columns = [c for c in frame.columns if c != id_column]
    return frame[frame.duplicated(subset=columns, keep=False)]


def near_duplicates(
    frame: pd.DataFrame,
    numeric: list[str],
    tolerance: float = 0.01,
    id_column: str = ID,
) -> pd.DataFrame:
    """Pairs that agree on every category and on every amount within ``tolerance``.

    The default of one cent catches a record entered twice with a rounding
    difference. A relative criterion such as a small Gower distance does
    not work here: within a joining month first-month spend varies by a few
    tenths on a range of fifteen, so hundreds of genuine pairs would be
    flagged. Exact duplicates are reported separately and left out.
    """
    categories = [c for c in frame.columns if c not in (*numeric, id_column)]
    text = frame[categories].astype(object).fillna("<missing>").astype(str)
    keys = text.agg("|".join, axis=1).to_numpy()
    amounts = frame[numeric].to_numpy(float)
    ids = frame[id_column].to_numpy()
    rows = []
    for key in np.unique(keys):
        idx = np.flatnonzero(keys == key)
        if len(idx) < 2:
            continue
        a = amounts[idx]
        diff = np.abs(a[:, None, :] - a[None, :, :])
        # Two missing values count as equal, one missing value as different.
        both = np.isnan(a[:, None, :]) & np.isnan(a[None, :, :])
        diff = np.where(both, 0.0, np.nan_to_num(diff, nan=np.inf))
        worst = diff.max(axis=2)
        # A small allowance keeps differences of exactly one cent in.
        close = (worst <= tolerance + 1e-9) & (worst > 0)
        i, j = np.where(np.triu(close, k=1))
        rows += [
            {"id_a": ids[idx[p]], "id_b": ids[idx[q]], "max_difference": worst[p, q]}
            for p, q in zip(i, j, strict=True)
        ]
    return pd.DataFrame(rows, columns=["id_a", "id_b", "max_difference"])


def intake_summary(
    frame: pd.DataFrame, numeric: list[str], id_column: str = ID
) -> dict[str, int]:
    return {
        "rows": len(frame),
        "duplicate_id_rows": len(duplicate_ids(frame, id_column)),
        "exact_duplicate_rows": len(exact_duplicates(frame, id_column)),
        "near_duplicate_pairs": len(
            near_duplicates(frame, numeric, id_column=id_column)
        ),
    }
