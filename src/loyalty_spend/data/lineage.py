"""Checks that two extracts describe the same customers before they are joined.

Matching ids are not enough: if attributes that cannot change (region, tenure
band, joining month) agree no more often than chance, the files come from
different populations or different draws and must not be linked.
"""

from __future__ import annotations

import pandas as pd

from loyalty_spend.data.schema import CATEGORICAL, ID


def chance_agreement(a: pd.Series, b: pd.Series) -> float:
    """Agreement expected if the two columns were independent.

    Equal to the sum over levels of the product of the marginal shares, the
    quantity used in Cohen's kappa.
    """
    pa = a.astype(str).value_counts(normalize=True)
    pb = b.astype(str).value_counts(normalize=True)
    return float((pa * pb.reindex(pa.index).fillna(0.0)).sum())


def compare_extracts(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: list[str] | None = None,
    key: str = ID,
) -> pd.DataFrame:
    """Observed and chance agreement per attribute on the shared ids.

    ``kappa`` is Cohen's kappa; values near zero mean the attribute carries no
    information about the matching record.
    """
    columns = columns or [c for c in CATEGORICAL if c in left and c in right]
    merged = left.merge(right, on=key, suffixes=("_l", "_r"))
    rows = []
    for col in columns:
        a, b = merged[f"{col}_l"].astype(str), merged[f"{col}_r"].astype(str)
        observed = float((a == b).mean())
        expected = chance_agreement(a, b)
        kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
        rows.append(
            {
                "column": col,
                "n_shared_ids": len(merged),
                "observed_agreement": observed,
                "chance_agreement": expected,
                "kappa": kappa,
            }
        )
    return pd.DataFrame(rows)


def linkable(table: pd.DataFrame, min_kappa: float = 0.8) -> bool:
    """True when every attribute agrees well beyond chance."""
    return bool((table["kappa"] >= min_kappa).all())
