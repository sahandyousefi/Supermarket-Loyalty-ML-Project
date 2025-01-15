"""Checks that two extracts describe the same customers before they are joined.

Matching ids are not enough: if attributes that cannot change (region, tenure
band, joining month) agree no more often than chance, the files come from
different populations or different draws and must not be linked.
"""

from __future__ import annotations

import pandas as pd


def chance_agreement(a: pd.Series, b: pd.Series) -> float:
    """Agreement expected if the two columns were independent.

    Equal to the sum over levels of the product of the marginal shares, the
    quantity used in Cohen's kappa.
    """
    pa = a.astype(str).value_counts(normalize=True)
    pb = b.astype(str).value_counts(normalize=True)
    return float((pa * pb.reindex(pa.index).fillna(0.0)).sum())
