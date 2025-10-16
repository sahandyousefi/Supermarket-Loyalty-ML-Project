"""Control charts on a stream of scored batches.

Once the spend of scored members is known, every batch yields one
standardised mean residual per tenure band,
:math:`(\\bar e_{jt} - 0) / (s_j / \\sqrt{n_{jt}})`, with :math:`s_j` the
out-of-fold residual standard deviation of band :math:`j`. The univariate
charts watch the sum of the band statistics divided by the square root of
the number of bands, which is standard normal in control; the MHWMA chart
watches the vector.

Without time-stamped data the stream is simulated by drawing members with
replacement from the out-of-fold residuals, and a change is injected by
lowering the spend of one band from a given batch on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from loyalty_spend.monitoring.control_charts import (
    cusum_chart,
    ewma_chart,
    hwma_chart,
    mhwma_chart,
)

# Limits for an in-control run length of 370 batches, from compare_charts
# with 2,000 replications (see docs/methodology.md, section 12c).
LIMITS = {"ewma": 2.713, "cusum": 4.786, "hwma": 2.801, "mhwma": 16.948}


def band_stream(
    residuals: ArrayLike,
    bands: ArrayLike,
    batch_size: int = 100,
    n_batches: int = 200,
    shift: float = 0.0,
    band: str | None = None,
    start: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """Standardised mean residual per band for every simulated batch."""
    e = np.asarray(residuals, float)
    b = np.asarray(bands).astype(str)
    levels = sorted(set(b))
    centre = {lv: e[b == lv].mean() for lv in levels}
    sd = {lv: e[b == lv].std(ddof=1) for lv in levels}
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_batches):
        idx = rng.integers(0, len(e), batch_size)
        be, bb = e[idx].copy(), b[idx]
        if band is not None and t >= start:
            be[bb == band] -= shift
        row = {}
        for lv in levels:
            sel = bb == lv
            n = int(sel.sum())
            row[lv] = (
                (be[sel].mean() - centre[lv]) / (sd[lv] / np.sqrt(n)) if n else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=levels)


def chart_stream(stream: pd.DataFrame, lam: float = 0.1) -> pd.DataFrame:
    """First signal of each chart, as a one-based batch number."""
    overall = stream.sum(axis=1).to_numpy() / np.sqrt(stream.shape[1])
    results = {
        "EWMA": ewma_chart(overall, lam, LIMITS["ewma"]),
        "CUSUM": cusum_chart(overall, 0.5, LIMITS["cusum"]),
        "HWMA": hwma_chart(overall, lam, LIMITS["hwma"]),
        "MHWMA": mhwma_chart(stream.to_numpy(), lam, LIMITS["mhwma"]),
    }
    return pd.DataFrame(
        [
            {
                "chart": name,
                "first_signal": r.first_signal,
                "signals": int(r.signals.sum()),
            }
            for name, r in results.items()
        ]
    )
