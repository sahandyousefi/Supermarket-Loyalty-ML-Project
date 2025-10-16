"""Comparison of control charts at a common in-control run length.

Every chart is first calibrated to the same in-control run length, then its
run length is simulated for mean shifts of several sizes and for an
increase in spread. Two settings are reported: zero state, with the shift
present from the first batch, and steady state, with the shift starting
after ``tau`` in-control batches and false alarms before it discarded.

The batch statistic is the vector of mean standardised residuals of the
``dim`` tenure bands. Univariate charts watch the standardised overall
mean, :math:`\\sum_j x_j / \\sqrt{d}`, so a shift :math:`\\delta` confined
to one band reaches them as :math:`\\delta / \\sqrt{d}`; the MHWMA chart
watches the vector itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from loyalty_spend.monitoring.run_length import (
    LIMIT_PARAMETER,
    calibrate,
    simulate_run_lengths,
    summarise,
)


@dataclass
class Design:
    name: str
    chart: str
    params: dict[str, Any]
    bracket: tuple[float, float]
    multivariate: bool = False
    calibrated: dict[str, Any] = field(default_factory=dict)


DEFAULT_DESIGNS = (
    Design("Shewhart", "ewma", {"lam": 1.0}, (2.0, 4.0)),
    Design("EWMA 0.1", "ewma", {"lam": 0.1}, (2.0, 3.5)),
    Design("CUSUM k=0.5", "cusum", {"k": 0.5}, (2.0, 7.0)),
    Design("HWMA 0.1", "hwma", {"lam": 0.1}, (1.5, 4.0)),
    Design("MHWMA 0.1", "mhwma", {"lam": 0.1}, (5.0, 40.0), multivariate=True),
)


def compare_charts(
    designs: Sequence[Design] = DEFAULT_DESIGNS,
    deltas: Sequence[float] = (0.5, 1.0, 2.0),
    scale: float = 1.5,
    dim: int = 5,
    target: float = 370.0,
    tau: int = 50,
    n_reps: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for d in designs:
        k = dim if d.multivariate else 1
        limit, ic = calibrate(
            d.chart, d.params, target, *d.bracket, n_reps=n_reps, dim=k, seed=seed
        )
        d.calibrated = {**d.params, LIMIT_PARAMETER[d.chart]: limit}
        scenarios: list[tuple[str, Any, float]] = [
            (f"shift {delta:g} in one band", delta, 1.0) for delta in deltas
        ]
        scenarios.append((f"spread x{scale:g}", 0.0, scale))
        for label, delta, sd in scenarios:
            if d.multivariate:
                shift: Any = np.r_[delta, np.zeros(dim - 1)]
            else:
                shift = delta / np.sqrt(dim)
            for state, start in (("zero state", 0), ("steady state", tau)):
                rl = simulate_run_lengths(
                    d.chart,
                    d.calibrated,
                    n_reps,
                    horizon=int(10 * target),
                    shift=shift,
                    scale=sd,
                    tau=start,
                    dim=k,
                    seed=seed + 1,
                )
                s = summarise(rl, int(10 * target))
                rows.append(
                    {
                        "chart": d.name,
                        "limit": limit,
                        "arl0": ic.mean,
                        "scenario": label,
                        "state": state,
                        "arl": s.mean,
                        "se": s.se,
                    }
                )
    return pd.DataFrame(rows)
