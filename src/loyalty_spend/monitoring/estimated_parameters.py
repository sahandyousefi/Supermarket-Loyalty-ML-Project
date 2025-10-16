r"""Run length when the in-control mean and spread are estimated.

Control limits assume a known in-control mean and standard deviation. In
practice both come from a Phase I sample of :math:`m` batches, and the
chart then runs on :math:`(x - \hat\mu)/\hat\sigma`, which in control has
mean :math:`-\hat\mu/\hat\sigma` and standard deviation
:math:`1/\hat\sigma`. The run length given the estimates (the conditional
ARL) varies from one Phase I sample to the next; its spread, not only its
mean, decides whether the nominal false alarm rate can be trusted (Jensen
et al., 2006; Saleh et al., 2015).

Conditional run lengths of the EWMA chart with asymptotic limits are
computed exactly enough by the Markov chain approximation of Brook and
Evans (1972): the in-control region is cut into :math:`N` states, the
transition probabilities follow from the normal distribution of the next
batch, and the vector of run lengths solves
:math:`(I - P)\,\mathrm{ARL} = \mathbf 1`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def ewma_arl_markov(
    lam: float,
    L: float,
    shift: float = 0.0,
    scale: float = 1.0,
    n_states: int = 201,
) -> float:
    """ARL of a two-sided EWMA chart with asymptotic limits, from zero."""
    h = L * np.sqrt(lam / (2 - lam))
    w = 2 * h / n_states
    centres = -h + w * (np.arange(n_states) + 0.5)
    # Next statistic lam * x + (1 - lam) * c falls into state j.
    upper = (centres[None, :] + w / 2 - (1 - lam) * centres[:, None]) / lam
    lower = (centres[None, :] - w / 2 - (1 - lam) * centres[:, None]) / lam
    P = norm.cdf(upper, shift, scale) - norm.cdf(lower, shift, scale)
    arl = np.linalg.solve(np.eye(n_states) - P, np.ones(n_states))
    return float(np.interp(0.0, centres, arl))


def conditional_arl(
    m: int,
    lam: float = 0.1,
    L: float = 2.814,
    n_phase1: int = 2000,
    seed: int = 0,
    n_states: int = 151,
) -> np.ndarray:
    """In-control ARL given estimates from ``m`` standard normal batches."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_phase1, m))
    mu, sd = x.mean(axis=1), x.std(axis=1, ddof=1)
    return np.array(
        [
            ewma_arl_markov(lam, L, -a / s, 1 / s, n_states)
            for a, s in zip(mu, sd, strict=True)
        ]
    )


def estimation_effect(
    sizes: tuple[int, ...] = (25, 50, 100, 250, 1000),
    lam: float = 0.1,
    L: float = 2.814,
    n_phase1: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Distribution of the conditional in-control ARL by Phase I size."""
    nominal = ewma_arl_markov(lam, L)
    rows = []
    for m in sizes:
        arl = conditional_arl(m, lam, L, n_phase1, seed)
        rows.append(
            {
                "phase1_batches": m,
                "nominal_arl": nominal,
                "mean": float(arl.mean()),
                "p10": float(np.quantile(arl, 0.1)),
                "median": float(np.median(arl)),
                "p90": float(np.quantile(arl, 0.9)),
                "share_below_80pct": float((arl < 0.8 * nominal).mean()),
            }
        )
    return pd.DataFrame(rows)
