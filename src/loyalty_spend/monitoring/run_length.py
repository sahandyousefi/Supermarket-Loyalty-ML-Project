"""Run lengths of control charts by Monte Carlo.

A chart is judged by its run length, the number of batches until it
signals: long when the process is in control (ARL0), short after a shift
(ARL1). Streams are simulated in blocks, one row per replication, and the
chart statistics are computed for all replications at once. Runs that do
not signal within the horizon are censored at it and reported, so that a
mean run length is never silently biased downwards.

Zero-state run lengths start the shift at the first batch. Steady-state
performance is measured by the conditional expected delay: the shift
starts at batch ``tau``, runs that signal before it are discarded as false
alarms, and the delay is counted from ``tau``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from loyalty_spend.monitoring.control_charts import _hwma_factor


def _ewma(x: np.ndarray, lam: float, L: float, exact: bool = True) -> np.ndarray:
    z = np.empty_like(x)
    prev = np.zeros(x.shape[0])
    for i in range(x.shape[1]):
        prev = lam * x[:, i] + (1 - lam) * prev
        z[:, i] = prev
    t = np.arange(1, x.shape[1] + 1)
    factor = 1 - (1 - lam) ** (2 * t) if exact else np.ones(x.shape[1])
    return np.abs(z) > L * np.sqrt(lam / (2 - lam) * factor)


def _cusum(x: np.ndarray, k: float, h: float) -> np.ndarray:
    out = np.empty(x.shape, bool)
    cp = np.zeros(x.shape[0])
    cm = np.zeros(x.shape[0])
    for i in range(x.shape[1]):
        cp = np.maximum(0.0, x[:, i] - k + cp)
        cm = np.maximum(0.0, -k - x[:, i] + cm)
        out[:, i] = (cp > h) | (cm > h)
    return out


def _hwma(x: np.ndarray, lam: float, L: float) -> np.ndarray:
    n = x.shape[1]
    past = np.zeros_like(x)
    past[:, 1:] = np.cumsum(x, axis=1)[:, :-1] / np.arange(1, n)
    stat = lam * x + (1 - lam) * past
    return np.abs(stat) > L * np.sqrt(_hwma_factor(n, lam))


def _mhwma(x: np.ndarray, lam: float, h: float) -> np.ndarray:
    # x has shape (replications, batches, dimension) with identity covariance.
    n = x.shape[1]
    past = np.zeros_like(x)
    past[:, 1:] = np.cumsum(x, axis=1)[:, :-1] / np.arange(1, n)[None, :, None]
    hv = lam * x + (1 - lam) * past
    return (hv**2).sum(axis=2) / _hwma_factor(n, lam) > h


_CHARTS: dict[str, Callable[..., np.ndarray]] = {
    "ewma": _ewma,
    "cusum": _cusum,
    "hwma": _hwma,
    "mhwma": _mhwma,
}


@dataclass
class RunLengthSummary:
    mean: float
    se: float
    median: float
    censored: float
    n: int


def summarise(run_lengths: np.ndarray, horizon: int) -> RunLengthSummary:
    rl = np.asarray(run_lengths, float)
    return RunLengthSummary(
        float(rl.mean()),
        float(rl.std(ddof=1) / np.sqrt(len(rl))) if len(rl) > 1 else np.nan,
        float(np.median(rl)),
        float((rl > horizon).mean()),
        len(rl),
    )


def simulate_run_lengths(
    chart: str,
    params: dict[str, Any],
    n_reps: int = 2000,
    horizon: int = 5000,
    shift: float | np.ndarray = 0.0,
    scale: float = 1.0,
    tau: int = 0,
    dim: int = 1,
    seed: int = 0,
    block: int = 500,
) -> np.ndarray:
    """Run lengths counted from batch ``tau``; censored runs get ``horizon + 1``.

    Before ``tau`` the batch means are standard normal; from ``tau`` on they
    have mean ``shift`` (a vector for ``mhwma``) and standard deviation
    ``scale``. Runs with a signal before ``tau`` are dropped.
    """
    if chart not in _CHARTS:
        raise KeyError(f"unknown chart '{chart}'")
    rng = np.random.default_rng(seed)
    out = []
    length = tau + horizon
    for start in range(0, n_reps, block):
        m = min(block, n_reps - start)
        shape = (m, length, dim) if chart == "mhwma" else (m, length)
        x = rng.standard_normal(shape)
        x[:, tau:] = x[:, tau:] * scale + shift
        signal = _CHARTS[chart](x, **params)
        first = np.where(signal.any(axis=1), signal.argmax(axis=1) + 1, length + 1)
        keep = first > tau
        out.append(first[keep] - tau)
    return np.concatenate(out)


# The limit parameter of each chart, which run length increases with.
LIMIT_PARAMETER = {"ewma": "L", "cusum": "h", "hwma": "L", "mhwma": "h"}


def calibrate(
    chart: str,
    params: dict[str, Any],
    target: float,
    low: float,
    high: float,
    n_reps: int = 2000,
    dim: int = 1,
    seed: int = 0,
    tol: float = 0.005,
) -> tuple[float, RunLengthSummary]:
    """Control limit that gives in-control run length ``target``.

    Bisection on the limit, with the same random streams at every step so
    that the estimated run length is monotone in the limit.
    """
    name = LIMIT_PARAMETER[chart]
    horizon = int(10 * target)

    def arl(limit: float) -> RunLengthSummary:
        rl = simulate_run_lengths(
            chart, {**params, name: limit}, n_reps, horizon, dim=dim, seed=seed
        )
        return summarise(np.minimum(rl, horizon + 1), horizon)

    if arl(low).mean > target or arl(high).mean < target:
        raise ValueError("target run length is not bracketed by [low, high]")
    while high - low > tol:
        mid = (low + high) / 2
        if arl(mid).mean < target:
            low = mid
        else:
            high = mid
    limit = (low + high) / 2
    return limit, arl(limit)
