r"""Convergence diagnostics for MCMC draws.

Draws are arrays of shape ``(chains, iterations)``. The split :math:`\hat R`
cuts every chain in half and compares between- and within-chain variance
(Gelman et al., 2013; Vehtari et al., 2021); values above 1.01 signal that
the chains have not mixed. The effective sample size uses the multi-chain
autocorrelation estimate with Geyer's initial monotone sequence, and tells
how many independent draws the correlated chains are worth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def _split(draws: np.ndarray) -> np.ndarray:
    half = draws.shape[1] // 2
    return np.concatenate([draws[:, :half], draws[:, half : 2 * half]], axis=0)


def split_rhat(draws: ArrayLike) -> float:
    x = _split(np.asarray(draws, float))
    m, n = x.shape
    chain_means = x.mean(axis=1)
    within = x.var(axis=1, ddof=1).mean()
    between = n * chain_means.var(ddof=1)
    var_plus = (n - 1) / n * within + between / n
    return float(np.sqrt(var_plus / within)) if within > 0 else float("nan")


def _autocovariance(x: np.ndarray) -> np.ndarray:
    n = len(x)
    centred = x - x.mean()
    size = 2 ** int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(centred, size)
    acov = np.fft.irfft(f * np.conj(f), size)[:n] / n
    return acov


def effective_sample_size(draws: ArrayLike) -> float:
    x = np.asarray(draws, float)
    m, n = x.shape
    acov = np.array([_autocovariance(chain) for chain in x])
    chain_var = acov[:, 0] * n / (n - 1)
    within = chain_var.mean()
    var_plus = within * (n - 1) / n + x.mean(axis=1).var(ddof=1) if m > 1 else within
    if var_plus <= 0:
        return float(m * n)
    rho = 1 - (within - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    # Geyer: sum pairs of autocorrelations while positive, monotone.
    pairs = rho[:-1:2] + rho[1::2]
    total, previous = 0.0, np.inf
    for value in pairs:
        if value <= 0:
            break
        value = min(value, previous)
        total += value
        previous = value
    tau = -1 + 2 * total
    return float(m * n / max(tau, 1.0 / np.log10(m * n)))


def summarise_draws(draws: dict[str, np.ndarray]) -> pd.DataFrame:
    """Mean, sd, 90% interval, R-hat and ESS for named (chains, iterations) arrays."""
    rows = []
    for name, x in draws.items():
        flat = x.reshape(-1)
        rows.append(
            {
                "parameter": name,
                "mean": float(flat.mean()),
                "sd": float(flat.std(ddof=1)),
                "q05": float(np.quantile(flat, 0.05)),
                "q95": float(np.quantile(flat, 0.95)),
                "rhat": split_rhat(x),
                "ess": effective_sample_size(x),
            }
        )
    return pd.DataFrame(rows)
