r"""Group-sequential design with an alpha-spending function.

The experiment is analysed at :math:`K` looks with information fractions
:math:`0 < t_1 < \dots < t_K = 1`. The score process
:math:`S(t) = Z(t)\sqrt t` is Brownian motion with drift :math:`\theta`,
so its increments are independent. A spending function :math:`\alpha(t)`
fixes how much of the two-sided type I error may be used by each look
(Lan and DeMets, 1983): the O'Brien-Fleming type
:math:`2 - 2\Phi(z_{1-\alpha/2}/\sqrt t)` spends almost nothing early,
the Pocock type :math:`\alpha\log(1 + (e - 1)t)` spends evenly.

Boundaries are found one look at a time: the density of :math:`S` on the
paths that have not stopped is carried forward by numerical integration
(Simpson's rule over the continuation region) of its convolution with the
normal increment (Armitage, McPherson and Rowe, 1969), and the boundary
:math:`c_k` solves
:math:`P(\text{first crossing at } k) = \alpha(t_k) - \alpha(t_{k-1})`
(Jennison and Turnbull, 2000, chapter 19). The same recursion with a drift
gives the stopping probabilities, power and expected sample size.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def obrien_fleming_spending(alpha: float) -> Callable[[float], float]:
    z = norm.ppf(1 - alpha / 2)
    return lambda t: float(2 - 2 * norm.cdf(z / np.sqrt(t))) if t > 0 else 0.0


def pocock_spending(alpha: float) -> Callable[[float], float]:
    return lambda t: float(alpha * np.log(1 + (np.e - 1) * t))


SPENDING = {"obrien-fleming": obrien_fleming_spending, "pocock": pocock_spending}


@dataclass
class _State:
    """Density of S on the continuation region of the last look."""

    grid: np.ndarray
    weights: np.ndarray
    density: np.ndarray
    t: float


def _simpson(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    n = n + 1 - n % 2
    grid = np.linspace(a, b, n)
    w = np.ones(n)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    return grid, w * (b - a) / (3 * (n - 1))


def _cross_prob(state: _State | None, t: float, c: float, drift: float) -> float:
    """Probability of continuing to look ``t`` and crossing ``c`` there."""
    edge = c * np.sqrt(t)
    if state is None:
        m0, s0 = drift * t, np.sqrt(t)
        return float(norm.cdf(-edge, m0, s0) + norm.sf(edge, m0, s0))
    dt = t - state.t
    means = state.grid + drift * dt
    tail = norm.cdf(-edge, means, np.sqrt(dt)) + norm.sf(edge, means, np.sqrt(dt))
    return float(np.sum(state.weights * state.density * tail))


def _continue(state: _State | None, t: float, c: float, drift: float, n: int) -> _State:
    edge = c * np.sqrt(t)
    grid, weights = _simpson(-edge, edge, n)
    if state is None:
        dens = norm.pdf(grid, drift * t, np.sqrt(t))
    else:
        dt = t - state.t
        kernel = norm.pdf(grid[:, None], state.grid[None, :] + drift * dt, np.sqrt(dt))
        dens = kernel @ (state.weights * state.density)
    return _State(grid, weights, dens, t)


def _excess(c: float, state: _State | None, t: float, target: float) -> float:
    return _cross_prob(state, t, c, 0.0) - target


def boundaries(
    fractions: Sequence[float],
    alpha: float = 0.05,
    spending: str | Callable[[float], float] = "obrien-fleming",
    n_grid: int = 401,
) -> np.ndarray:
    """Two-sided critical values of the Z statistic at every look.

    ``spending`` names a spending function for the two-sided ``alpha`` or
    is a function of the information fraction itself.
    """
    t = np.asarray(fractions, float)
    if np.any(np.diff(t) <= 0) or t[-1] != 1.0 or t[0] <= 0:
        raise ValueError("fractions must increase strictly to 1")
    spend = SPENDING[spending](alpha) if isinstance(spending, str) else spending
    state: _State | None = None
    out, spent = [], 0.0
    for tk in t:
        target = spend(tk) - spent
        c = brentq(_excess, 0.5, 40.0, args=(state, float(tk), target))
        out.append(c)
        spent += target
        state = _continue(state, tk, c, 0.0, n_grid)
    return np.array(out)


@dataclass
class DesignProperties:
    stop_probabilities: np.ndarray
    power: float
    expected_fraction: float


def design_properties(
    fractions: Sequence[float],
    bounds: Sequence[float],
    drift: float,
    n_grid: int = 401,
) -> DesignProperties:
    """Stopping probabilities, power and expected information under a drift.

    ``drift`` is the expected Z statistic at the final look,
    :math:`\\delta / \\mathrm{SE}` with the full sample.
    """
    state: _State | None = None
    stops = []
    for tk, c in zip(fractions, bounds, strict=True):
        stops.append(_cross_prob(state, tk, c, drift))
        state = _continue(state, tk, c, drift, n_grid)
    stops_arr = np.array(stops)
    t = np.asarray(fractions, float)
    reach = 1 - np.r_[0.0, np.cumsum(stops_arr)[:-1]]
    # Information used: a look at t_k is reached unless stopped before it.
    expected = float(t[0] + np.sum(np.diff(t) * reach[1:]))
    return DesignProperties(stops_arr, float(stops_arr.sum()), expected)
