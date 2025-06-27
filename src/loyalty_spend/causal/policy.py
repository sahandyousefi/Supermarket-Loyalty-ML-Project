r"""Targeting the promotion and the value of a targeting rule.

Offering the promotion to member :math:`i` costs :math:`c` and changes spend
by :math:`\tau(x_i)`, of which a share :math:`m` (the margin) is profit. A
rule :math:`\pi(x) \in \{0, 1\}` is worth, per member, relative to offering
it to nobody,

.. math::

    V(\pi) = \mathbb E\big[\pi(X)\,(m\,\tau(X) - c)\big]
    \approx \frac1n \sum_i \pi(x_i)\,(m\,\psi_i - c),

where :math:`\psi_i` are doubly robust scores, which makes the estimate
unbiased for any fixed rule (Athey and Wager, 2021). The rule must be
learned on other data than the scores it is evaluated on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats


def threshold_rule(tau_hat: ArrayLike, cost: float, margin: float = 1.0) -> np.ndarray:
    """Offer the promotion where its predicted profit exceeds its cost."""
    if margin <= 0:
        raise ValueError("margin must be positive")
    return np.asarray(tau_hat, float) * margin > cost


def policy_value(
    policy: ArrayLike, scores: ArrayLike, cost: float, margin: float = 1.0
) -> dict[str, float]:
    """Doubly robust value per member, with a standard error and interval."""
    pi = np.asarray(policy, float)
    gain = pi * (margin * np.asarray(scores, float) - cost)
    n = len(gain)
    value = float(gain.mean())
    se = float(gain.std(ddof=1) / np.sqrt(n))
    z = stats.norm.ppf(0.975)
    return {
        "value": value,
        "se": se,
        "ci_lower": value - z * se,
        "ci_upper": value + z * se,
        "share_treated": float(pi.mean()),
    }


def compare_policies(
    tau_hat: ArrayLike, scores: ArrayLike, cost: float, margin: float = 1.0
) -> pd.DataFrame:
    """Value of offering the promotion to nobody, everybody, or by the rule."""
    n = len(np.asarray(scores))
    rules = {
        "nobody": np.zeros(n, bool),
        "everybody": np.ones(n, bool),
        "predicted profit above cost": threshold_rule(tau_hat, cost, margin),
    }
    rows = [
        {"policy": name, **policy_value(rule, scores, cost, margin)}
        for name, rule in rules.items()
    ]
    return pd.DataFrame(rows)
