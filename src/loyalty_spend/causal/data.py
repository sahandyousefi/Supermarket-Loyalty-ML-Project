r"""Causal question about the sign-up promotion.

Members who joined through the promotion may spend differently from members
who did not, but the comparison of raw means mixes the effect of the
promotion with differences in who took it up. With potential outcomes
:math:`Y(1)` and :math:`Y(0)` (annual spend had the member joined with or
without the promotion), the estimands are

.. math::

    \tau_{\mathrm{ATE}} = \mathbb E[Y(1) - Y(0)], \qquad
    \tau_{\mathrm{ATT}} = \mathbb E[Y(1) - Y(0) \mid T = 1].

They are identified from observational data under three assumptions:

1. *Consistency*: the observed spend is the potential outcome of the
   treatment actually received.
2. *Unconfoundedness*: :math:`(Y(0), Y(1)) \perp T \mid X`, that is, the
   recorded attributes capture everything that drove both promotion uptake
   and spend.
3. *Overlap*: :math:`0 < \Pr(T = 1 \mid X) < 1` for every member.

The first-month variables are measured after joining and can themselves be
affected by the promotion. Adjusting for them would block part of the
effect, so the default covariates are those fixed at sign-up: region,
joining month and tenure band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from loyalty_spend.data.schema import TARGET

TREATMENT = "promotion"
PRE_TREATMENT = ["region", "joining_month", "loyalty_years"]
POST_TREATMENT = ["first_month", "items_in_first_month"]


@dataclass
class CausalData:
    """Covariates, binary treatment and outcome of an observational study."""

    X: pd.DataFrame
    t: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        if not (len(self.X) == len(self.t) == len(self.y)):
            raise ValueError("X, t and y must have the same length")
        if not set(np.unique(self.t)) <= {0, 1}:
            raise ValueError("treatment must be coded 0 or 1")
        if self.t.min() == self.t.max():
            raise ValueError("need both treated and control units")

    @property
    def n_treated(self) -> int:
        return int(self.t.sum())

    @property
    def n_control(self) -> int:
        return int(len(self.t) - self.t.sum())

    def naive_difference(self) -> float:
        """Difference in mean outcome between treated and control units."""
        return float(self.y[self.t == 1].mean() - self.y[self.t == 0].mean())


def from_frame(
    df: pd.DataFrame,
    covariates: list[str] | None = None,
    treatment: str = TREATMENT,
    outcome: str = TARGET,
) -> CausalData:
    """Build the study from a cleaned frame; rows with missing values are dropped."""
    covariates = list(PRE_TREATMENT if covariates is None else covariates)
    if treatment in covariates:
        raise ValueError("the treatment cannot also be a covariate")
    cols = [*covariates, treatment, outcome]
    data = df[cols].dropna()
    t = (data[treatment].astype(str) == "Yes").to_numpy(dtype=int)
    y = data[outcome].astype(float).to_numpy()
    return CausalData(data[covariates].reset_index(drop=True), t, y)
