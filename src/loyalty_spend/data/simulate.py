"""Semi-synthetic benchmark calibrated to the training data.

The supplied training data is almost noise free: a linear model explains
more than 99.9% of the variance, so it cannot show when flexible models are
worth their cost. The simulator keeps the joint distribution of the real
features (rows are resampled) and the fitted linear structure, and adds
controlled departures from it: measurement noise, heteroscedasticity, smooth
nonlinear effects and interactions. Because the noise is known, every model
can be compared with the irreducible error.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    """Departures from the linear reference model.

    Parameters
    ----------
    name : str
    noise : float
        Noise standard deviation as a share of the target's standard deviation.
    heteroscedastic : bool
        Scale the noise with the reference prediction, so high spenders are
        less predictable.
    nonlinear : float
        Weight of the smooth nonlinear terms, in units of the target's
        standard deviation.
    interaction : float
        Weight of the interaction terms, in the same units.
    """

    name: str
    noise: float = 0.05
    heteroscedastic: bool = False
    nonlinear: float = 0.0
    interaction: float = 0.0


SCENARIOS: dict[str, Scenario] = {
    "linear": Scenario("linear", noise=0.05),
    "noisy": Scenario("noisy", noise=0.20, heteroscedastic=True),
    "nonlinear": Scenario("nonlinear", noise=0.05, nonlinear=0.25),
    "interactions": Scenario(
        "interactions", noise=0.05, nonlinear=0.25, interaction=0.25
    ),
}
