r"""Gaussian process regression on mixed amounts and categories.

Amounts are standardised and categories one-hot encoded by the shared
preprocessing. The covariance is the sum of three parts,

.. math::

    k(x, x') = \sigma_0^2 + x^\top x' \;+\; \sigma_f^2
    \exp\Big(-\tfrac12 \sum_j (x_j - x'_j)^2 / \ell_j^2\Big)
    \;+\; \sigma_n^2 \mathbb{1}[x = x'].

The linear part carries the additive main effects that OLS already
captures; the RBF part with one length scale per column (automatic
relevance determination) adds smooth interactions, and on a dummy column
it acts as an exchangeable kernel for that category, since every pair of
different levels is at the same distance. Long fitted length scales mark
columns the interactions do not need. The white noise part is the
member-level noise and is what the predictive interval adds to the
uncertainty of the mean.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    Kernel,
    WhiteKernel,
)
from sklearn.pipeline import Pipeline

from loyalty_spend.features.preprocess import make_preprocessor
from loyalty_spend.models.registry import register


def mixed_kernel(n_columns: int) -> Kernel:
    return (
        DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
        + ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(n_columns), length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(0.1, (1e-6, 10.0))
    )


class MixedGaussianProcess(GaussianProcessRegressor):
    """GP whose ARD kernel is sized to the encoded columns at fit time."""

    def __init__(self, n_restarts_optimizer: int = 2, random_state: int = 0):
        super().__init__(
            kernel=None,
            normalize_y=True,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )

    def fit(self, X, y):  # type: ignore[no-untyped-def]
        self.kernel = mixed_kernel(np.asarray(X).shape[1])
        return super().fit(X, y)


@register("gp", family="kernel")
def gaussian_process(n_restarts_optimizer: int = 2, random_state: int = 0) -> Pipeline:
    """GP with a linear plus ARD RBF kernel on the one-hot encoding."""
    return Pipeline(
        [
            ("prep", make_preprocessor("onehot")),
            ("model", MixedGaussianProcess(n_restarts_optimizer, random_state)),
        ]
    )
