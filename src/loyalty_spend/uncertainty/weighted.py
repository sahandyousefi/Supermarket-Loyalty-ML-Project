r"""Conformal prediction under covariate shift.

When the customers to be scored differ from the training customers in their
features but not in how spend depends on the features, exchangeability
fails and split conformal intervals lose their coverage. Tibshirani et al.
(2019) restore it by weighting the calibration scores with the likelihood
ratio :math:`w(x) = dP_{\text{new}}(x) / dP_{\text{train}}(x)`. For a new
point :math:`x`, the interval uses the :math:`1-\alpha` quantile of

.. math::

    \sum_{i=1}^n p_i(x)\,\delta_{s_i} + p_{n+1}(x)\,\delta_{+\infty},
    \qquad p_i(x) = \frac{w(X_i)}{\sum_{j} w(X_j) + w(x)},

which reduces to the usual split conformal quantile when all weights are
equal.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def weighted_conformal_quantile(
    scores: ArrayLike,
    calibration_weights: ArrayLike,
    test_weights: ArrayLike,
    alpha: float,
) -> np.ndarray:
    """Weighted ``1 - alpha`` quantile for each test weight.

    Returns infinity for a test point whose own weight is so large that the
    calibration mass cannot reach ``1 - alpha``.
    """
    s = np.asarray(scores, float)
    w = np.asarray(calibration_weights, float)
    w_test = np.atleast_1d(np.asarray(test_weights, float))
    if s.shape != w.shape:
        raise ValueError("need one weight per calibration score")
    if np.any(w < 0) or np.any(w_test < 0):
        raise ValueError("weights must be non-negative")
    order = np.argsort(s)
    s, w = s[order], w[order]
    cum = np.cumsum(w)
    total = cum[-1] + w_test
    # First sorted score whose normalised cumulative mass reaches 1 - alpha.
    target = (1 - alpha) * total[:, None]
    idx = np.argmax(cum[None, :] >= target - 1e-12, axis=1)
    reached = cum[-1] >= (1 - alpha) * total - 1e-12
    return np.where(reached, s[idx], np.inf)
