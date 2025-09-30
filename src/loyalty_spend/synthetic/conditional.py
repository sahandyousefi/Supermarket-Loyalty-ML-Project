"""Synthesiser with amounts conditional on the categories.

Fidelity checks on the stratified copula show that first-month spend and
basket size are almost determined by the joining month (correlation ratio
above 0.99) and spend by the tenure band. A Gaussian copula cannot hold
such relations for a nominal attribute with twelve levels, whatever order
the levels are given. Here the categories are generated first, by a
stratified copula, and every amount is the sum of an additive main-effects
fit on the categories and a residual. The residuals of all amounts, with
their missingness, come from a second Gaussian copula, so dependence
between amounts that the categories do not explain is kept as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from loyalty_spend.synthetic.copula import (
    GaussianCopulaSynthesizer,
    StratifiedCopulaSynthesizer,
)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not isinstance(
        s.dtype, pd.CategoricalDtype
    )


@dataclass
class _Amount:
    model: Ridge
    integer: bool
    lower: float
    upper: float


@dataclass
class ConditionalCopulaSynthesizer:
    by: str | None = None
    alpha: float = 1e-3
    seed: int = 0
    columns_: list[str] = field(default_factory=list)
    categorical_: list[str] = field(default_factory=list)
    amounts_: dict[str, _Amount] = field(default_factory=dict)
    design_columns_: list[str] = field(default_factory=list)
    categories_model_: (
        GaussianCopulaSynthesizer | StratifiedCopulaSynthesizer | None
    ) = None
    residual_model_: GaussianCopulaSynthesizer | None = None

    def _design(self, frame: pd.DataFrame) -> pd.DataFrame:
        dummies = pd.get_dummies(frame[self.categorical_].astype(object), dtype=float)
        return dummies.reindex(columns=self.design_columns_, fill_value=0.0)

    def fit(self, frame: pd.DataFrame) -> ConditionalCopulaSynthesizer:
        self.columns_ = list(frame.columns)
        self.categorical_ = [c for c in frame.columns if not _is_numeric(frame[c])]
        amounts = [c for c in frame.columns if _is_numeric(frame[c])]
        cats = frame[self.categorical_]
        self.categories_model_ = (
            StratifiedCopulaSynthesizer(self.by, seed=self.seed).fit(cats)
            if self.by
            else GaussianCopulaSynthesizer(seed=self.seed).fit(cats)
        )
        self.design_columns_ = list(
            pd.get_dummies(cats.astype(object), dtype=float).columns
        )
        D = self._design(frame)
        residuals = {}
        for col in amounts:
            y = frame[col].to_numpy(float)
            ok = ~np.isnan(y)
            model = Ridge(alpha=self.alpha).fit(D[ok], y[ok])
            r = np.full(len(y), np.nan)
            r[ok] = y[ok] - model.predict(D[ok])
            residuals[col] = r
            obs = y[ok]
            self.amounts_[col] = _Amount(
                model,
                bool(np.all(obs == np.round(obs))),
                float(obs.min()),
                float(obs.max()),
            )
        self.residual_model_ = GaussianCopulaSynthesizer(seed=self.seed + 7).fit(
            pd.DataFrame(residuals)
        )
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if self.categories_model_ is None or self.residual_model_ is None:
            raise RuntimeError("call fit first")
        rng = np.random.default_rng(self.seed + 1 if seed is None else seed)
        cats = self.categories_model_.sample(n, int(rng.integers(2**31)))
        resid = self.residual_model_.sample(n, int(rng.integers(2**31)))
        D = self._design(cats)
        out = cats.copy()
        for col, spec in self.amounts_.items():
            x = spec.model.predict(D) + resid[col].to_numpy(float)
            x = np.clip(x, spec.lower, spec.upper)
            if spec.integer:
                x = np.round(x)
            out[col] = x
        return out[self.columns_]
