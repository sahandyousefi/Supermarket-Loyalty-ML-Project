r"""Gaussian copula synthesiser for mixed numeric and categorical data.

Every column is mapped to a standard normal score. A numeric column uses
its empirical distribution function, :math:`z = \Phi^{-1}(\hat F(x))` with
ranks divided by :math:`n + 1`. A categorical column is laid out on
:math:`(0, 1)` in its declared level order, each level taking an interval
as wide as its frequency, and each value is drawn uniformly inside its
interval before the normal quantile is taken. Missing numeric values get a
separate indicator column, so missingness keeps its dependence on the
other attributes. The correlation matrix of the scores defines the copula.
New rows are drawn from :math:`N(0, R)` and mapped back through the
empirical quantile functions and the level intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from loyalty_spend.arrays import FloatArray, ObjectArray


def nearest_correlation(r: FloatArray, floor: float = 1e-6) -> FloatArray:
    """Clip eigenvalues at ``floor`` and rescale to a unit diagonal."""
    w, v = np.linalg.eigh((r + r.T) / 2)
    fixed = (v * np.maximum(w, floor)) @ v.T
    d = np.sqrt(np.diag(fixed))
    return np.asarray(fixed / np.outer(d, d), float)


@dataclass
class _Numeric:
    sorted_values: FloatArray
    integer: bool
    lower: float
    upper: float


@dataclass
class _Categorical:
    levels: list[str]
    edges: FloatArray


@dataclass
class GaussianCopulaSynthesizer:
    shrinkage: float = 0.0
    seed: int = 0
    columns_: list[str] = field(default_factory=list)
    numeric_: dict[str, _Numeric] = field(default_factory=dict)
    categorical_: dict[str, _Categorical] = field(default_factory=dict)
    latent_: list[str] = field(default_factory=list)
    corr_: FloatArray | None = None
    dtypes_: dict[str, object] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> GaussianCopulaSynthesizer:
        rng = np.random.default_rng(self.seed)
        self.columns_ = list(frame.columns)
        self.dtypes_ = dict(frame.dtypes)
        scores: dict[str, FloatArray] = {}
        for col in self.columns_:
            s = frame[col]
            if pd.api.types.is_numeric_dtype(s) and not isinstance(
                s.dtype, pd.CategoricalDtype
            ):
                x = s.to_numpy(float)
                miss = np.isnan(x)
                obs = x[~miss]
                self.numeric_[col] = _Numeric(
                    np.sort(obs),
                    bool(np.all(obs == np.round(obs))),
                    float(obs.min()),
                    float(obs.max()),
                )
                z = np.empty(len(x))
                ranks = stats.rankdata(obs, method="average")
                z[~miss] = stats.norm.ppf(ranks / (len(obs) + 1))
                # Missing values take a random score so the column stays usable.
                z[miss] = rng.standard_normal(miss.sum())
                scores[col] = z
                if miss.any():
                    scores[f"{col}__missing"] = self._encode_categorical(
                        f"{col}__missing",
                        pd.Series(np.where(miss, "yes", "no")),
                        ["no", "yes"],
                        rng,
                    )
            else:
                cats = s.astype("category")
                levels = [str(c) for c in cats.cat.categories]
                labels = cats.astype(str).where(cats.notna(), "__nan__")
                if (labels == "__nan__").any():
                    levels.append("__nan__")
                scores[col] = self._encode_categorical(col, labels, levels, rng)
        self.latent_ = list(scores)
        latent = np.column_stack([scores[c] for c in self.latent_])
        r = np.atleast_2d(np.corrcoef(latent, rowvar=False))
        r = (1 - self.shrinkage) * r + self.shrinkage * np.eye(len(r))
        self.corr_ = nearest_correlation(r)
        return self

    def _encode_categorical(
        self,
        col: str,
        labels: pd.Series,
        levels: list[str],
        rng: np.random.Generator,
    ) -> FloatArray:
        freq = labels.value_counts(normalize=True).reindex(levels, fill_value=0.0)
        edges = np.concatenate([[0.0], np.cumsum(freq.to_numpy())])
        edges[-1] = 1.0
        self.categorical_[col] = _Categorical(levels, edges)
        pos = labels.map({lv: i for i, lv in enumerate(levels)}).to_numpy(int)
        u = edges[pos] + rng.uniform(size=len(pos)) * (edges[pos + 1] - edges[pos])
        return np.asarray(stats.norm.ppf(np.clip(u, 1e-9, 1 - 1e-9)), float)

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if self.corr_ is None:
            raise RuntimeError("call fit first")
        rng = np.random.default_rng(self.seed + 1 if seed is None else seed)
        z = rng.multivariate_normal(np.zeros(len(self.latent_)), self.corr_, size=n)
        u = pd.DataFrame(stats.norm.cdf(z), columns=self.latent_)
        out: dict[str, object] = {}
        for col in self.columns_:
            if col in self.numeric_:
                spec = self.numeric_[col]
                m = len(spec.sorted_values)
                # Linear interpolation of the empirical quantile function.
                grid = np.arange(1, m + 1) / (m + 1)
                x = np.interp(u[col], grid, spec.sorted_values)
                if spec.integer:
                    x = np.round(x)
                missing_col = f"{col}__missing"
                if missing_col in self.categorical_:
                    flag = self._decode(missing_col, u[missing_col].to_numpy())
                    x = np.where(flag == "yes", np.nan, x)
                out[col] = x
            else:
                labels = self._decode(col, u[col].to_numpy())
                values = pd.Series(labels).replace("__nan__", np.nan)
                spec_levels = [
                    lv for lv in self.categorical_[col].levels if lv != "__nan__"
                ]
                dtype = self.dtypes_.get(col)
                if isinstance(dtype, pd.CategoricalDtype):
                    out[col] = values.astype(object).astype(dtype)
                else:
                    out[col] = pd.Categorical(values, categories=spec_levels)
        return pd.DataFrame(out, columns=self.columns_)

    def _decode(self, col: str, u: FloatArray) -> ObjectArray:
        spec = self.categorical_[col]
        idx = np.searchsorted(spec.edges, u, side="right") - 1
        idx = np.clip(idx, 0, len(spec.levels) - 1)
        return np.asarray(np.asarray(spec.levels, dtype=object)[idx], dtype=object)


@dataclass
class StratifiedCopulaSynthesizer:
    """One copula per level of a categorical attribute.

    A single Gaussian copula captures monotone dependence only. Spend is not
    monotone in the tenure band (members of one to three years spend least),
    so a copula over all members keeps the marginals but loses most of the
    relation. Fitting a copula within each level restores it; levels with
    fewer than ``min_size`` members share one copula that keeps the level as
    a column.
    """

    by: str
    min_size: int = 30
    shrinkage: float = 0.0
    seed: int = 0
    shares_: pd.Series | None = None
    models_: dict[str, GaussianCopulaSynthesizer] = field(default_factory=dict)
    pooled_: GaussianCopulaSynthesizer | None = None
    small_: list[str] = field(default_factory=list)
    dtypes_: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> StratifiedCopulaSynthesizer:
        self.dtypes_ = frame.dtypes
        labels = frame[self.by].astype(str)
        counts = labels.value_counts()
        self.shares_ = counts / counts.sum()
        self.small_ = [lv for lv, n in counts.items() if n < self.min_size]
        for i, (level, n) in enumerate(counts.items()):
            if n >= self.min_size:
                part = frame.loc[labels == level].drop(columns=self.by)
                self.models_[str(level)] = GaussianCopulaSynthesizer(
                    self.shrinkage, self.seed + i
                ).fit(part)
        if self.small_:
            rest = frame.loc[labels.isin(self.small_)]
            self.pooled_ = GaussianCopulaSynthesizer(self.shrinkage, self.seed).fit(
                rest
            )
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if self.shares_ is None or self.dtypes_ is None:
            raise RuntimeError("call fit first")
        rng = np.random.default_rng(self.seed + 1 if seed is None else seed)
        counts = rng.multinomial(n, self.shares_.to_numpy())
        parts = []
        pooled_n = 0
        for level, k in zip(self.shares_.index, counts, strict=True):
            if k == 0:
                continue
            if level in self.models_:
                part = self.models_[level].sample(k, int(rng.integers(2**31)))
                part[self.by] = level
                parts.append(part)
            else:
                pooled_n += k
        if pooled_n and self.pooled_ is not None:
            parts.append(self.pooled_.sample(pooled_n, int(rng.integers(2**31))))
        out = pd.concat(parts, ignore_index=True)
        out = out.sample(frac=1.0, random_state=int(rng.integers(2**31)))
        out = out.reset_index(drop=True)[list(self.dtypes_.index)]
        for col, dtype in self.dtypes_.items():
            if isinstance(dtype, pd.CategoricalDtype):
                out[col] = out[col].astype(object).astype(dtype)
        return out
