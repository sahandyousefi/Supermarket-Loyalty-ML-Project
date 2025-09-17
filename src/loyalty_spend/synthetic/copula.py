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


def nearest_correlation(r: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Clip eigenvalues at ``floor`` and rescale to a unit diagonal."""
    w, v = np.linalg.eigh((r + r.T) / 2)
    fixed = (v * np.maximum(w, floor)) @ v.T
    d = np.sqrt(np.diag(fixed))
    return fixed / np.outer(d, d)


@dataclass
class _Numeric:
    sorted_values: np.ndarray
    integer: bool
    lower: float
    upper: float


@dataclass
class _Categorical:
    levels: list[str]
    edges: np.ndarray


@dataclass
class GaussianCopulaSynthesizer:
    shrinkage: float = 0.0
    seed: int = 0
    columns_: list[str] = field(default_factory=list)
    numeric_: dict[str, _Numeric] = field(default_factory=dict)
    categorical_: dict[str, _Categorical] = field(default_factory=dict)
    latent_: list[str] = field(default_factory=list)
    corr_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> GaussianCopulaSynthesizer:
        rng = np.random.default_rng(self.seed)
        self.columns_ = list(frame.columns)
        scores: dict[str, np.ndarray] = {}
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
        z = np.column_stack([scores[c] for c in self.latent_])
        r = np.atleast_2d(np.corrcoef(z, rowvar=False))
        r = (1 - self.shrinkage) * r + self.shrinkage * np.eye(len(r))
        self.corr_ = nearest_correlation(r)
        return self

    def _encode_categorical(
        self,
        col: str,
        labels: pd.Series,
        levels: list[str],
        rng: np.random.Generator,
    ) -> np.ndarray:
        freq = labels.value_counts(normalize=True).reindex(levels, fill_value=0.0)
        edges = np.concatenate([[0.0], np.cumsum(freq.to_numpy())])
        edges[-1] = 1.0
        self.categorical_[col] = _Categorical(levels, edges)
        pos = labels.map({lv: i for i, lv in enumerate(levels)}).to_numpy(int)
        u = edges[pos] + rng.uniform(size=len(pos)) * (edges[pos + 1] - edges[pos])
        return stats.norm.ppf(np.clip(u, 1e-9, 1 - 1e-9))

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
                out[col] = pd.Categorical(values, categories=spec_levels)
        return pd.DataFrame(out, columns=self.columns_)

    def _decode(self, col: str, u: np.ndarray) -> np.ndarray:
        spec = self.categorical_[col]
        idx = np.searchsorted(spec.edges, u, side="right") - 1
        idx = np.clip(idx, 0, len(spec.levels) - 1)
        return np.asarray(spec.levels, dtype=object)[idx]
