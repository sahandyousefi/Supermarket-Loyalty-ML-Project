r"""Does the error variance differ between groups of members?

Three tests are reported for out-of-fold residuals grouped by an
attribute. Bartlett's test is most powerful under normal errors but
rejects for heavy tails alone. The Brown-Forsythe test (Levene's test on
absolute deviations from the group medians) keeps its level without
normality. The Koenker form of the Breusch-Pagan test regresses the
squared residuals on the group dummies and compares :math:`nR^2` with a
:math:`\chi^2` distribution with one degree of freedom per extra group,
which is robust to non-normal errors (Koenker, 1981). The variance of
every group is given with a :math:`\chi^2` interval, which assumes normal
errors within the group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats


def group_variances(
    residuals: ArrayLike, groups: ArrayLike, level: float = 0.95
) -> pd.DataFrame:
    e = np.asarray(residuals, float)
    g = np.asarray(groups).astype(str)
    tail = (1 - level) / 2
    rows = []
    for lv in sorted(set(g)):
        x = e[g == lv]
        n, var = len(x), float(x.var(ddof=1))
        rows.append(
            {
                "group": lv,
                "n": n,
                "sd": np.sqrt(var),
                "sd_low": np.sqrt((n - 1) * var / stats.chi2.ppf(1 - tail, n - 1)),
                "sd_high": np.sqrt((n - 1) * var / stats.chi2.ppf(tail, n - 1)),
            }
        )
    return pd.DataFrame(rows)


def breusch_pagan_koenker(residuals: ArrayLike, groups: ArrayLike) -> dict[str, float]:
    e2 = np.asarray(residuals, float) ** 2
    D = pd.get_dummies(pd.Series(np.asarray(groups).astype(str)), drop_first=True)
    X = np.column_stack([np.ones(len(e2)), D.to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, e2, rcond=None)
    fitted = X @ beta
    r2 = 1 - np.sum((e2 - fitted) ** 2) / np.sum((e2 - e2.mean()) ** 2)
    lm = len(e2) * r2
    df = X.shape[1] - 1
    return {"lm": float(lm), "df": float(df), "p_value": float(stats.chi2.sf(lm, df))}


def variance_tests(residuals: ArrayLike, groups: ArrayLike) -> pd.DataFrame:
    e = np.asarray(residuals, float)
    g = np.asarray(groups).astype(str)
    parts = [e[g == lv] for lv in sorted(set(g))]
    bartlett = stats.bartlett(*parts)
    brown = stats.levene(*parts, center="median")
    bp = breusch_pagan_koenker(e, g)
    return pd.DataFrame(
        [
            {
                "test": "Bartlett",
                "statistic": bartlett.statistic,
                "p_value": bartlett.pvalue,
            },
            {
                "test": "Brown-Forsythe",
                "statistic": brown.statistic,
                "p_value": brown.pvalue,
            },
            {
                "test": "Breusch-Pagan (Koenker)",
                "statistic": bp["lm"],
                "p_value": bp["p_value"],
            },
        ]
    )


def variance_profile(
    residuals: ArrayLike,
    frame: pd.DataFrame,
    group: str,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Residual SD of every group next to the within-group SD of other columns.

    If the error variance of a group follows the spread of an observed
    attribute within that group, the attribute is a candidate for modelling
    the variance; the second result gives the rank correlation across
    groups between the residual SD and the SD of each column.
    """
    data = frame[[group, *columns]].copy()
    data["residual"] = np.asarray(residuals, float)
    table = data.groupby(group, observed=True)[["residual", *columns]].std()
    table.columns = [f"sd_{c}" for c in table.columns]
    corr = pd.Series(
        {
            c: float(stats.spearmanr(table["sd_residual"], table[f"sd_{c}"]).statistic)
            for c in columns
        },
        name="spearman_with_residual_sd",
    )
    return table.reset_index(), corr
