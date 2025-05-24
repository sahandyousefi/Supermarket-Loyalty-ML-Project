"""Multivariate drift with a domain classifier.

Univariate checks miss shifts in the joint distribution, for example a
change in which regions the long-tenure members come from. A classifier
trained to tell the reference rows from the new rows detects any such shift:
if the two samples come from the same distribution, its out-of-fold AUC is
0.5 up to sampling noise. The fitted probabilities are also density-ratio
estimates, which the weighted conformal intervals reuse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from loyalty_spend.data.schema import FEATURES
from loyalty_spend.features.preprocess import make_preprocessor


@dataclass
class DomainClassifierResult:
    auc: float
    p_value: float
    scores: np.ndarray
    is_current: np.ndarray

    def density_ratio(self, clip: float = 20.0) -> np.ndarray:
        """Estimated ``p_current(x) / p_reference(x)`` for every row.

        The odds are rescaled by the sample sizes and clipped, because a
        handful of extreme weights would otherwise dominate any weighted
        estimate.
        """
        p = np.clip(self.scores, 1e-6, 1 - 1e-6)
        n_cur = self.is_current.sum()
        n_ref = len(self.is_current) - n_cur
        ratio = p / (1 - p) * n_ref / max(n_cur, 1)
        return np.clip(ratio, 1 / clip, clip)


def make_domain_classifier(seed: int = 0) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor("ordinal")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            ),
        ]
    )


def domain_classifier_test(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 0,
) -> DomainClassifierResult:
    """Out-of-fold AUC of a classifier separating ``current`` from ``reference``.

    The p-value is that of the Mann-Whitney U test on the out-of-fold
    scores, whose statistic is the AUC times ``n_ref * n_cur``.
    """
    features = list(FEATURES if features is None else features)
    X = pd.concat([reference[features], current[features]], ignore_index=True)
    is_current = np.r_[np.zeros(len(reference)), np.ones(len(current))].astype(int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_predict(
        make_domain_classifier(seed), X, is_current, cv=cv, method="predict_proba"
    )[:, 1]
    auc = float(roc_auc_score(is_current, scores))
    test = stats.mannwhitneyu(
        scores[is_current == 1], scores[is_current == 0], alternative="greater"
    )
    return DomainClassifierResult(auc, float(test.pvalue), scores, is_current)
