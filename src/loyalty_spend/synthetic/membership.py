"""Distance-based membership inference against synthetic data.

The attacker holds the released synthetic rows and a set of candidate
records, half of them members of the data the synthesiser was fitted on
and half real records it never saw. Each candidate is scored by minus its
Gower distance to the nearest synthetic row: if the synthesiser memorised
its training members, they lie closer to the release than other people
do. The area under the ROC curve of this score is 0.5 when membership
cannot be told apart, and the membership advantage
:math:`\\max_t (\\mathrm{TPR}_t - \\mathrm{FPR}_t)` bounds what the best
threshold gains over guessing (Yeom et al., 2018). A bootstrap over
candidates gives an interval for the AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from loyalty_spend.synthetic.privacy import gower_cross


def membership_inference(
    members: pd.DataFrame,
    non_members: pd.DataFrame,
    synth: pd.DataFrame,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    n = min(len(members), len(non_members))
    rng = np.random.default_rng(seed)
    cand = pd.concat(
        [
            members.iloc[rng.choice(len(members), n, replace=False)],
            non_members.iloc[rng.choice(len(non_members), n, replace=False)],
        ],
        ignore_index=True,
    )
    label = np.r_[np.ones(n), np.zeros(n)]
    score = -np.nanmin(gower_cross(cand, synth), axis=1)
    auc = float(roc_auc_score(label, score))
    fpr, tpr, _ = roc_curve(label, score)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, 2 * n, 2 * n)
        if 0 < label[idx].sum() < 2 * n:
            boot.append(roc_auc_score(label[idx], score[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "auc": auc,
        "auc_lower": float(lo),
        "auc_upper": float(hi),
        "advantage": float(np.max(tpr - fpr)),
        "n_candidates": float(2 * n),
    }
