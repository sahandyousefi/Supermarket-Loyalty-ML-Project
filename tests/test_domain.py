import numpy as np
import pytest

from loyalty_spend.monitoring.domain import domain_classifier_test


def test_same_distribution_gives_chance_auc(sim_df):
    result = domain_classifier_test(sim_df.iloc[:300], sim_df.iloc[300:])
    assert abs(result.auc - 0.5) < 0.08
    assert result.p_value > 0.01


def test_joint_shift_without_marginal_shift_is_detected(sim_df):
    # Swap tenure between regions: every marginal stays the same, the joint
    # distribution of region and tenure does not.
    ref = sim_df.iloc[:300].copy()
    cur = sim_df.iloc[300:].copy()
    cur = cur.sort_values("region").reset_index(drop=True)
    cur["loyalty_years"] = cur["loyalty_years"].sort_values().to_numpy()
    result = domain_classifier_test(ref, cur)
    assert result.auc > 0.65
    assert result.p_value < 1e-4


def test_density_ratio_is_positive_and_clipped(sim_df):
    result = domain_classifier_test(sim_df.iloc[:300], sim_df.iloc[300:])
    w = result.density_ratio(clip=5.0)
    assert np.all(w >= 0.2 - 1e-12) and np.all(w <= 5.0 + 1e-12)
    assert np.median(w) == pytest.approx(1.0, rel=0.3)
