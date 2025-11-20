import numpy as np
import pytest

from loyalty_spend.robustness.heteroscedasticity import (
    breusch_pagan_koenker,
    group_variances,
    variance_tests,
)


def test_tests_keep_their_level_with_equal_variances():
    rng = np.random.default_rng(0)
    rejections = np.zeros(3)
    for _ in range(300):
        g = rng.choice(list("abcd"), 200)
        e = rng.standard_t(5, 200)
        rejections += variance_tests(e, g)["p_value"].to_numpy() < 0.05
    rates = dict(zip(["bartlett", "brown", "bp"], rejections / 300, strict=True))
    # Bartlett is not robust to heavy tails; the other two hold their level.
    assert rates["brown"] == pytest.approx(0.05, abs=0.035)
    assert rates["bp"] == pytest.approx(0.05, abs=0.035)
    assert rates["bartlett"] > rates["brown"]


def test_unequal_variance_is_detected():
    rng = np.random.default_rng(1)
    g = np.repeat(["a", "b"], 150)
    e = rng.normal(0, np.where(g == "a", 1.0, 1.6))
    assert (variance_tests(e, g)["p_value"] < 0.01).all()


def test_breusch_pagan_with_two_groups_is_a_scaled_t_test():
    rng = np.random.default_rng(2)
    g = np.repeat(["a", "b"], 100)
    e = rng.normal(0, np.where(g == "a", 1.0, 1.3))
    res = breusch_pagan_koenker(e, g)
    assert res["df"] == 1
    # n R^2 equals n r^2 for the point-biserial correlation of e^2 with g.
    r = np.corrcoef(e**2, g == "b")[0, 1]
    assert res["lm"] == pytest.approx(200 * r**2)


def test_group_sd_intervals_cover_the_truth():
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(300):
        table = group_variances(rng.normal(0, 2.0, 40), np.zeros(40))
        hits += table["sd_low"].item() <= 2.0 <= table["sd_high"].item()
    assert hits / 300 == pytest.approx(0.95, abs=0.035)


def test_variance_profile_finds_the_driving_attribute():
    import pandas as pd

    from loyalty_spend.robustness.heteroscedasticity import variance_profile

    rng = np.random.default_rng(4)
    g = np.repeat(np.arange(10), 100)
    scale = np.linspace(0.1, 1.0, 10)[g]
    frame = pd.DataFrame(
        {
            "g": g.astype(str),
            "driver": rng.normal(0, scale),
            "unrelated": rng.normal(0, 1, len(g)),
        }
    )
    resid = rng.normal(0, scale)
    table, corr = variance_profile(resid, frame, "g", ["driver", "unrelated"])
    assert len(table) == 10 and "sd_driver" in table
    assert corr["driver"] > 0.9 and abs(corr["unrelated"]) < 0.7
