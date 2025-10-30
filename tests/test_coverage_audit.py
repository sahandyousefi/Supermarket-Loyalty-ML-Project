import numpy as np
import pandas as pd
import pytest

from loyalty_spend.uncertainty.audit import (
    benjamini_hochberg,
    clopper_pearson,
    coverage_audit,
)
from loyalty_spend.uncertainty.conformal import Intervals


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(5, 20)
    assert lo == pytest.approx(0.0866, abs=1e-4)
    assert hi == pytest.approx(0.4910, abs=1e-4)
    assert clopper_pearson(0, 10)[0] == 0.0 and clopper_pearson(10, 10)[1] == 1.0


def test_benjamini_hochberg_matches_definition():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(benjamini_hochberg(p), [0.04, 0.16 / 3, 0.16 / 3, 0.2])


def test_audit_flags_the_undercovered_group():
    rng = np.random.default_rng(0)
    n = 3000
    group = rng.choice(["a", "b", "c"], n)
    sd = np.where(group == "c", 2.0, 1.0)
    y = rng.normal(0, sd)
    # One interval width for all: band c is too narrow for its spread.
    half = 1.645 * np.sqrt(np.mean(sd**2)) * 0.9
    iv = Intervals(np.full(n, -half), np.full(n, half))
    table = coverage_audit(iv, y, pd.DataFrame({"g": group}), ["g"]).set_index("level")
    assert table.loc["c", "flagged"] and table.loc["c", "ci_high"] < 0.9
    assert not table.loc["a", "flagged"] or table.loc["a", "coverage"] > 0.9
