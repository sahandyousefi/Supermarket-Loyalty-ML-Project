import numpy as np
import pytest
from scipy.stats import norm

from loyalty_spend.experiment.sequential import boundaries, design_properties

FIVE = [0.2, 0.4, 0.6, 0.8, 1.0]


def test_single_look_is_the_fixed_design():
    assert boundaries([1.0])[0] == pytest.approx(1.95996, abs=1e-3)


def test_obrien_fleming_boundaries_match_published_values():
    # gsDesign spends 0.025 on each side with the one-sided O'Brien-Fleming
    # function; five equally spaced looks give these bounds (sfLDOF).
    z = norm.ppf(1 - 0.025 / 2)

    def each_side(t):
        return 2 * (2 - 2 * norm.cdf(z / np.sqrt(t)))

    expected = [4.877, 3.357, 2.680, 2.290, 2.031]
    assert np.allclose(boundaries(FIVE, spending=each_side), expected, atol=0.005)
    # The two-sided version spends more at the first look.
    assert boundaries(FIVE)[0] == pytest.approx(norm.ppf(1 - 0.025) / np.sqrt(0.2))


def test_pocock_type_boundaries_are_nearly_flat():
    c = boundaries(FIVE, spending="pocock")
    assert c.max() - c.min() < 0.1 and 2.3 < c.mean() < 2.5


def test_type_one_error_by_simulation():
    c = boundaries(FIVE)
    rng = np.random.default_rng(0)
    steps = rng.standard_normal((200000, 5)) * np.sqrt(0.2)
    z = np.cumsum(steps, axis=1) / np.sqrt(FIVE)
    rejected = (np.abs(z) >= c).any(axis=1).mean()
    assert rejected == pytest.approx(0.05, abs=0.002)
    props = design_properties(FIVE, c, drift=0.0)
    assert props.power == pytest.approx(0.05, abs=1e-3)


def test_power_and_expected_size_under_the_alternative():
    c = boundaries(FIVE)
    drift = 1.95996 + 0.84162  # 80% power for the fixed design
    props = design_properties(FIVE, c, drift)
    assert 0.77 < props.power < 0.80
    assert props.expected_fraction < 0.8
    with pytest.raises(ValueError):
        boundaries([0.5, 0.4, 1.0])
