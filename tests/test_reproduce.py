import pytest

from loyalty_spend.ops.reproduce import (
    Reference,
    compare,
    load_reference,
    lookup,
    write_reference,
)

SUMMARY = {
    "drivers": {"welch": {"f": 2506.25}, "lmg": [{"r2_share": 0.88}]},
    "promotion": {"estimates": [{"estimate": -0.99}, {"estimate": -0.017}]},
}


def test_lookup_follows_keys_and_positions():
    assert lookup(SUMMARY, "drivers.welch.f") == 2506.25
    assert lookup(SUMMARY, "promotion.estimates[1].estimate") == -0.017
    assert lookup(SUMMARY, "drivers.lmg[0].r2_share") == 0.88


def test_compare_flags_values_outside_their_tolerance():
    refs = [
        Reference("drivers.welch.f", 2506.0, 1e-3),
        Reference("promotion.estimates[1].estimate", -0.0171, 1e-3),
        Reference("drivers.missing", 1.0, 1e-3),
    ]
    table = compare(SUMMARY, refs).set_index("path")
    assert table.loc["drivers.welch.f", "ok"]
    assert not table.loc["promotion.estimates[1].estimate", "ok"]
    assert not table.loc["drivers.missing", "ok"]


def test_reference_round_trip(tmp_path):
    out = tmp_path / "ref.yaml"
    write_reference(SUMMARY, {"drivers.welch.f": 1e-6}, out, ["drivers"])
    refs = load_reference(out)
    assert refs[0].value == pytest.approx(2506.25) and refs[0].rel_tol == 1e-6
    assert compare(SUMMARY, refs)["ok"].all()
