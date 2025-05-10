import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from loyalty_spend.data.cleaning import (
    PLACEHOLDERS,
    coerce_numeric,
    normalize_category,
)
from loyalty_spend.data.schema import PROMOTION, REGIONS

amounts = st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False)
placeholders = st.sampled_from(sorted(PLACEHOLDERS))
padding = st.sampled_from(["", " ", "  ", "\t"])


@settings(max_examples=150, deadline=None)
@given(
    st.lists(
        st.one_of(
            amounts.map(lambda v: ("num", v)), placeholders.map(lambda p: ("ph", p))
        ),
        min_size=1,
        max_size=40,
    ),
    padding,
)
def test_numbers_survive_and_placeholders_become_missing(items, pad):
    text = [pad + (repr(v) if kind == "num" else v) + pad for kind, v in items]
    values, n_placeholder, n_invalid = coerce_numeric(pd.Series(text, dtype=object))
    expected = [v if kind == "num" else np.nan for kind, v in items]
    np.testing.assert_allclose(values.to_numpy(), expected, rtol=1e-12)
    assert n_placeholder == sum(kind == "ph" for kind, _ in items)
    assert n_invalid == 0


@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.text(alphabet="abcxyz", min_size=1, max_size=5), min_size=1, max_size=20
    )
)
def test_unparseable_text_is_counted_as_invalid(words):
    values, _, n_invalid = coerce_numeric(pd.Series(words, dtype=object))
    assert values.isna().all()
    assert n_invalid == sum(w.strip().lower() not in PLACEHOLDERS for w in words)


def _variants(level: str):
    return st.sampled_from(
        [level, level.upper(), level.lower(), level.title()]
    ).flatmap(lambda v: padding.map(lambda p: p + v + p))


@settings(max_examples=150, deadline=None)
@given(
    st.lists(st.sampled_from(REGIONS + PROMOTION), min_size=1, max_size=30).flatmap(
        lambda lv: st.tuples(st.just(lv), st.tuples(*[_variants(x) for x in lv]))
    )
)
def test_case_and_space_variants_map_to_the_canonical_level(case):
    levels, variants = case
    levels = list(levels)
    allowed = REGIONS + PROMOTION
    cleaned, changed = normalize_category(
        pd.Series(list(variants), dtype=object), allowed
    )
    assert cleaned.tolist() == levels
    # Surrounding whitespace is stripped silently; only relabelled values count.
    assert changed == sum(v.strip() != lv for v, lv in zip(variants, levels))


@settings(max_examples=100, deadline=None)
@given(st.lists(st.sampled_from(REGIONS), min_size=1, max_size=30))
def test_normalization_is_idempotent(levels):
    once, _ = normalize_category(pd.Series(levels, dtype=object), REGIONS)
    twice, changed = normalize_category(once, REGIONS)
    assert once.tolist() == twice.tolist()
    assert changed == 0
