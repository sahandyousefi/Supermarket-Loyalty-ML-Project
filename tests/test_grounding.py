import pytest

from loyalty_spend.llm.facts import Fact
from loyalty_spend.llm.grounding import check_numbers, extract_numbers

FACTS = [
    Fact("a", "Tenure band explains 88.6% of the variance in annual spend."),
    Fact("b", "The expected reward cost of the current schedule is 1218.42."),
    Fact(
        "c", "The estimated effect is -0.015, with a 95% interval from -0.056 to 0.026."
    ),
]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Costs 1,218.42 in total.", ["1218.42"]),
        ("Explains 88.6% and 12 more.", ["88.6%", "12"]),
        ("An effect of −0.015.", ["-0.015"]),
        ("Model ft_transformer2 and v1.3 are names.", []),
    ],
)
def test_number_extraction(text, expected):
    assert extract_numbers(text) == expected


def test_faithful_text_is_grounded():
    text = "Tenure explains 88.6% of the variance; rewards should cost about 1,218.42."
    assert check_numbers(text, FACTS).grounded


def test_invented_or_recomputed_numbers_are_flagged():
    text = "Tenure explains 89% of variance, 88.6 points, and costs rise 0.4% to 1223."
    result = check_numbers(text, FACTS)
    assert result.unsupported == ["89%", "88.6", "0.4%", "1223"]
    assert check_numbers("The 10+ band.", FACTS, allowed={"10"}).grounded
