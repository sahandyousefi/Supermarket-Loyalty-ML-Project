from loyalty_spend.llm.citations import split_sentences, verify_citations
from loyalty_spend.llm.corpus import Chunk

PASSAGES = [
    Chunk(
        "a",
        "m.md",
        "Intervals",
        "Split conformal intervals cover 90% of new customers on average.",
    ),
    Chunk(
        "b", "m.md", "Costs", "Expected reward cost is 1218.42 for the scoring file."
    ),
]


def test_supported_and_unsupported_sentences():
    answer = (
        "Split conformal intervals cover 90% of new customers [1]. "
        "Expected reward cost is 1300.00 for the file [2]. "
        "Deep ensembles are always better."
    )
    checks = verify_citations(answer, PASSAGES)
    assert [c.supported for c in checks] == [True, False, False]
    assert checks[1].missing_numbers == ("1300.00",)
    assert checks[2].cited == ()


def test_wrong_passage_has_low_overlap():
    check = verify_citations(
        "Split conformal intervals cover 90% of new customers [2].", PASSAGES
    )[0]
    assert check.overlap < 0.5 and not check.supported


def test_sentence_split_keeps_decimals():
    assert split_sentences("Cost is 3.79% of spend. It is small.") == [
        "Cost is 3.79% of spend.",
        "It is small.",
    ]
