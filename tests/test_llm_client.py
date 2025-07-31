from loyalty_spend.llm.client import MockLLM, approx_tokens


def test_mock_is_deterministic_and_records_calls():
    llm = MockLLM()
    a = llm.complete("system", "question")
    b = llm.complete("system", "question")
    c = llm.complete("system", "other question")
    assert a.text == b.text != c.text
    assert len(llm.calls) == 3
    assert a.prompt_tokens == approx_tokens("system") + approx_tokens("question")


def test_custom_responder():
    llm = MockLLM(lambda system, user: user.upper())
    assert llm.complete("s", "hello").text == "HELLO"
    assert approx_tokens("") == 1
