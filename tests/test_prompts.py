import pytest

from loyalty_spend.llm.prompts import available_prompts, load_prompt


def test_every_prompt_loads_and_renders():
    assert available_prompts() == ["answer", "narrative", "reasons"]
    for name in available_prompts():
        prompt = load_prompt(name)
        assert prompt.version >= 1 and prompt.system
        system, user = prompt.render(**{f: f"<{f}>" for f in prompt.fields})
        assert all(f"<{f}>" in user for f in prompt.fields)
        assert "JSON" in system


def test_missing_fields_and_unknown_prompt():
    with pytest.raises(KeyError, match="question"):
        load_prompt("answer").render(passages="x")
    with pytest.raises(KeyError):
        load_prompt("nope")
