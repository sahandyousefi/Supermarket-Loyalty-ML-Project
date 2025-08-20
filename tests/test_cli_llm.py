import json

from loyalty_spend.cli import main


def test_narrate_offline_prints_the_facts(capsys, monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    assert main(["narrate"]) == 0
    out = capsys.readouterr().out
    assert "Tenure band explains" in out


def test_narrate_with_a_configured_model(tmp_path, capsys, monkeypatch):
    from loyalty_spend.llm import client as client_module
    from loyalty_spend.llm.client import MockLLM

    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({"design": {"cost_to_spend": 0.0379}}))
    reply = json.dumps(
        {"headline": "Rewards cost 3.79% of expected spend.", "points": []}
    )
    monkeypatch.setattr(
        client_module, "client_from_env", lambda: MockLLM(lambda s, u: reply)
    )
    assert main(["narrate", "--summary", str(summary)]) == 0
    out = capsys.readouterr().out
    assert "Rewards cost 3.79%" in out and "prompt v1" in out
