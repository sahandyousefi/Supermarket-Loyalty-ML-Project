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


class _Response:
    def __init__(self, payload):
        import json

        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _reply(text):
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        "model": "m",
    }


def test_http_client_sends_the_protocol_and_retries(monkeypatch):
    import json
    import urllib.error
    import urllib.request

    import pytest

    from loyalty_spend.llm.client import HTTPChatClient

    sent, attempts = [], {"n": 0}

    def fake_urlopen(request, timeout):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "slow down", {}, None)
        sent.append((request.full_url, json.loads(request.data), request.headers))
        return _Response(_reply("hello"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = HTTPChatClient("http://llm.local/v1/", "m", api_key="k", backoff=0.0)
    out = client.complete("sys", "user", json_mode=True)
    assert out.text == "hello" and out.prompt_tokens == 7 and attempts["n"] == 2
    url, payload, headers = sent[0]
    assert url == "http://llm.local/v1/chat/completions"
    assert payload["response_format"] == {"type": "json_object"}
    assert headers["Authorization"] == "Bearer k"

    def bad_request(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", bad_request)
    with pytest.raises(urllib.error.HTTPError):
        client.complete("sys", "user")


def test_http_client_needs_configuration(monkeypatch):
    import pytest

    from loyalty_spend.llm.client import HTTPChatClient

    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(ValueError):
        HTTPChatClient()
