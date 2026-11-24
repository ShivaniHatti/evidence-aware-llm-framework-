from src.llm.client import LLMClient


def test_local_fallback_when_no_key():
    client = LLMClient(api_key="", mode="auto")
    resp = client.classify_and_explain("Please verify your account immediately.")
    assert resp.used_llm is False
    assert resp.success is True
    assert resp.error is not None or resp.reasoning


def test_force_local_mode():
    client = LLMClient(api_key="sk-fake-key-for-test", mode="local")
    resp = client.classify_and_explain("Hello")
    assert resp.used_llm is False
    assert "local" in resp.reasoning.lower() or resp.label is None


def test_api_failure_falls_back(monkeypatch):
    client = LLMClient(api_key="sk-test-invalid", mode="auto", timeout=1)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated API failure")

    monkeypatch.setattr(client, "_call_api", boom)
    resp = client.classify_and_explain("Verify your account now.")
    assert resp.used_llm is False
    assert resp.error is not None
    assert "failure" in resp.error.lower()
