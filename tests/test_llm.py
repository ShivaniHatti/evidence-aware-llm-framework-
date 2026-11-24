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


def test_successful_llm_call(monkeypatch):
    """Test that successful LLM API call sets used_llm=True and returns parsed data."""
    client = LLMClient(api_key="sk-test-valid", mode="api", model="gpt-4o-mini")

    def mock_api_call(text):
        from src.llm.client import LLMResponse
        return LLMResponse(
            success=True,
            used_llm=True,
            label="phishing",
            confidence=0.92,
            reasoning="Message contains urgent verification request",
            explanation="The message asks for immediate account verification, which is a common phishing tactic.",
            probabilities={"phishing": 0.92, "legitimate": 0.06, "uncertain": 0.02},
        )

    monkeypatch.setattr(client, "_call_api", mock_api_call)
    resp = client.classify_and_explain("Verify your account immediately or it will be suspended.")
    
    assert resp.used_llm is True
    assert resp.success is True
    assert resp.label == "phishing"
    assert resp.confidence == 0.92
    assert resp.reasoning is not None
    assert resp.explanation is not None
    assert resp.error is None


def test_llm_mode_auto_with_key(monkeypatch):
    """Test that LLM_MODE=auto uses API when key is available."""
    client = LLMClient(api_key="sk-test-valid", mode="auto", model="gpt-4o-mini")

    def mock_api_call(text):
        from src.llm.client import LLMResponse
        return LLMResponse(
            success=True,
            used_llm=True,
            label="legitimate",
            confidence=0.85,
            reasoning="Normal order confirmation",
            explanation="This is a standard order confirmation message.",
            probabilities={"phishing": 0.10, "legitimate": 0.85, "uncertain": 0.05},
        )

    monkeypatch.setattr(client, "_call_api", mock_api_call)
    resp = client.classify_and_explain("Thank you for your order. Order #12345 confirmed.")
    
    assert resp.used_llm is True
    assert resp.label == "legitimate"


def test_llm_mode_api_without_key():
    """Test that LLM_MODE=api falls back when no key is present."""
    client = LLMClient(api_key="", mode="api")
    resp = client.classify_and_explain("Test message")
    
    assert resp.used_llm is False
    assert resp.error is not None
    assert "API_KEY" in resp.error or "key" in resp.error.lower()
