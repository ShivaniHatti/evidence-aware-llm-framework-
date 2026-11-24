"""
Configurable LLM integration with graceful local fallback.

Never hardcodes credentials. Reads OPENAI_API_KEY from environment.
Clearly labels outputs as LLM-generated vs locally generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
import os
import re

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass
class LLMResponse:
    success: bool
    used_llm: bool
    label: Optional[str]
    confidence: Optional[float]
    reasoning: str
    explanation: str
    probabilities: Dict[str, float]
    error: Optional[str] = None
    raw: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "used_llm": self.used_llm,
            "label": self.label,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "explanation": self.explanation,
            "probabilities": self.probabilities,
            "error": self.error,
        }


class LLMClient:
    """OpenAI-compatible chat client with deterministic local fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        mode: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.mode = (mode or os.getenv("LLM_MODE", "auto")).lower()
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and self.mode != "local"

    def classify_and_explain(self, text: str) -> LLMResponse:
        """Attempt LLM call; on missing key or failure, return local fallback."""
        if self.mode == "local" or not self.api_key:
            return self._local_fallback(
                text,
                error=None
                if self.mode == "local"
                else "OPENAI_API_KEY not set; using local fallback.",
            )

        try:
            return self._call_api(text)
        except Exception as exc:  # noqa: BLE001 — intentional graceful degradation
            return self._local_fallback(text, error=f"LLM API failure: {exc}")

    def _call_api(self, text: str) -> LLMResponse:
        import requests

        system = (
            "You are an assistant for phishing/suspicious message classification. "
            "Respond ONLY with JSON containing keys: "
            "label (phishing|legitimate|uncertain), confidence (0-1), "
            "reasoning (short), explanation (short), "
            "probabilities (object with phishing/legitimate/uncertain). "
            "Do not invent quotes that are not in the message."
        )
        user = f"Classify this message and explain using only its content:\n\n{text}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = self._parse_json(content)
        label = parsed.get("label", "uncertain")
        if label not in ("phishing", "legitimate", "uncertain"):
            label = "uncertain"
        return LLMResponse(
            success=True,
            used_llm=True,
            label=label,
            confidence=parsed.get("confidence"),
            reasoning=str(parsed.get("reasoning", "")),
            explanation=str(parsed.get("explanation", "")),
            probabilities=parsed.get("probabilities") or {},
            raw=content,
        )

    def _parse_json(self, content: str) -> Dict[str, Any]:
        content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            raise

    def _local_fallback(self, text: str, error: Optional[str]) -> LLMResponse:
        """Deterministic local stub — clearly NOT an LLM response."""
        return LLMResponse(
            success=True,
            used_llm=False,
            label=None,  # Let the local classifier decide
            confidence=None,
            reasoning="Local fallback active; no external LLM was invoked.",
            explanation="",
            probabilities={},
            error=error,
        )
