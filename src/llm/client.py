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
import time

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
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.mode = (mode or os.getenv("LLM_MODE", "auto")).lower()
        self.timeout = timeout
        self.max_retries = max_retries

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

        # Retry logic with exponential backoff for rate limits
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
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
            except requests.exceptions.HTTPError as e:
                last_error = e
                error_data = self._extract_error_data(e.response)
                error_type = error_data.get("type", "")
                error_message = error_data.get("message", str(e))

                # Check if this is a quota/billing error (not retryable)
                if error_type == "insufficient_quota" or "quota" in error_message.lower() or "billing" in error_message.lower():
                    return self._local_fallback(
                        text,
                        error=f"OpenAI quota/billing error: {error_message}. Please check your OpenAI account billing.",
                    )

                # Check if this is a rate limit error (retryable)
                if e.response.status_code == 429:
                    if attempt < self.max_retries:
                        # Exponential backoff: 1s, 2s, 4s
                        backoff = 2 ** attempt
                        time.sleep(backoff)
                        continue
                    else:
                        return self._local_fallback(
                            text,
                            error=f"Rate limit exceeded after {self.max_retries} retries: {error_message}",
                        )

                # Other HTTP errors - don't retry
                return self._local_fallback(
                    text,
                    error=f"HTTP error {e.response.status_code}: {error_message}",
                )
            except requests.exceptions.RequestException as e:
                last_error = e
                # Network errors - retry once
                if attempt < 1:
                    time.sleep(1)
                    continue
                return self._local_fallback(
                    text,
                    error=f"Network error: {str(e)}",
                )

        # Should not reach here, but fallback just in case
        return self._local_fallback(
            text,
            error=f"API call failed after retries: {str(last_error)}",
        )

    def _extract_error_data(self, response) -> Dict[str, str]:
        """Extract error type and message from OpenAI error response."""
        try:
            data = response.json()
            if "error" in data:
                error = data["error"]
                return {
                    "type": error.get("type", ""),
                    "message": error.get("message", str(data)),
                }
            return {"type": "", "message": str(data)}
        except (json.JSONDecodeError, ValueError):
            return {"type": "", "message": response.text}

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
