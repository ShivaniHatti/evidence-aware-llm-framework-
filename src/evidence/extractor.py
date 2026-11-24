"""Evidence identification: extract supporting/contradictory spans with positions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import re


# Lexical cues for phishing/suspicious messages (demonstration domain).
PHISHING_CUES = [
    r"verify\s+(your\s+)?account",
    r"confirm\s+(your\s+)?identity",
    r"urgent(ly)?",
    r"immediate(ly)?",
    r"suspend(ed|sion)?",
    r"click\s+(here|the\s+link|below)",
    r"act\s+now",
    r"limited\s+time",
    r"password\s+(expir|reset|update)",
    r"unusual\s+activity",
    r"security\s+alert",
    r"wire\s+transfer",
    r"gift\s+card",
    r"bitcoin|cryptocurrency|crypto\s+wallet",
    r"won\s+(a\s+)?prize|lottery|inheritance",
    r"provide\s+(your\s+)?(ssn|social\s+security|credit\s+card|bank\s+details|otp|pin)",
    r"login\s+(here|now|immediately)",
    r"http[s]?://\S+",
    r"bit\.ly/\S+|tinyurl\.com/\S+",
    r"dear\s+(customer|user|valued\s+customer)",
    r"account\s+will\s+be\s+(closed|locked|disabled)",
]

LEGITIMATE_CUES = [
    r"unsubscribe",
    r"no\s+action\s+(is\s+)?required",
    r"for\s+your\s+records",
    r"order\s+(confirmation|receipt|number)",
    r"shipping\s+(confirmation|update|notification)",
    r"meeting\s+(invite|invitation|reminder)",
    r"calendar\s+invite",
    r"thank\s+you\s+for\s+(your\s+)?(purchase|order|subscription)",
    r"your\s+receipt",
    r"newsletter",
    r"privacy\s+policy",
    r"customer\s+support\s+hours",
]


@dataclass
class EvidenceSpan:
    """A single evidence phrase with exact source location."""

    text: str
    start: int
    end: int
    cue_type: str  # "phishing" | "legitimate" | "neutral" | "custom"
    role_hint: str  # "supporting_candidate" | "contradictory_candidate" | "neutral"
    source: str = "rule_based"  # or "llm" | "manual"

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "cue_type": self.cue_type,
            "role_hint": self.role_hint,
            "source": self.source,
        }


@dataclass
class EvidenceBundle:
    """All candidate evidence extracted from an input."""

    spans: List[EvidenceSpan] = field(default_factory=list)
    sentences: List[Tuple[int, int, str]] = field(default_factory=list)

    def supporting_candidates(self, predicted_label: str) -> List[EvidenceSpan]:
        if predicted_label == "phishing":
            return [s for s in self.spans if s.cue_type == "phishing"]
        if predicted_label == "legitimate":
            return [s for s in self.spans if s.cue_type == "legitimate"]
        return list(self.spans)

    def contradictory_candidates(self, predicted_label: str) -> List[EvidenceSpan]:
        if predicted_label == "phishing":
            return [s for s in self.spans if s.cue_type == "legitimate"]
        if predicted_label == "legitimate":
            return [s for s in self.spans if s.cue_type == "phishing"]
        return []


class EvidenceExtractor:
    """Identify relevant phrases and contextual clues with character positions."""

    def __init__(self) -> None:
        self._phishing_patterns = [
            (re.compile(p, re.IGNORECASE), p) for p in PHISHING_CUES
        ]
        self._legitimate_patterns = [
            (re.compile(p, re.IGNORECASE), p) for p in LEGITIMATE_CUES
        ]

    def extract(
        self,
        text: str,
        sentence_spans: Optional[List[Tuple[int, int, str]]] = None,
    ) -> EvidenceBundle:
        spans: List[EvidenceSpan] = []
        seen = set()

        for pattern, _ in self._phishing_patterns:
            for m in pattern.finditer(text):
                key = (m.start(), m.end())
                if key in seen:
                    continue
                seen.add(key)
                spans.append(
                    EvidenceSpan(
                        text=text[m.start() : m.end()],
                        start=m.start(),
                        end=m.end(),
                        cue_type="phishing",
                        role_hint="supporting_candidate",
                        source="rule_based",
                    )
                )

        for pattern, _ in self._legitimate_patterns:
            for m in pattern.finditer(text):
                key = (m.start(), m.end())
                if key in seen:
                    continue
                seen.add(key)
                spans.append(
                    EvidenceSpan(
                        text=text[m.start() : m.end()],
                        start=m.start(),
                        end=m.end(),
                        cue_type="legitimate",
                        role_hint="supporting_candidate",
                        source="rule_based",
                    )
                )

        spans.sort(key=lambda s: s.start)
        return EvidenceBundle(spans=spans, sentences=sentence_spans or [])

    def make_manual_span(
        self,
        text: str,
        phrase: str,
        cue_type: str = "custom",
        role_hint: str = "supporting_candidate",
        source: str = "manual",
    ) -> Optional[EvidenceSpan]:
        """Locate a phrase in text; return None if not found (used by lab tests)."""
        idx = text.find(phrase)
        if idx < 0:
            # Case-insensitive fallback for localization attempt
            lower = text.lower()
            p = phrase.lower()
            idx = lower.find(p)
            if idx < 0:
                return None
            phrase = text[idx : idx + len(phrase)]
        return EvidenceSpan(
            text=phrase,
            start=idx,
            end=idx + len(phrase),
            cue_type=cue_type,
            role_hint=role_hint,
            source=source,
        )

    def fabricate_span(
        self,
        phrase: str,
        fake_start: int = 0,
        cue_type: str = "phishing",
    ) -> EvidenceSpan:
        """Create an evidence claim that may NOT exist in the input (adversarial test)."""
        return EvidenceSpan(
            text=phrase,
            start=fake_start,
            end=fake_start + len(phrase),
            cue_type=cue_type,
            role_hint="supporting_candidate",
            source="fabricated",
        )
