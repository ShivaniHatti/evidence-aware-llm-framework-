"""
Evidence Validation Module — independent checks that do not rely on the LLM
validating its own output.

Checks:
1. Existence — does the span text actually occur in the original input?
2. Localization — can we identify exact character positions?
3. Relevance — is the evidence relevant to the predicted category?
4. Support — does it support, contradict, or remain neutral?
5. Unsupported explanation — claims in explanation not traceable to input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import re

from src.evidence.extractor import EvidenceSpan, PHISHING_CUES, LEGITIMATE_CUES


@dataclass
class SpanValidationResult:
    span: EvidenceSpan
    exists_in_input: bool
    localized: bool
    actual_start: Optional[int]
    actual_end: Optional[int]
    relevance: str  # "relevant" | "irrelevant" | "unknown"
    support: str  # "supports" | "contradicts" | "neutral"
    rejection_reason: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "span": self.span.as_dict(),
            "exists_in_input": self.exists_in_input,
            "localized": self.localized,
            "actual_start": self.actual_start,
            "actual_end": self.actual_end,
            "relevance": self.relevance,
            "support": self.support,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class ValidationReport:
    predicted_label: str
    span_results: List[SpanValidationResult] = field(default_factory=list)
    valid_supporting: List[SpanValidationResult] = field(default_factory=list)
    valid_contradictory: List[SpanValidationResult] = field(default_factory=list)
    rejected_spans: List[SpanValidationResult] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    existence_pass_rate: float = 0.0
    all_evidence_valid: bool = False

    def as_dict(self) -> dict:
        return {
            "predicted_label": self.predicted_label,
            "span_results": [r.as_dict() for r in self.span_results],
            "valid_supporting": [r.as_dict() for r in self.valid_supporting],
            "valid_contradictory": [r.as_dict() for r in self.valid_contradictory],
            "rejected_spans": [r.as_dict() for r in self.rejected_spans],
            "unsupported_claims": self.unsupported_claims,
            "existence_pass_rate": self.existence_pass_rate,
            "all_evidence_valid": self.all_evidence_valid,
        }


class EvidenceValidator:
    """Deterministic existence/localization + independent relevance/support checks."""

    def __init__(self) -> None:
        self._phish_res = [re.compile(p, re.IGNORECASE) for p in PHISHING_CUES]
        self._legit_res = [re.compile(p, re.IGNORECASE) for p in LEGITIMATE_CUES]

    def validate_span(
        self,
        original_text: str,
        span: EvidenceSpan,
        predicted_label: str,
    ) -> SpanValidationResult:
        """Run existence, localization, relevance, and support checks on one span."""
        exists, loc_start, loc_end = self._check_existence(original_text, span)
        localized = exists and loc_start is not None

        rejection = None
        if not exists:
            rejection = (
                f"Evidence phrase does not occur in the original input: "
                f"'{span.text}'"
            )
        elif span.start != loc_start or span.end != loc_end:
            # Positions wrong but text exists — correct localization, flag mismatch.
            if span.source == "fabricated" or (
                span.start >= 0
                and (
                    span.start >= len(original_text)
                    or original_text[span.start : span.end] != span.text
                )
            ):
                rejection = (
                    f"Invalid evidence positions: claimed [{span.start}:{span.end}] "
                    f"but actual occurrence is at [{loc_start}:{loc_end}]."
                )

        # Existence alone is not support. Assess relevance independently.
        relevance = self._assess_relevance(span, predicted_label)
        support = self._assess_support(span, predicted_label)

        # If rejected for existence, force support to neutral for safety.
        if not exists:
            relevance = "unknown"
            support = "neutral"

        return SpanValidationResult(
            span=span,
            exists_in_input=exists,
            localized=localized,
            actual_start=loc_start,
            actual_end=loc_end,
            relevance=relevance,
            support=support if exists else "neutral",
            rejection_reason=rejection,
        )

    def validate_spans(
        self,
        original_text: str,
        spans: List[EvidenceSpan],
        predicted_label: str,
    ) -> ValidationReport:
        results = [
            self.validate_span(original_text, s, predicted_label) for s in spans
        ]
        valid_supporting = []
        valid_contradictory = []
        rejected = []

        for r in results:
            if not r.exists_in_input or r.rejection_reason:
                rejected.append(r)
                continue
            if r.support == "supports" and r.relevance == "relevant":
                valid_supporting.append(r)
            elif r.support == "contradicts":
                valid_contradictory.append(r)

        existence_ok = sum(1 for r in results if r.exists_in_input)
        rate = (existence_ok / len(results)) if results else 1.0

        return ValidationReport(
            predicted_label=predicted_label,
            span_results=results,
            valid_supporting=valid_supporting,
            valid_contradictory=valid_contradictory,
            rejected_spans=rejected,
            unsupported_claims=[],
            existence_pass_rate=rate,
            all_evidence_valid=len(rejected) == 0,
        )

    def detect_unsupported_claims(
        self,
        original_text: str,
        explanation_text: str,
        known_evidence_texts: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Detect explanation claims that cannot be traced to the original input.

        Strategy (deterministic, independent of the generator):
        - Extract quoted phrases and strong claim fragments from the explanation.
        - Flag any that do not appear (case-insensitive substring) in the input
          and are not among already-validated evidence texts.
        """
        unsupported: List[str] = []
        known: Set[str] = set()
        if known_evidence_texts:
            known = {k.lower().strip() for k in known_evidence_texts}

        # Quoted claims
        for m in re.finditer(r"[\"']([^\"']{4,120})[\"']", explanation_text):
            claim = m.group(1).strip()
            if not self._claim_traceable(original_text, claim, known):
                unsupported.append(claim)

        # Patterned assertive fragments often hallucinated by LLMs
        claim_patterns = [
            r"because (?:the )?(?:message |email )?(?:says|mentions|contains|states) (.+?)(?:\.|$)",
            r"evidence(?: that)?:?\s*[\"']?(.{8,100}?)[\"']?(?:\.|$)",
            r"based on (?:the )?(?:phrase|words?|sentence)\s*[\"']?(.{5,80}?)[\"']?",
        ]
        for pat in claim_patterns:
            for m in re.finditer(pat, explanation_text, re.IGNORECASE):
                claim = m.group(1).strip(" .,;:\"'")
                if len(claim) >= 5 and not self._claim_traceable(original_text, claim, known):
                    unsupported.append(claim)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in unsupported:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    def attach_unsupported_claims(
        self, report: ValidationReport, claims: List[str]
    ) -> ValidationReport:
        report.unsupported_claims = claims
        return report

    def _check_existence(
        self, original_text: str, span: EvidenceSpan
    ) -> tuple:
        """Return (exists, actual_start, actual_end)."""
        phrase = span.text
        if not phrase:
            return False, None, None

        # Exact match at claimed position
        if (
            0 <= span.start < len(original_text)
            and span.end <= len(original_text)
            and original_text[span.start : span.end] == phrase
        ):
            return True, span.start, span.end

        # Exact substring search
        idx = original_text.find(phrase)
        if idx >= 0:
            return True, idx, idx + len(phrase)

        # Case-insensitive
        idx = original_text.lower().find(phrase.lower())
        if idx >= 0:
            return True, idx, idx + len(phrase)

        # Collapse whitespace comparison
        compact_orig = re.sub(r"\s+", " ", original_text)
        compact_phrase = re.sub(r"\s+", " ", phrase)
        idx = compact_orig.lower().find(compact_phrase.lower())
        if idx >= 0:
            # Approximate localization on original (best-effort)
            return True, idx, idx + len(compact_phrase)

        return False, None, None

    def _assess_relevance(self, span: EvidenceSpan, predicted_label: str) -> str:
        """
        Independent relevance: cue type vs predicted category.
        Appearance in text ≠ relevance to the prediction.
        """
        if predicted_label == "uncertain":
            return "relevant" if span.cue_type in ("phishing", "legitimate") else "unknown"

        if span.cue_type == "phishing":
            return "relevant" if predicted_label == "phishing" else "irrelevant"
        if span.cue_type == "legitimate":
            return "relevant" if predicted_label == "legitimate" else "irrelevant"
        if span.cue_type == "custom":
            # Heuristic: match against cue libraries
            text = span.text
            if any(p.search(text) for p in self._phish_res):
                return "relevant" if predicted_label == "phishing" else "irrelevant"
            if any(p.search(text) for p in self._legit_res):
                return "relevant" if predicted_label == "legitimate" else "irrelevant"
            return "unknown"
        return "unknown"

    def _assess_support(self, span: EvidenceSpan, predicted_label: str) -> str:
        """Does this evidence support, contradict, or stay neutral toward the label?"""
        if predicted_label == "uncertain":
            return "neutral"

        if span.cue_type == "phishing":
            if predicted_label == "phishing":
                return "supports"
            if predicted_label == "legitimate":
                return "contradicts"
        if span.cue_type == "legitimate":
            if predicted_label == "legitimate":
                return "supports"
            if predicted_label == "phishing":
                return "contradicts"

        # Custom: try library match
        text = span.text
        is_phish = any(p.search(text) for p in self._phish_res)
        is_legit = any(p.search(text) for p in self._legit_res)
        if is_phish and not is_legit:
            return "supports" if predicted_label == "phishing" else "contradicts"
        if is_legit and not is_phish:
            return "supports" if predicted_label == "legitimate" else "contradicts"
        return "neutral"

    def _claim_traceable(
        self, original_text: str, claim: str, known: Set[str]
    ) -> bool:
        c = claim.lower().strip()
        if c in known:
            return True
        if c in original_text.lower():
            return True
        # Token overlap threshold for soft tracing
        claim_tokens = set(re.findall(r"[a-z0-9]{3,}", c))
        if not claim_tokens:
            return False
        orig_tokens = set(re.findall(r"[a-z0-9]{3,}", original_text.lower()))
        overlap = len(claim_tokens & orig_tokens) / len(claim_tokens)
        return overlap >= 0.8
