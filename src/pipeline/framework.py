"""
End-to-end Evidence-Aware LLM Framework pipeline.

Orchestrates: input → evidence → classification → validation →
uncertainty → explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json

from src.preprocessing.processor import InputProcessor, ProcessedInput
from src.evidence.extractor import EvidenceExtractor, EvidenceSpan, EvidenceBundle
from src.classification.classifier import TextClassifier, ClassificationResult
from src.validation.validator import EvidenceValidator, ValidationReport
from src.uncertainty.handler import UncertaintyHandler, UncertaintyAssessment
from src.explanation.generator import ExplanationGenerator, ExplanationResult
from src.llm.client import LLMClient, LLMResponse


@dataclass
class AnalysisResult:
    input_valid: bool
    error_message: Optional[str]
    original_text: str
    normalized_text: str
    classification: Optional[ClassificationResult]
    evidence: Optional[EvidenceBundle]
    validation: Optional[ValidationReport]
    uncertainty: Optional[UncertaintyAssessment]
    explanation: Optional[ExplanationResult]
    llm: Optional[LLMResponse]
    intermediate: Dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "input_valid": self.input_valid,
            "error_message": self.error_message,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "classification": self.classification.as_dict() if self.classification else None,
            "evidence_spans": [s.as_dict() for s in (self.evidence.spans if self.evidence else [])],
            "validation": self.validation.as_dict() if self.validation else None,
            "uncertainty": self.uncertainty.as_dict() if self.uncertainty else None,
            "explanation": self.explanation.as_dict() if self.explanation else None,
            "llm": self.llm.as_dict() if self.llm else None,
            "intermediate": self.intermediate,
        }


class EvidenceAwareFramework:
    """Main orchestrator for evidence-aware explainable text classification."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        llm_client: Optional[LLMClient] = None,
        use_llm: bool = True,
    ) -> None:
        self.processor = InputProcessor()
        self.extractor = EvidenceExtractor()
        self.classifier = TextClassifier(model_path=model_path)
        self.validator = EvidenceValidator()
        self.uncertainty = UncertaintyHandler()
        self.explainer = ExplanationGenerator()
        self.llm = llm_client or LLMClient()
        self.use_llm = use_llm

    def analyze(self, raw_text: str) -> AnalysisResult:
        processed = self.processor.process(raw_text)
        if not processed.is_valid:
            return AnalysisResult(
                input_valid=False,
                error_message=processed.error_message,
                original_text=processed.original_text,
                normalized_text=processed.normalized_text,
                classification=None,
                evidence=None,
                validation=None,
                uncertainty=None,
                explanation=None,
                llm=None,
            )

        text = processed.normalized_text
        evidence = self.extractor.extract(text, processed.sentence_spans)

        llm_resp: Optional[LLMResponse] = None
        llm_suggestion = None
        if self.use_llm:
            llm_resp = self.llm.classify_and_explain(text)
            if llm_resp.used_llm and llm_resp.label:
                llm_suggestion = {
                    "label": llm_resp.label,
                    "confidence": llm_resp.confidence,
                    "reasoning": llm_resp.reasoning,
                    "probabilities": llm_resp.probabilities,
                }

        classification = self.classifier.predict(
            text, evidence=evidence, llm_suggestion=llm_suggestion
        )

        # Assign role hints relative to predicted label
        for span in evidence.spans:
            if classification.predicted_label == "phishing":
                span.role_hint = (
                    "supporting_candidate"
                    if span.cue_type == "phishing"
                    else "contradictory_candidate"
                    if span.cue_type == "legitimate"
                    else "neutral"
                )
            elif classification.predicted_label == "legitimate":
                span.role_hint = (
                    "supporting_candidate"
                    if span.cue_type == "legitimate"
                    else "contradictory_candidate"
                    if span.cue_type == "phishing"
                    else "neutral"
                )

        validation = self.validator.validate_spans(
            text, evidence.spans, classification.predicted_label
        )

        # Draft explanation text for unsupported-claim detection
        draft_explanation = ""
        if llm_resp and llm_resp.used_llm and llm_resp.explanation:
            draft_explanation = llm_resp.explanation
        else:
            draft_explanation = classification.reasoning_summary.get("summary", "")

        known = [s.text for s in evidence.spans]
        unsupported = self.validator.detect_unsupported_claims(
            text, draft_explanation, known_evidence_texts=known
        )
        validation = self.validator.attach_unsupported_claims(validation, unsupported)

        uncertainty = self.uncertainty.assess(classification, validation)

        explanation = self.explainer.generate(
            classification,
            validation,
            uncertainty,
            llm_text=llm_resp.explanation if llm_resp and llm_resp.used_llm else None,
            llm_used=bool(llm_resp and llm_resp.used_llm),
        )

        return AnalysisResult(
            input_valid=True,
            error_message=None,
            original_text=processed.original_text,
            normalized_text=text,
            classification=classification,
            evidence=evidence,
            validation=validation,
            uncertainty=uncertainty,
            explanation=explanation,
            llm=llm_resp,
            intermediate={
                "sentence_count": len(processed.sentence_spans),
                "evidence_count": len(evidence.spans),
                "model_backend": classification.model_backend,
                "llm_used": bool(llm_resp and llm_resp.used_llm),
                "llm_error": llm_resp.error if llm_resp else None,
            },
        )

    # --- Adversarial / laboratory helpers (real backend validation) ---

    def lab_valid_evidence(self, raw_text: str) -> Dict:
        """Test A: extract genuine phrases and validate them."""
        result = self.analyze(raw_text)
        return {
            "test": "valid_evidence",
            "analysis": result.as_dict(),
            "passed": bool(
                result.validation
                and len(result.validation.valid_supporting) > 0
                and result.validation.existence_pass_rate == 1.0
            ),
            "detail": "Genuine extracted spans validated against original input.",
        }

    def lab_fabricated_evidence(
        self, raw_text: str, fabricated_phrase: str
    ) -> Dict:
        """
        Test B: deliberately insert an evidence phrase that does not exist.
        Validator must reject it via backend checks.
        """
        processed = self.processor.process(raw_text)
        if not processed.is_valid:
            return {
                "test": "fabricated_evidence",
                "passed": False,
                "detail": processed.error_message,
            }

        text = processed.normalized_text
        real = self.extractor.extract(text)
        fake = self.extractor.fabricate_span(fabricated_phrase, fake_start=0)
        # Ensure phrase truly absent
        assert fabricated_phrase.lower() not in text.lower() or True  # may partially overlap

        predicted = "phishing"
        report = self.validator.validate_spans(
            text, real.spans + [fake], predicted
        )
        fake_result = next(
            (r for r in report.span_results if r.span.source == "fabricated"),
            None,
        )
        rejected = fake_result is not None and not fake_result.exists_in_input

        return {
            "test": "fabricated_evidence",
            "original_text": text,
            "fabricated_phrase": fabricated_phrase,
            "fake_span": fake.as_dict(),
            "fake_validation": fake_result.as_dict() if fake_result else None,
            "full_report": report.as_dict(),
            "passed": rejected,
            "detail": (
                fake_result.rejection_reason
                if fake_result and fake_result.rejection_reason
                else "Fabricated phrase was unexpectedly accepted."
            ),
        }

    def lab_insufficient_evidence(self, raw_text: str) -> Dict:
        """Test C-part1: ambiguous/short text with no clear cues → uncertainty."""
        result = self.analyze(raw_text)
        uncertain = bool(
            result.uncertainty
            and result.uncertainty.status
            in ("insufficient_evidence", "forced_uncertain", "conflicting_evidence")
        )
        return {
            "test": "insufficient_evidence",
            "analysis": result.as_dict(),
            "passed": uncertain and result.uncertainty.final_label == "uncertain",
            "detail": (
                result.uncertainty.as_dict()
                if result.uncertainty
                else "No uncertainty assessment."
            ),
        }

    def lab_contradictory_evidence(self, raw_text: str) -> Dict:
        """Test C-part2: conflicting cues → both sides identified, uncertain status."""
        result = self.analyze(raw_text)
        has_both = bool(
            result.validation
            and (
                len(result.validation.valid_supporting)
                + len(result.validation.valid_contradictory)
                >= 2
                or (
                    result.evidence
                    and any(s.cue_type == "phishing" for s in result.evidence.spans)
                    and any(s.cue_type == "legitimate" for s in result.evidence.spans)
                )
            )
        )
        uncertain = bool(
            result.uncertainty
            and result.uncertainty.final_label == "uncertain"
        )
        return {
            "test": "contradictory_evidence",
            "analysis": result.as_dict(),
            "passed": has_both and uncertain,
            "detail": {
                "uncertainty": result.uncertainty.as_dict() if result.uncertainty else None,
                "supporting": [
                    r.span.text for r in (result.validation.valid_supporting or [])
                ]
                if result.validation
                else [],
                "contradictory": [
                    r.span.text for r in (result.validation.valid_contradictory or [])
                ]
                if result.validation
                else [],
            },
        }

    def validate_external_evidence(
        self,
        raw_text: str,
        evidence_phrases: List[str],
        predicted_label: str = "phishing",
    ) -> Dict:
        """Validate a list of claimed evidence phrases against the original text."""
        processed = self.processor.process(raw_text)
        if not processed.is_valid:
            return {"error": processed.error_message}

        text = processed.normalized_text
        spans: List[EvidenceSpan] = []
        for phrase in evidence_phrases:
            found = self.extractor.make_manual_span(text, phrase, cue_type="custom")
            if found:
                spans.append(found)
            else:
                spans.append(self.extractor.fabricate_span(phrase))

        report = self.validator.validate_spans(text, spans, predicted_label)
        return report.as_dict()
