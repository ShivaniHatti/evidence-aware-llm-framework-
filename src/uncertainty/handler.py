"""Uncertainty handling: insufficient, conflicting, or invalid evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.validation.validator import ValidationReport
from src.classification.classifier import ClassificationResult


@dataclass
class UncertaintyAssessment:
    is_uncertain: bool
    status: str  # "sufficient" | "insufficient_evidence" | "conflicting_evidence" | "invalid_evidence" | "forced_uncertain"
    reasons: list
    final_label: str
    evidence_sufficiency: str  # distinct from model confidence

    def as_dict(self) -> dict:
        return {
            "is_uncertain": self.is_uncertain,
            "status": self.status,
            "reasons": self.reasons,
            "final_label": self.final_label,
            "evidence_sufficiency": self.evidence_sufficiency,
        }


class UncertaintyHandler:
    """
    Avoid forced classification when evidence is inadequate.
    Distinguishes model confidence from evidence sufficiency.
    """

    MIN_SUPPORTING = 1

    def assess(
        self,
        classification: ClassificationResult,
        validation: ValidationReport,
    ) -> UncertaintyAssessment:
        reasons = []
        predicted = classification.predicted_label

        n_support = len(validation.valid_supporting)
        n_contradict = len(validation.valid_contradictory)
        n_rejected = len(validation.rejected_spans)
        n_unsupported = len(validation.unsupported_claims)

        if n_rejected > 0:
            reasons.append(
                f"{n_rejected} evidence span(s) failed existence/localization checks."
            )

        if n_unsupported > 0:
            reasons.append(
                f"{n_unsupported} unsupported claim(s) detected in the explanation."
            )

        if n_support > 0 and n_contradict > 0:
            reasons.append(
                f"Conflicting evidence: {n_support} supporting vs {n_contradict} contradictory span(s)."
            )
            return UncertaintyAssessment(
                is_uncertain=True,
                status="conflicting_evidence",
                reasons=reasons,
                final_label="uncertain",
                evidence_sufficiency="conflicting",
            )

        if n_rejected > 0 and n_support == 0:
            reasons.append("No valid supporting evidence after rejecting invalid references.")
            return UncertaintyAssessment(
                is_uncertain=True,
                status="invalid_evidence",
                reasons=reasons,
                final_label="uncertain",
                evidence_sufficiency="invalid",
            )

        if n_support < self.MIN_SUPPORTING:
            reasons.append(
                "Insufficient validated supporting evidence for a confident category decision."
            )
            return UncertaintyAssessment(
                is_uncertain=True,
                status="insufficient_evidence",
                reasons=reasons,
                final_label="uncertain",
                evidence_sufficiency="insufficient",
            )

        # Enough supporting evidence, no contradiction
        if predicted == "uncertain":
            # Classifier was uncertain but we have support — keep classifier cue mild
            reasons.append(
                "Classifier was provisional uncertain, but validated supporting evidence exists; "
                "retaining cautious uncertain status."
            )
            return UncertaintyAssessment(
                is_uncertain=True,
                status="forced_uncertain",
                reasons=reasons,
                final_label="uncertain",
                evidence_sufficiency="partial",
            )

        return UncertaintyAssessment(
            is_uncertain=False,
            status="sufficient",
            reasons=["Validated supporting evidence is present without contradiction."],
            final_label=predicted,
            evidence_sufficiency="sufficient",
        )
