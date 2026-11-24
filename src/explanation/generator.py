"""Evidence-grounded explanation generation. Every reference must be traceable."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.classification.classifier import ClassificationResult
from src.validation.validator import ValidationReport
from src.uncertainty.handler import UncertaintyAssessment


@dataclass
class ExplanationResult:
    predicted_category: str
    supporting_evidence: List[str]
    contradictory_evidence: List[str]
    validation_summary: str
    unsupported_claims: List[str]
    uncertainty_status: str
    final_explanation: str
    generation_backend: str  # "local_template" | "llm"
    evidence_traceable: bool = True

    def as_dict(self) -> dict:
        return {
            "predicted_category": self.predicted_category,
            "supporting_evidence": self.supporting_evidence,
            "contradictory_evidence": self.contradictory_evidence,
            "validation_summary": self.validation_summary,
            "unsupported_claims": self.unsupported_claims,
            "uncertainty_status": self.uncertainty_status,
            "final_explanation": self.final_explanation,
            "generation_backend": self.generation_backend,
            "evidence_traceable": self.evidence_traceable,
        }


class ExplanationGenerator:
    """Generate structured, evidence-grounded explanations (local template by default)."""

    def generate(
        self,
        classification: ClassificationResult,
        validation: ValidationReport,
        uncertainty: UncertaintyAssessment,
        llm_text: Optional[str] = None,
        llm_used: bool = False,
    ) -> ExplanationResult:
        supporting = [
            r.span.text for r in validation.valid_supporting
        ]
        contradictory = [
            r.span.text for r in validation.valid_contradictory
        ]
        rejected = [
            f"'{r.span.text}' ({r.rejection_reason})"
            for r in validation.rejected_spans
        ]

        final_label = uncertainty.final_label
        conf_note = ""
        if classification.confidence is not None:
            conf_note = (
                f" Model confidence estimate: {classification.confidence:.2f} "
                f"(not calibrated; distinct from evidence sufficiency="
                f"{uncertainty.evidence_sufficiency})."
            )
        else:
            conf_note = (
                f" Evidence sufficiency: {uncertainty.evidence_sufficiency}."
            )

        validation_summary = (
            f"Validated supporting spans: {len(supporting)}. "
            f"Contradictory spans: {len(contradictory)}. "
            f"Rejected spans: {len(validation.rejected_spans)}. "
            f"Existence pass rate: {validation.existence_pass_rate:.0%}."
        )

        parts = [
            f"Final category: {final_label}.",
            f"Uncertainty status: {uncertainty.status}.",
        ]
        if uncertainty.reasons:
            parts.append("Reasons: " + " ".join(uncertainty.reasons))
        if supporting:
            quoted = ", ".join(f'"{s}"' for s in supporting)
            parts.append(f"Supporting evidence from the input: {quoted}.")
        else:
            parts.append("No validated supporting evidence was found in the input.")
        if contradictory:
            quoted = ", ".join(f'"{s}"' for s in contradictory)
            parts.append(f"Contradictory evidence from the input: {quoted}.")
        if rejected:
            parts.append("Rejected evidence references: " + "; ".join(rejected) + ".")
        if validation.unsupported_claims:
            parts.append(
                "Unsupported claims (not traceable to input): "
                + "; ".join(f'"{c}"' for c in validation.unsupported_claims)
                + "."
            )
        parts.append(validation_summary + conf_note)
        parts.append(
            f"Classification backend: {classification.model_backend}; "
            f"reasoning was treated as unverified until evidence checks completed."
        )

        local_text = " ".join(parts)

        if llm_used and llm_text:
            # Prefer LLM narrative but append validation grounding.
            final = (
                f"[LLM-assisted explanation]\n{llm_text.strip()}\n\n"
                f"[Evidence-grounded validation summary]\n{local_text}"
            )
            backend = "llm"
        else:
            final = local_text
            backend = "local_template"

        return ExplanationResult(
            predicted_category=final_label,
            supporting_evidence=supporting,
            contradictory_evidence=contradictory,
            validation_summary=validation_summary,
            unsupported_claims=list(validation.unsupported_claims),
            uncertainty_status=uncertainty.status,
            final_explanation=final,
            generation_backend=backend,
            evidence_traceable=len(validation.unsupported_claims) == 0,
        )
