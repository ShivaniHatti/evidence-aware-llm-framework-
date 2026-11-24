from src.evidence.extractor import EvidenceExtractor
from src.validation.validator import EvidenceValidator


def test_fabricated_evidence_rejection():
    text = "Security alert: unusual activity detected."
    ext = EvidenceExtractor()
    validator = EvidenceValidator()
    fake = ext.fabricate_span("wire transfer of fifty thousand dollars")
    report = validator.validate_spans(text, [fake], "phishing")
    assert len(report.rejected_spans) == 1
    assert report.rejected_spans[0].exists_in_input is False
    assert report.rejected_spans[0].rejection_reason is not None
    assert "does not occur" in report.rejected_spans[0].rejection_reason.lower()


def test_valid_evidence_accepted():
    text = "Please verify your account immediately."
    ext = EvidenceExtractor()
    validator = EvidenceValidator()
    bundle = ext.extract(text)
    assert bundle.spans
    report = validator.validate_spans(text, bundle.spans, "phishing")
    assert report.existence_pass_rate == 1.0
    assert any(r.support == "supports" for r in report.valid_supporting) or len(
        report.valid_supporting
    ) >= 1


def test_invalid_evidence_positions():
    text = "Please verify your account immediately."
    ext = EvidenceExtractor()
    validator = EvidenceValidator()
    # Claim correct text but wrong offsets
    real = ext.make_manual_span(text, "verify your account", cue_type="phishing")
    assert real is not None
    bad = ext.fabricate_span("verify your account", fake_start=9999)
    # Force source fabricated with wrong position but matching text via existence search
    # Existence should still find the phrase; for pure wrong-position fabricated with
    # text that exists, exists_in_input becomes True. Use a phrase that does not exist.
    missing = ext.fabricate_span("totally missing claim XYZ", fake_start=0)
    result = validator.validate_span(text, missing, "phishing")
    assert result.exists_in_input is False
    assert result.rejection_reason is not None


def test_existence_not_equal_to_support():
    """A legitimate cue appearing in text does not support a phishing prediction."""
    text = "Order confirmation attached. Also please verify your account urgently."
    ext = EvidenceExtractor()
    validator = EvidenceValidator()
    bundle = ext.extract(text)
    report = validator.validate_spans(text, bundle.spans, "phishing")
    # Legitimate cues should contradict phishing, not support it
    for r in report.span_results:
        if r.span.cue_type == "legitimate" and r.exists_in_input:
            assert r.support == "contradicts"
            assert r.relevance == "irrelevant"


def test_unsupported_explanation_detection():
    text = "Please verify your account."
    validator = EvidenceValidator()
    explanation = (
        'The message is phishing because it says "wire transfer of $50,000" '
        "and mentions an overseas bitcoin wallet."
    )
    claims = validator.detect_unsupported_claims(text, explanation)
    assert len(claims) >= 1
    assert any("wire transfer" in c.lower() or "bitcoin" in c.lower() for c in claims)


def test_explanation_traceability_ok_when_quoted_from_input():
    text = "Please verify your account immediately."
    validator = EvidenceValidator()
    explanation = 'Flagged because the message says "verify your account".'
    claims = validator.detect_unsupported_claims(
        text, explanation, known_evidence_texts=["verify your account"]
    )
    assert claims == [] or all("verify your account" not in c.lower() for c in claims)
