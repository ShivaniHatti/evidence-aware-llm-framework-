from src.evidence.extractor import EvidenceExtractor
from src.preprocessing.processor import InputProcessor


def test_evidence_extraction_finds_phishing_cues():
    text = "Please verify your account immediately due to unusual activity."
    ext = EvidenceExtractor()
    bundle = ext.extract(text)
    assert len(bundle.spans) >= 2
    types = {s.cue_type for s in bundle.spans}
    assert "phishing" in types
    for s in bundle.spans:
        assert text[s.start : s.end] == s.text


def test_exact_evidence_matching_positions():
    text = "Act now or your account will be suspended."
    ext = EvidenceExtractor()
    bundle = ext.extract(text)
    assert bundle.spans
    for s in bundle.spans:
        assert 0 <= s.start < s.end <= len(text)
        assert text[s.start : s.end] == s.text


def test_manual_span_localization():
    text = "Thank you for your purchase. No action is required."
    ext = EvidenceExtractor()
    span = ext.make_manual_span(text, "No action is required", cue_type="legitimate")
    assert span is not None
    assert text[span.start : span.end] == "No action is required"


def test_fabricate_span_marks_source():
    ext = EvidenceExtractor()
    fake = ext.fabricate_span("this phrase is invented")
    assert fake.source == "fabricated"
