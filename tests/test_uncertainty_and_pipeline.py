def test_insufficient_evidence(framework, demo_samples):
    sample = next(s for s in demo_samples["samples"] if s["purpose"] == "insufficient_evidence")
    out = framework.lab_insufficient_evidence(sample["text"])
    assert out["passed"] is True
    assert out["analysis"]["uncertainty"]["final_label"] == "uncertain"


def test_contradictory_evidence(framework, demo_samples):
    sample = next(s for s in demo_samples["samples"] if s["purpose"] == "contradictory_evidence")
    out = framework.lab_contradictory_evidence(sample["text"])
    assert out["passed"] is True
    analysis = out["analysis"]
    assert analysis["uncertainty"]["final_label"] == "uncertain"
    cues = {e["cue_type"] for e in analysis["evidence_spans"]}
    assert "phishing" in cues and "legitimate" in cues


def test_valid_evidence_lab(framework, demo_samples):
    sample = next(s for s in demo_samples["samples"] if s["id"] == "demo_phish_01")
    out = framework.lab_valid_evidence(sample["text"])
    assert out["passed"] is True
    assert out["analysis"]["validation"]["existence_pass_rate"] == 1.0


def test_fabricated_evidence_lab(framework, demo_samples):
    sample = next(s for s in demo_samples["samples"] if s["id"] == "demo_fabricated_base")
    phrase = sample["fabricated_phrase"]
    out = framework.lab_fabricated_evidence(sample["text"], phrase)
    assert out["passed"] is True
    assert out["fake_validation"]["exists_in_input"] is False
    assert "does not occur" in (out["detail"] or "").lower()


def test_end_to_end_analyze_valid(framework):
    text = (
        "Dear customer, unusual activity on your account. "
        "Please verify your account immediately or it will be suspended."
    )
    result = framework.analyze(text)
    assert result.input_valid is True
    assert result.classification is not None
    assert result.validation is not None
    assert result.uncertainty is not None
    assert result.explanation is not None
    assert result.explanation.final_explanation
    # Supporting evidence should be traceable
    for s in result.explanation.supporting_evidence:
        assert s.lower() in result.normalized_text.lower()


def test_empty_input_pipeline(framework):
    result = framework.analyze("")
    assert result.input_valid is False
    assert result.error_message
