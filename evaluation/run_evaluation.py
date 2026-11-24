"""
Evaluation framework comparing:
  Baseline: ordinary classification + generated explanation (no evidence validation)
  Proposed: classification with evidence identification and validation

Writes actual computed metrics to results/ — does not fabricate scores.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.framework import EvidenceAwareFramework
from src.llm.client import LLMClient
from src.evidence.extractor import EvidenceExtractor
from src.validation.validator import EvidenceValidator


def load_eval() -> List[dict]:
    path = ROOT / "data" / "evaluation" / "eval_dataset.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["samples"]


def span_overlap(pred_text: str, gold_text: str) -> bool:
    a, b = pred_text.lower(), gold_text.lower()
    return a in b or b in a or a == b


def evidence_span_metrics(
    predicted_spans: List[str], gold_spans: List[str]
) -> Tuple[float, float, float]:
    if not gold_spans and not predicted_spans:
        return 1.0, 1.0, 1.0
    if not gold_spans:
        return 0.0, 1.0, 0.0
    if not predicted_spans:
        return 1.0, 0.0, 0.0

    matched_gold = set()
    matched_pred = set()
    for i, p in enumerate(predicted_spans):
        for j, g in enumerate(gold_spans):
            if j in matched_gold:
                continue
            if span_overlap(p, g):
                matched_gold.add(j)
                matched_pred.add(i)
                break
    precision = len(matched_pred) / len(predicted_spans)
    recall = len(matched_gold) / len(gold_spans)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def run() -> dict:
    model_path = ROOT / "results" / "models" / "tfidf_logreg.joblib"
    fw = EvidenceAwareFramework(
        model_path=model_path if model_path.exists() else None,
        llm_client=LLMClient(mode="local"),
        use_llm=True,
    )
    samples = load_eval()

    y_true = []
    y_pred_proposed = []
    y_pred_baseline = []

    evidence_prs = []
    existence_correct = 0
    existence_total = 0
    unsupported_detection_cases = 0
    unsupported_detected = 0
    insufficient_cases = 0
    insufficient_detected = 0
    conflict_cases = 0
    conflict_detected = 0

    extractor = EvidenceExtractor()
    validator = EvidenceValidator()

    per_sample = []

    for s in samples:
        text = s["text"]
        gold = s["label"]
        y_true.append(gold)

        # Proposed framework
        result = fw.analyze(text)
        proposed_label = (
            result.uncertainty.final_label
            if result.uncertainty
            else (result.classification.predicted_label if result.classification else "uncertain")
        )
        y_pred_proposed.append(proposed_label)

        # Baseline: classifier label without forcing uncertainty from evidence rules
        baseline_label = (
            result.classification.predicted_label if result.classification else "uncertain"
        )
        # Baseline ignores validation — if classifier said phishing/legitimate, keep it
        # even when evidence is insufficient (ordinary classifier behavior).
        if result.classification and result.classification.model_backend:
            # Re-predict without evidence-driven uncertain override for a fairer baseline
            from src.classification.classifier import TextClassifier

            bare = TextClassifier(model_path=model_path if model_path.exists() else None)
            # Use pipeline directly if fitted
            if bare.pipeline and hasattr(bare.pipeline.named_steps["clf"], "classes_"):
                baseline_label = bare.pipeline.predict([text])[0]
            else:
                baseline_label = result.classification.predicted_label
        y_pred_baseline.append(baseline_label)

        pred_spans = [sp.text for sp in (result.evidence.spans if result.evidence else [])]
        gold_spans = s.get("evidence_spans", [])
        p, r, f1 = evidence_span_metrics(pred_spans, gold_spans)
        evidence_prs.append({"id": s["id"], "precision": p, "recall": r, "f1": f1})

        # Existence validation accuracy on gold spans
        for g in gold_spans:
            existence_total += 1
            span = extractor.make_manual_span(text, g, cue_type="custom")
            if span is None:
                # Gold annotation missing from text — skip
                existence_total -= 1
                continue
            vr = validator.validate_span(text, span, gold if gold != "uncertain" else "phishing")
            if vr.exists_in_input:
                existence_correct += 1

        # Fabricated unsupported detection probe (only for phishing samples)
        if gold == "phishing":
            unsupported_detection_cases += 1
            fake_phrase = "WIRE_TRANSFER_TO_MOON_BASE_XYZ_999"
            lab = fw.lab_fabricated_evidence(text, fake_phrase)
            if lab.get("passed"):
                unsupported_detected += 1

        if gold == "uncertain" and not gold_spans:
            insufficient_cases += 1
            if result.uncertainty and result.uncertainty.status in (
                "insufficient_evidence",
                "forced_uncertain",
                "conflicting_evidence",
                "invalid_evidence",
            ):
                insufficient_detected += 1

        if gold == "uncertain" and gold_spans:
            conflict_cases += 1
            if result.uncertainty and result.uncertainty.final_label == "uncertain":
                conflict_detected += 1

        per_sample.append(
            {
                "id": s["id"],
                "gold": gold,
                "baseline_pred": baseline_label,
                "proposed_pred": proposed_label,
                "uncertainty_status": result.uncertainty.status if result.uncertainty else None,
                "evidence_f1": f1,
            }
        )

    labels_order = ["phishing", "legitimate", "uncertain"]

    def metrics_block(y_pred: List[str]) -> dict:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(
                precision_score(y_true, y_pred, labels=labels_order, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_true, y_pred, labels=labels_order, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_true, y_pred, labels=labels_order, average="macro", zero_division=0)
            ),
            "confusion_matrix": {
                "labels": labels_order,
                "matrix": confusion_matrix(y_true, y_pred, labels=labels_order).tolist(),
            },
            "classification_report": classification_report(
                y_true, y_pred, labels=labels_order, zero_division=0, output_dict=True
            ),
        }

    avg_ev_p = float(np.mean([x["precision"] for x in evidence_prs])) if evidence_prs else 0.0
    avg_ev_r = float(np.mean([x["recall"] for x in evidence_prs])) if evidence_prs else 0.0
    avg_ev_f1 = float(np.mean([x["f1"] for x in evidence_prs])) if evidence_prs else 0.0

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(samples),
        "label_distribution": dict(Counter(y_true)),
        "baseline": {
            "description": "Ordinary TF-IDF classification without evidence-validation override.",
            "metrics": metrics_block(y_pred_baseline),
        },
        "proposed": {
            "description": "Evidence-aware framework with validation and uncertainty handling.",
            "metrics": metrics_block(y_pred_proposed),
        },
        "evidence_metrics": {
            "span_precision_avg": avg_ev_p,
            "span_recall_avg": avg_ev_r,
            "span_f1_avg": avg_ev_f1,
            "per_sample": evidence_prs,
            "existence_validation_accuracy": (
                float(existence_correct / existence_total) if existence_total else None
            ),
            "existence_total": existence_total,
            "fabricated_evidence_detection_rate": (
                float(unsupported_detected / unsupported_detection_cases)
                if unsupported_detection_cases
                else None
            ),
            "fabricated_cases": unsupported_detection_cases,
        },
        "uncertainty_metrics": {
            "insufficient_detection_rate": (
                float(insufficient_detected / insufficient_cases) if insufficient_cases else None
            ),
            "insufficient_cases": insufficient_cases,
            "contradictory_handling_rate": (
                float(conflict_detected / conflict_cases) if conflict_cases else None
            ),
            "contradictory_cases": conflict_cases,
        },
        "per_sample": per_sample,
        "limitations": [
            "Small synthetic evaluation set; scores are for prototype demonstration only.",
            "Not a production phishing detector; domain cues are lexical heuristics.",
            "Model confidence is not temperature-calibrated.",
            "Relevance/support use independent rule-based checks, not causal reasoning.",
        ],
    }

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "evaluation_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}")
    print(
        "Proposed F1(macro)=",
        report["proposed"]["metrics"]["f1_macro"],
        " Baseline F1(macro)=",
        report["baseline"]["metrics"]["f1_macro"],
    )
    return report


if __name__ == "__main__":
    run()
