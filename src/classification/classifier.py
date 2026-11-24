"""Classification module: predict category with structured reasoning (not verified evidence)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evidence.extractor import EvidenceBundle


LABELS = ("phishing", "legitimate", "uncertain")


@dataclass
class ClassificationResult:
    predicted_label: str
    confidence: Optional[float]
    confidence_is_calibrated: bool
    reasoning_summary: Dict
    model_backend: str  # "sklearn_local" | "llm" | "hybrid"
    class_probabilities: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "confidence_is_calibrated": self.confidence_is_calibrated,
            "reasoning_summary": self.reasoning_summary,
            "model_backend": self.model_backend,
            "class_probabilities": self.class_probabilities,
        }


class TextClassifier:
    """
    Local TF-IDF + Logistic Regression classifier with optional LLM override.

    Confidence from predict_proba is reported as an estimate, not calibrated certainty.
    Evidence sufficiency is handled separately by the uncertainty module.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None
        if model_path and Path(model_path).exists():
            self.pipeline = joblib.load(model_path)
        else:
            self.pipeline = self._build_default_pipeline()

    def _build_default_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=5000,
                        lowercase=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(self, texts: List[str], labels: List[str]) -> "TextClassifier":
        # Train only on phishing/legitimate; uncertain is reserved for evidence rules.
        filtered = [(t, y) for t, y in zip(texts, labels) if y in ("phishing", "legitimate")]
        if not filtered:
            raise ValueError("Need at least one phishing/legitimate training example.")
        X, y = zip(*filtered)
        self.pipeline.fit(list(X), list(y))
        return self

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    def predict(
        self,
        text: str,
        evidence: Optional[EvidenceBundle] = None,
        llm_suggestion: Optional[Dict] = None,
    ) -> ClassificationResult:
        """
        Predict label. If llm_suggestion is provided and valid, merge cautiously;
        otherwise use local sklearn model + evidence heuristics.
        """
        if llm_suggestion and llm_suggestion.get("label") in ("phishing", "legitimate", "uncertain"):
            label = llm_suggestion["label"]
            conf = llm_suggestion.get("confidence")
            probs = llm_suggestion.get("probabilities") or {}
            reasoning = {
                "source": "llm",
                "summary": llm_suggestion.get("reasoning", ""),
                "note": "LLM reasoning is unverified until evidence validation.",
            }
            return ClassificationResult(
                predicted_label=label,
                confidence=float(conf) if conf is not None else None,
                confidence_is_calibrated=False,
                reasoning_summary=reasoning,
                model_backend="llm",
                class_probabilities={k: float(v) for k, v in probs.items()},
            )

        # Local path
        assert self.pipeline is not None
        # If pipeline not fitted, use evidence-only heuristic.
        if not hasattr(self.pipeline.named_steps["clf"], "classes_"):
            return self._evidence_heuristic(text, evidence)

        proba = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.named_steps["clf"].classes_)
        class_probs = {c: float(proba[i]) for i, c in enumerate(classes)}
        for lab in ("phishing", "legitimate"):
            class_probs.setdefault(lab, 0.0)

        pred_idx = int(np.argmax(proba))
        pred = classes[pred_idx]
        conf = float(proba[pred_idx])

        # Soft evidence adjustment: if strong conflicting cues, prefer uncertain.
        phish_n = len(evidence.supporting_candidates("phishing")) if evidence else 0
        legit_n = len(evidence.supporting_candidates("legitimate")) if evidence else 0

        reasoning = {
            "source": "sklearn_local",
            "summary": (
                f"Local classifier predicted '{pred}' with raw probability {conf:.3f}. "
                f"Phishing cues found: {phish_n}; legitimate cues found: {legit_n}."
            ),
            "phishing_cue_count": phish_n,
            "legitimate_cue_count": legit_n,
            "note": "Model confidence is not evidence sufficiency.",
        }

        # If both cue types present strongly, mark uncertain at classification stage.
        if phish_n >= 1 and legit_n >= 1 and abs(phish_n - legit_n) <= 1:
            return ClassificationResult(
                predicted_label="uncertain",
                confidence=None,
                confidence_is_calibrated=False,
                reasoning_summary={
                    **reasoning,
                    "summary": reasoning["summary"]
                    + " Conflicting cues present; provisional label set to uncertain.",
                },
                model_backend="sklearn_local",
                class_probabilities=class_probs,
            )

        # No cues and low confidence -> uncertain
        if phish_n == 0 and legit_n == 0 and conf < 0.6:
            return ClassificationResult(
                predicted_label="uncertain",
                confidence=conf,
                confidence_is_calibrated=False,
                reasoning_summary={
                    **reasoning,
                    "summary": reasoning["summary"]
                    + " Few lexical cues and modest model probability; provisional uncertain.",
                },
                model_backend="sklearn_local",
                class_probabilities=class_probs,
            )

        return ClassificationResult(
            predicted_label=pred,
            confidence=conf,
            confidence_is_calibrated=False,
            reasoning_summary=reasoning,
            model_backend="sklearn_local",
            class_probabilities=class_probs,
        )

    def _evidence_heuristic(
        self, text: str, evidence: Optional[EvidenceBundle]
    ) -> ClassificationResult:
        phish_n = len(evidence.supporting_candidates("phishing")) if evidence else 0
        legit_n = len(evidence.supporting_candidates("legitimate")) if evidence else 0
        urgency = bool(re.search(r"urgent|immediately|act now", text, re.I))

        if phish_n > legit_n and phish_n > 0:
            label = "phishing"
            conf = min(0.5 + 0.1 * phish_n + (0.1 if urgency else 0), 0.95)
        elif legit_n > phish_n and legit_n > 0:
            label = "legitimate"
            conf = min(0.5 + 0.1 * legit_n, 0.9)
        else:
            label = "uncertain"
            conf = None

        return ClassificationResult(
            predicted_label=label,
            confidence=conf,
            confidence_is_calibrated=False,
            reasoning_summary={
                "source": "evidence_heuristic",
                "summary": f"Unfitted model; used cue counts (phish={phish_n}, legit={legit_n}).",
                "phishing_cue_count": phish_n,
                "legitimate_cue_count": legit_n,
            },
            model_backend="sklearn_local",
            class_probabilities={},
        )
