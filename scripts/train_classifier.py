"""Train the local TF-IDF + Logistic Regression classifier and save the model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classification.classifier import TextClassifier


def main() -> None:
    train_path = ROOT / "data" / "evaluation" / "train_dataset.json"
    model_path = ROOT / "results" / "models" / "tfidf_logreg.joblib"

    with open(train_path, encoding="utf-8") as f:
        data = json.load(f)

    texts = [s["text"] for s in data["samples"]]
    labels = [s["label"] for s in data["samples"]]

    clf = TextClassifier()
    clf.fit(texts, labels)
    clf.save(model_path)
    print(f"Trained on {len(texts)} samples.")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
