from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.framework import EvidenceAwareFramework
from src.llm.client import LLMClient


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def framework(project_root: Path) -> EvidenceAwareFramework:
    model_path = project_root / "results" / "models" / "tfidf_logreg.joblib"
    return EvidenceAwareFramework(
        model_path=model_path if model_path.exists() else None,
        llm_client=LLMClient(mode="local"),
        use_llm=True,
    )


@pytest.fixture(scope="session")
def demo_samples(project_root: Path) -> dict:
    path = project_root / "data" / "demo" / "demonstration_samples.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)
