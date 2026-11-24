# Implemented Features and Remaining Limitations

## Implemented features

| # | Feature | Status |
|---|---------|--------|
| 1 | Text input processing with original-text preservation | Done |
| 2 | Evidence identification with character offsets | Done |
| 3 | Local TF-IDF classification (+ optional LLM assist) | Done |
| 4 | Supporting / contradictory evidence roles | Done |
| 5 | Evidence existence validation | Done |
| 6 | Evidence localization | Done |
| 7 | Relevance and support checks (independent of existence) | Done |
| 8 | Unsupported explanation detection | Done |
| 9 | Uncertainty handling (insufficient / conflicting / invalid) | Done |
| 10 | Evidence-grounded explanation generation | Done |
| 11 | Streamlit main analysis UI with highlighting | Done |
| 12 | Evidence Validation Laboratory (4 adversarial tests) | Done |
| 13 | Fabricated-evidence backend rejection | Done |
| 14 | Local fallback when API key missing / API fails | Done |
| 15 | Automated pytest suite (23 tests) | Done |
| 16 | Evaluation script writing real metrics to `results/` | Done |
| 17 | Documentation + examiner demo guide | Done |

## Important technical choices

- **Domain:** phishing / suspicious message classification (demonstration only).
- **Stack:** Python 3.11, Streamlit, scikit-learn, optional OpenAI-compatible API.
- **Validation independence:** existence/localization are deterministic string checks; relevance/support use separate rule-based cue analysis — the LLM never validates its own claims alone.
- **Fallback labelling:** UI and explanations state whether outputs are local or LLM-generated.

## Remaining limitations

1. Lexical cue matching is **not** causal or logical proof.
2. Relevance/support are heuristic, not NLI/entailment models.
3. Model confidence is **not** calibrated.
4. Evaluation uses a **small synthetic** set (n=20); high proposed F1 reflects alignment with designed cues and uncertainty rules, not production performance.
5. Optional LLM path needs a configured API key; default demos use local fallback.
6. Not a production phishing detector — academic research prototype only.
7. No large public corpus is bundled (avoids licensing redistribution issues); see `docs/datasets.md` for how to extend.

## Verification performed in this build

- `python scripts/train_classifier.py` — model trained and saved  
- `python -m pytest tests -v` — **23 passed** (see `results/test_results.txt`)  
- `python evaluation/run_evaluation.py` — metrics in `results/evaluation_metrics.json`  
- Streamlit app launches at `http://localhost:8501`  
