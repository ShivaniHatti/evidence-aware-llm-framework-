# Evidence-Aware LLM Framework for Explainable Text Classification

Academic NLP research prototype that classifies text **and** verifies whether explanations are supported by evidence in the original input.

**Demonstration domain:** phishing / suspicious messages (`phishing` | `legitimate` | `uncertain`).

> This is **not** a production-grade phishing detection system.

## Research question

How can we make LLM-assisted text classification more transparent and trustworthy by checking whether its explanations are actually supported by evidence in the original input?

## What this project does

1. Text input processing  
2. Relevant evidence identification (with character offsets)  
3. LLM-assisted or local text classification  
4. Supporting and contradictory evidence identification  
5. Evidence existence & localization validation  
6. Evidence relevance and support validation  
7. Unsupported explanation detection  
8. Uncertainty handling  
9. Evidence-grounded explanation generation  
10. Clear Streamlit presentation + Evidence Validation Laboratory  

## Project structure

```
NLP/
├── app/streamlit_app.py          # UI
├── src/
│   ├── preprocessing/            # Input processing
│   ├── evidence/                 # Evidence identification
│   ├── classification/           # Local (+ optional LLM) classifier
│   ├── validation/               # Independent evidence checks
│   ├── uncertainty/              # Insufficient / conflicting evidence
│   ├── explanation/              # Grounded explanations
│   ├── llm/                      # API client + local fallback
│   └── pipeline/                 # End-to-end orchestrator
├── data/demo/                    # Examiner demonstration samples
├── data/evaluation/              # Train/eval synthetic sets
├── evaluation/run_evaluation.py  # Metrics (writes real results)
├── tests/                        # Automated tests
├── results/                      # Models + evaluation outputs
├── docs/                         # Architecture & demo guide
├── scripts/train_classifier.py
├── requirements.txt
└── .env.example
```

## Quick start

Requires **Python 3.11+** (recommended).

```powershell
cd "c:\Users\Shivani R Hatti\OneDrive\Desktop\NLP"
python -m pip install -r requirements.txt
python scripts/train_classifier.py
python -m pytest tests -v
python evaluation/run_evaluation.py
python -m streamlit run app/streamlit_app.py
```

### Optional LLM configuration

1. Copy `.env.example` to `.env`
2. Set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, `OPENAI_MODEL`)
3. Set `LLM_MODE=local` to force the deterministic local path

Without a key, the system uses the **local fallback** and labels outputs as locally generated.

## Evidence Validation Laboratory

In the Streamlit app, open **Evidence Validation Laboratory** to run:

| Test | Backend behaviour |
|------|-------------------|
| Valid evidence | Genuine spans validated against input |
| Fabricated evidence | Invented phrase rejected with reason |
| Insufficient evidence | Returns uncertain / insufficient status |
| Contradictory evidence | Both sides shown; avoids forced certainty |

## Evaluation

```powershell
python evaluation/run_evaluation.py
```

Metrics are written to `results/evaluation_metrics.json`. Scores are computed on a small synthetic set for reproducibility — they are not fabricated and should not be over-interpreted.

Compares:

- **Baseline:** ordinary classification without evidence-validation override  
- **Proposed:** classification + evidence validation + uncertainty handling  

## Testing

```powershell
python -m pytest tests -v
```

Covers empty/valid input, evidence extraction, exact matching, fabricated rejection, invalid positions, contradictory/insufficient evidence, explanation traceability, API failure, and local fallback.

## Documentation

- [Architecture](docs/architecture.md)
- [Datasets](docs/datasets.md)
- [Examiner demo guide](docs/examiner_demo_guide.md)

## Implemented features

- Modular evidence-aware pipeline with independent validation  
- Character-offset evidence highlighting in the UI  
- Adversarial laboratory (including fabricated-evidence rejection)  
- Local sklearn classifier + optional LLM assist  
- Graceful API failure / missing-key fallback  
- Automated tests and reproducible evaluation script  
- Academic documentation and live-demo guide  

## Limitations

- Lexical cue matching is not causal/logical proof of phishing  
- Relevance/support checks are rule-based, not deep semantic entailment  
- Model confidence is uncalibrated  
- Evaluation data is small and synthetic  
- Optional LLM path requires a configured API key  
- Not suitable for production security decisions  

## License / academic use

Original code and synthetic datasets created for this academic project. See `docs/datasets.md` for data notes.
