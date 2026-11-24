# Examiner Demonstration Guide (5–7 minutes)

**Title:** Evidence-Aware LLM Framework for Explainable Text Classification

**Disclaimer:** Academic prototype for phishing/suspicious-message demonstration. Not a production detector.

## Before you start

```powershell
cd "c:\Users\Shivani R Hatti\OneDrive\Desktop\NLP"
python -m pip install -r requirements.txt
python scripts/train_classifier.py
python -m streamlit run app/streamlit_app.py
```

Optional: copy `.env.example` to `.env` and set `OPENAI_API_KEY` for LLM assist. Without a key, the app uses the **local fallback** and labels outputs accordingly.

## Demo script

### 1. Normal classification with evidence (≈1 min)

- Open **Main Analysis**.
- Load sample `demo_phish_01`.
- Click **Analyze Text**.
- Show: final category, highlighted evidence in the original text, supporting spans with character positions.

### 2. Evidence validation (≈1 min)

- Expand **Intermediate processing**.
- Point out existence pass rate and validated supporting evidence.
- Emphasize that model confidence and evidence sufficiency are separate metrics.

### 3. Fabricated evidence rejection (≈1.5 min) — critical

- Go to **Evidence Validation Laboratory**.
- Select **Fabricated evidence test**.
- Keep the original security-alert message.
- Keep the fabricated phrase that does **not** appear in the input.
- Run the test.
- Show the backend rejection reason (`does not occur in the original input`).
- Emphasize this is a real validator call, not a hardcoded UI message.

### 4. Contradictory evidence (≈1 min)

- Run **Contradictory evidence test**.
- Show both phishing and legitimate cues.
- Show uncertainty status / final label `uncertain`.

### 5. Insufficient evidence (≈1 min)

- Run **Insufficient evidence test** (`Hi, are you free later today?`).
- Show insufficient-evidence status and avoidance of forced classification.

### 6. Final explanation (≈30 s)

- Return to a phishing sample on Main Analysis.
- Read the final explanation: category, supporting/contradictory evidence, validation, uncertainty.

### 7. Optional: tests & metrics (≈30 s)

```powershell
python -m pytest tests -v
python evaluation/run_evaluation.py
```

Open `results/evaluation_metrics.json` for actual computed scores (small synthetic set).

## Expected behaviours

| Scenario | Expected |
|----------|----------|
| Clear phishing cues | Phishing + validated supporting spans |
| Fabricated phrase | Rejected; exists_in_input = false |
| Conflicting cues | Uncertain; both sides listed |
| Vague short text | Uncertain; insufficient evidence |
