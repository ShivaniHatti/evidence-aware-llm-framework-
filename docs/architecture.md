# Architecture — Evidence-Aware LLM Framework

## Research question

How can we make LLM-assisted text classification more transparent and trustworthy by checking whether its explanations are actually supported by evidence in the original input?

## Design principles

1. **Existence ≠ support.** A phrase appearing in the input does not prove it supports the predicted label.
2. **Independent validation.** The LLM does not validate its own output; deterministic checks run separately.
3. **Confidence ≠ sufficiency.** Model probability is distinct from evidence sufficiency.
4. **Traceability.** Every evidence reference must map to character offsets in the original input.
5. **Graceful degradation.** Missing API keys or API failures fall back to a clearly labelled local path.

## Module workflow

```
Raw text
   │
   ▼
[A] Input Processing
   │  validate · light normalize · preserve original · sentence spans
   ▼
[B] Evidence Identification
   │  cue patterns · supporting/contradictory candidates · char offsets
   ▼
[C] Classification (local sklearn and/or optional LLM)
   │  predicted label · uncalibrated confidence · unverified reasoning
   ▼
[D] Evidence Validation  ◄── most important
   │  existence · localization · relevance · support · unsupported claims
   ▼
[E] Uncertainty Handling
   │  insufficient · conflicting · invalid → may override to "uncertain"
   ▼
[F] Explanation Generation
   │  evidence-grounded narrative · backend labelling (local vs LLM)
   ▼
[G] Results Interface (Streamlit)
```

## Adversarial Validation Laboratory

| Test | Purpose |
|------|---------|
| Valid evidence | Genuine spans pass existence/support checks |
| Fabricated evidence | Invented phrase is rejected by backend validator |
| Insufficient evidence | Sparse text yields uncertain status |
| Contradictory evidence | Both cue types identified; forced certainty avoided |

## Technology stack

- Python 3.11+
- Streamlit UI
- scikit-learn TF-IDF + Logistic Regression (local classifier)
- Optional OpenAI-compatible LLM via environment variables
- pytest for automated tests

## Domain note

Demonstration domain: phishing / suspicious message classification with labels `phishing`, `legitimate`, `uncertain`. This is an academic prototype, not a production security product.
