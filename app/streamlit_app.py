"""
Evidence-Aware LLM Framework — academic demonstration UI.

Displays genuine backend analysis results. Does not hardcode model outputs.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.framework import EvidenceAwareFramework
from src.llm.client import LLMClient


st.set_page_config(
    page_title="Evidence-Aware LLM Framework",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom colorful theme
st.markdown("""
<style>
   .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #f3e5f5 100%);
    }
    .main {
        background-color: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 30px;
        margin: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    h1 {
        color: #4a5568;
        font-weight: 800;
        background: linear-gradient(90deg, #26c6da, #ab47bc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2, h3 {
        color: #2d3748;
        font-weight: 700;
    }
    .stButton>button {
        background: linear-gradient(90deg, #26c6da, #ab47bc);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(38, 198, 218, 0.4);
    }
    .stTextInput>div>div>input, .stTextArea>div>textarea {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>textarea:focus {
        border-color: #26c6da;
        box-shadow: 0 0 0 3px rgba(38, 198, 218, 0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        color: #26c6da;
        font-weight: 700;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #26c6da 0%, #ab47bc 100%);
    }
    .sidebar .sidebar-content h1 {
        color: white !important;
        -webkit-text-fill-color: white !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_framework() -> EvidenceAwareFramework:
    model_path = ROOT / "results" / "models" / "tfidf_logreg.joblib"
    return EvidenceAwareFramework(
        model_path=model_path if model_path.exists() else None,
        llm_client=LLMClient(),
        use_llm=True,
    )


def load_demo_samples() -> dict:
    path = ROOT / "data" / "demo" / "demonstration_samples.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def highlight_spans(text: str, spans: list) -> str:
    """Highlight evidence phrases in original text using ordered non-overlapping marks."""
    if not spans:
        return f"<pre style='white-space:pre-wrap;font-family:Georgia,serif;background:#f8fafc;padding:20px;border-radius:10px;'>{html.escape(text)}</pre>"

    intervals = []
    for s in spans:
        start = s.get("actual_start", s.get("start"))
        end = s.get("actual_end", s.get("end"))
        support = s.get("support", "neutral")
        exists = s.get("exists_in_input", True)
        if start is None or end is None:
            continue
        if not exists:
            color = "#ff6b6b"  # Vibrant red for rejected
        elif support == "supports":
            color = "#51cf66"  # Vibrant green for supports
        elif support == "contradicts":
            color = "#ffd43b"  # Vibrant yellow for contradicts
        else:
            color = "#adb5bd"  # Grey for neutral
        intervals.append((start, end, color))

    intervals.sort(key=lambda x: x[0])
    merged = []
    for iv in intervals:
        if merged and iv[0] < merged[-1][1]:
            continue
        merged.append(iv)

    parts = []
    cursor = 0
    for start, end, color in merged:
        if start < cursor:
            continue
        parts.append(html.escape(text[cursor:start]))
        parts.append(
            f"<mark style='background:{color};padding:0 2px;'>"
            f"{html.escape(text[start:end])}</mark>"
        )
        cursor = end
    parts.append(html.escape(text[cursor:]))
    return (
        "<pre style='white-space:pre-wrap;font-family:Georgia,serif;line-height:1.8;background:#f8fafc;padding:25px;border-radius:15px;border:2px solid #e2e8f0;'>"
        + "".join(parts)
        + "</pre>"
    )


def render_analysis(result_dict: dict) -> None:
    if not result_dict.get("input_valid"):
        st.error(result_dict.get("error_message") or "Invalid input.")
        return

    classification = result_dict.get("classification") or {}
    validation = result_dict.get("validation") or {}
    uncertainty = result_dict.get("uncertainty") or {}
    explanation = result_dict.get("explanation") or {}
    llm = result_dict.get("llm") or {}
    intermediate = result_dict.get("intermediate") or {}

    final_label = uncertainty.get("final_label") or classification.get("predicted_label")
    st.subheader("Classification Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Final category", final_label or "—")
    conf = classification.get("confidence")
    c2.metric(
        "Model confidence (uncalibrated)",
        f"{conf:.2f}" if conf is not None else "n/a",
    )
    c3.metric("Evidence sufficiency", uncertainty.get("evidence_sufficiency", "—"))

    backend = classification.get("model_backend", "unknown")
    llm_used = intermediate.get("llm_used", False)
    st.caption(
        f"Classification backend: `{backend}`. "
        f"LLM used: **{llm_used}**. "
        + (
            f"LLM note: {llm.get('error')}"
            if llm and llm.get("error")
            else "Outputs clearly distinguish local vs LLM generation."
        )
    )

    st.subheader("Uncertainty Indicator")
    status = uncertainty.get("status", "—")
    if uncertainty.get("is_uncertain"):
        st.warning(f"Uncertain — status: `{status}`")
    else:
        st.success(f"Evidence sufficient — status: `{status}`")
    for reason in uncertainty.get("reasons") or []:
        st.write(f"- {reason}")

    st.subheader("Original Input with Evidence Highlights")
    span_results = validation.get("span_results") or []
    highlight_payload = []
    for r in span_results:
        span = r.get("span") or {}
        highlight_payload.append(
            {
                "start": span.get("start"),
                "end": span.get("end"),
                "actual_start": r.get("actual_start"),
                "actual_end": r.get("actual_end"),
                "support": r.get("support"),
                "exists_in_input": r.get("exists_in_input"),
            }
        )
    st.markdown(
        highlight_spans(result_dict.get("normalized_text", ""), highlight_payload),
        unsafe_allow_html=True,
    )
    st.caption("Green = supports · Yellow = contradicts · Grey = neutral · Red = rejected/missing")

    st.subheader("Evidence Identification")
    ev = result_dict.get("evidence_spans") or []
    if ev:
        st.dataframe(
            [
                {
                    "text": e["text"],
                    "positions": f"[{e['start']}:{e['end']}]",
                    "cue_type": e["cue_type"],
                    "role_hint": e["role_hint"],
                    "source": e["source"],
                }
                for e in ev
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No lexical evidence cues identified.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Supporting evidence (validated)**")
        support = validation.get("valid_supporting") or []
        if support:
            for r in support:
                st.write(f"- \"{r['span']['text']}\" @ [{r['actual_start']}:{r['actual_end']}]")
        else:
            st.write("_None_")
    with col_b:
        st.markdown("**Contradictory evidence (validated)**")
        contra = validation.get("valid_contradictory") or []
        if contra:
            for r in contra:
                st.write(f"- \"{r['span']['text']}\" @ [{r['actual_start']}:{r['actual_end']}]")
        else:
            st.write("_None_")

    st.subheader("Evidence Validation Status")
    st.write(
        f"Existence pass rate: **{validation.get('existence_pass_rate', 0):.0%}** · "
        f"All evidence valid: **{validation.get('all_evidence_valid')}**"
    )
    rejected = validation.get("rejected_spans") or []
    if rejected:
        st.markdown("**Rejected evidence**")
        for r in rejected:
            st.error(f"\"{r['span']['text']}\" — {r.get('rejection_reason')}")

    unsupported = validation.get("unsupported_claims") or explanation.get("unsupported_claims") or []
    st.subheader("Unsupported Evidence Warnings")
    if unsupported:
        for c in unsupported:
            st.warning(f"Unsupported claim: \"{c}\"")
    else:
        st.write("No unsupported claims detected in the draft explanation.")

    st.subheader("Final Explanation")
    gen = explanation.get("generation_backend", "local_template")
    st.caption(f"Explanation generation backend: `{gen}`")
    st.write(explanation.get("final_explanation") or "—")

    with st.expander("Intermediate processing (examiner view)"):
        st.json(
            {
                "intermediate": intermediate,
                "classification": classification,
                "validation_summary": {
                    "existence_pass_rate": validation.get("existence_pass_rate"),
                    "n_supporting": len(support),
                    "n_contradictory": len(contra),
                    "n_rejected": len(rejected),
                    "unsupported_claims": unsupported,
                },
                "uncertainty": uncertainty,
            }
        )


def page_main(fw: EvidenceAwareFramework) -> None:
    st.markdown("### 🔍 Evidence-Aware LLM Framework")
    st.markdown(
        "<div style='background: linear-gradient(90deg, #667eea, #764ba2); "
        "padding: 20px; border-radius: 15px; margin-bottom: 20px;'>"
        "<h3 style='color: white; margin: 0;'>🎯 Explainable and Evidence-Verified Text Classification</h3>"
        "<p style='color: rgba(255,255,255,0.9); margin: 10px 0 0 0;'>"
        "Academic research prototype — phishing/suspicious-message demonstration domain. "
        "Not a production security product."
        "</p></div>",
        unsafe_allow_html=True,
    )

    demos = load_demo_samples()
    sample_options = {
        f"📋 {s['id']} ({s['purpose']})": s["text"] for s in demos["samples"]
    }
    choice = st.selectbox(
        "🎨 Load a demonstration sample (optional)",
        ["—"] + list(sample_options.keys()),
    )
    default_text = sample_options.get(choice, "") if choice != "—" else ""

    st.markdown("### 📝 Text Input")
    text = st.text_area(
        "Enter your message below...",
        value=default_text,
        height=180,
        placeholder="Paste a message to analyze…",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Analyze Text", type="primary", use_container_width=True):
            with st.spinner("🔄 Running evidence-aware analysis…"):
                result = fw.analyze(text)
                st.session_state["last_result"] = result.as_dict()

    if "last_result" in st.session_state:
        st.markdown("---")
        render_analysis(st.session_state["last_result"])


def page_lab(fw: EvidenceAwareFramework) -> None:
    st.markdown("### 🧪 Evidence Validation Laboratory")
    st.markdown(
        "<div style='background: linear-gradient(90deg, #f093fb, #f5576c); "
        "padding: 20px; border-radius: 15px; margin-bottom: 20px;'>"
        "<h3 style='color: white; margin: 0;'>🔬 Adversarial Validation Demonstrations</h3>"
        "<p style='color: rgba(255,255,255,0.9); margin: 10px 0 0 0;'>"
        "Each test calls the backend validator; results are not hardcoded success messages."
        "</p></div>",
        unsafe_allow_html=True,
    )
    demos = load_demo_samples()
    by_purpose = {s["purpose"]: s for s in demos["samples"]}
    # Also index by id for fabricated base
    by_id = {s["id"]: s for s in demos["samples"]}

    test = st.radio(
        "🎯 Select laboratory test",
        [
            "✅ Valid evidence test",
            "❌ Fabricated evidence test",
            "⚠️ Insufficient evidence test",
            "🔄 Contradictory evidence test",
        ],
    )

    if test == "Valid evidence test":
        sample = by_purpose.get("valid_evidence") or by_id.get("demo_phish_01")
        text = st.text_area("Input", value=sample["text"], height=140)
        if st.button("Run valid evidence test"):
            out = fw.lab_valid_evidence(text)
            st.session_state["lab_out"] = out

    elif test == "Fabricated evidence test":
        sample = by_id["demo_fabricated_base"]
        text = st.text_area("Original input", value=sample["text"], height=120)
        fake = st.text_input(
            "Fabricated evidence phrase (must NOT appear in input)",
            value=sample.get(
                "fabricated_phrase",
                "wire transfer of $50,000 to an overseas bitcoin wallet",
            ),
        )
        if st.button("Run fabricated evidence test"):
            out = fw.lab_fabricated_evidence(text, fake)
            st.session_state["lab_out"] = out

    elif test == "Insufficient evidence test":
        sample = by_purpose["insufficient_evidence"]
        text = st.text_area("Input", value=sample["text"], height=100)
        if st.button("Run insufficient evidence test"):
            out = fw.lab_insufficient_evidence(text)
            st.session_state["lab_out"] = out

    else:
        sample = by_purpose["contradictory_evidence"]
        text = st.text_area("Input", value=sample["text"], height=140)
        if st.button("Run contradictory evidence test"):
            out = fw.lab_contradictory_evidence(text)
            st.session_state["lab_out"] = out

    if "lab_out" in st.session_state:
        out = st.session_state["lab_out"]
        passed = out.get("passed")
        if passed:
            st.success(f"Test `{out.get('test')}` PASSED (backend result).")
        else:
            st.error(f"Test `{out.get('test')}` FAILED (backend result).")

        st.subheader("Backend response")
        if out.get("test") == "fabricated_evidence":
            st.write("**Rejection reason:**", out.get("detail"))
            st.json(
                {
                    "fabricated_phrase": out.get("fabricated_phrase"),
                    "fake_span": out.get("fake_span"),
                    "fake_validation": out.get("fake_validation"),
                }
            )
            with st.expander("Full validation report"):
                st.json(out.get("full_report"))
        elif "analysis" in out:
            render_analysis(out["analysis"])
            with st.expander("Raw laboratory payload"):
                st.json({k: v for k, v in out.items() if k != "analysis"})
        else:
            st.json(out)


def main() -> None:
    st.sidebar.markdown("# 🧭 Navigation")
    page = st.sidebar.radio(
        "📍 Section",
        ["🏠 Main Analysis", "🧪 Evidence Validation Laboratory"],
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;'>"
        "<h4 style='color: white; margin: 0 0 10px 0;'>🤔 Research Question</h4>"
        "<p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 14px;'>"
        "How can we make LLM-assisted text classification more transparent "
        "by checking whether explanations are supported by evidence in the input?"
        "</p></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("💡 Local fallback is used when no API key is configured.")

    fw = get_framework()
    if page == "🏠 Main Analysis":
        page_main(fw)
    else:
        page_lab(fw)


if __name__ == "__main__":
    main()
