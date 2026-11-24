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
    
    # Dashboard Header with Summary Cards
    st.markdown("### 📊 Analysis Dashboard")
    st.markdown("---")
    
    # Summary Cards Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        label_color = "#51cf66" if final_label == "legitimate" else "#ff6b6b" if final_label == "phishing" else "#ffd43b"
        st.markdown(f"""
        <div style='background: white; padding: 20px; border-radius: 15px; border-left: 5px solid {label_color}; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
            <div style='color: #718096; font-size: 12px; font-weight: 600; margin-bottom: 5px;'>CLASSIFICATION</div>
            <div style='color: {label_color}; font-size: 24px; font-weight: 700;'>{final_label or '—'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        conf = classification.get("confidence")
        conf_color = "#51cf66" if conf and conf > 0.7 else "#ffd43b" if conf and conf > 0.5 else "#ff6b6b"
        st.markdown(f"""
        <div style='background: white; padding: 20px; border-radius: 15px; border-left: 5px solid #26c6da; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
            <div style='color: #718096; font-size: 12px; font-weight: 600; margin-bottom: 5px;'>CONFIDENCE</div>
            <div style='color: {conf_color}; font-size: 24px; font-weight: 700;'>{f'{conf:.2f}' if conf is not None else 'n/a'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        ev_count = len(result_dict.get("evidence_spans", []))
        st.markdown(f"""
        <div style='background: white; padding: 20px; border-radius: 15px; border-left: 5px solid #ab47bc; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
            <div style='color: #718096; font-size: 12px; font-weight: 600; margin-bottom: 5px;'>EVIDENCE SPANS</div>
            <div style='color: #ab47bc; font-size: 24px; font-weight: 700;'>{ev_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        is_uncertain = uncertainty.get("is_uncertain", False)
        status_color = "#ff6b6b" if is_uncertain else "#51cf66"
        status_text = "UNCERTAIN" if is_uncertain else "CERTAIN"
        st.markdown(f"""
        <div style='background: white; padding: 20px; border-radius: 15px; border-left: 5px solid {status_color}; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
            <div style='color: #718096; font-size: 12px; font-weight: 600; margin-bottom: 5px;'>STATUS</div>
            <div style='color: {status_color}; font-size: 24px; font-weight: 700;'>{status_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Backend Info
    backend = classification.get("model_backend", "unknown")
    llm_used = intermediate.get("llm_used", False)
    st.markdown(f"""
    <div style='background: #f8fafc; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0;'>
        <strong>🔧 Backend:</strong> <code>{backend}</code> · 
        <strong>🤖 LLM Used:</strong> <span style='color: {"#51cf66" if llm_used else "#adb5bd"}'>{str(llm_used)}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Uncertainty Details
    st.markdown("### ⚠️ Uncertainty Analysis")
    status = uncertainty.get("status", "—")
    if uncertainty.get("is_uncertain"):
        st.markdown(f"""
        <div style='background: #fff3cd; padding: 15px; border-radius: 10px; border-left: 4px solid #ffc107;'>
            <strong>⚠️ Uncertain Status:</strong> <code>{status}</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background: #d4edda; padding: 15px; border-radius: 10px; border-left: 4px solid #28a745;'>
            <strong>✅ Evidence Sufficient:</strong> <code>{status}</code>
        </div>
        """, unsafe_allow_html=True)
    
    if uncertainty.get("reasons"):
        st.markdown("**Reasons:**")
        for reason in uncertainty.get("reasons"):
            st.markdown(f"• {reason}")

    # Evidence Highlights Section
    st.markdown("### 🔍 Evidence Analysis")
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("#### 📝 Original Input with Highlights")
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
        st.markdown("""
        <div style='background: #f8fafc; padding: 10px; border-radius: 8px; font-size: 12px; color: #718096;'>
            <span style='color: #51cf66;'>●</span> Supports · 
            <span style='color: #ffd43b;'>●</span> Contradicts · 
            <span style='color: #adb5bd;'>●</span> Neutral · 
            <span style='color: #ff6b6b;'>●</span> Rejected
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown("#### 📊 Evidence Stats")
        ev = result_dict.get("evidence_spans") or []
        support = validation.get("valid_supporting") or []
        contra = validation.get("valid_contradictory") or []
        rejected = validation.get("rejected_spans") or []
        
        st.markdown(f"""
        <div style='background: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
            <div style='color: #51cf66; font-size: 28px; font-weight: 700;'>{len(support)}</div>
            <div style='color: #718096; font-size: 12px;'>Supporting</div>
        </div>
        <div style='background: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
            <div style='color: #ffd43b; font-size: 28px; font-weight: 700;'>{len(contra)}</div>
            <div style='color: #718096; font-size: 12px;'>Contradictory</div>
        </div>
        <div style='background: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
            <div style='color: #ff6b6b; font-size: 28px; font-weight: 700;'>{len(rejected)}</div>
            <div style='color: #718096; font-size: 12px;'>Rejected</div>
        </div>
        <div style='background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
            <div style='color: #26c6da; font-size: 28px; font-weight: 700;'>{validation.get('existence_pass_rate', 0):.0%}</div>
            <div style='color: #718096; font-size: 12px;'>Pass Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Evidence Table
    st.markdown("#### 📋 Evidence Identification")
    ev = result_dict.get("evidence_spans") or []
    if ev:
        st.dataframe(
            [
                {
                    "📄 Text": e["text"],
                    "📍 Position": f"[{e['start']}:{e['end']}]",
                    "🏷️ Type": e["cue_type"],
                    "🎯 Role": e["role_hint"],
                    "🔗 Source": e["source"],
                }
                for e in ev
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No lexical evidence cues identified.")
    
    # Validation Status
    st.markdown("### ✅ Validation Status")
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.markdown("#### ✨ Supporting Evidence")
        if support:
            for r in support:
                st.markdown(f"""
                <div style='background: #d4edda; padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #28a745;'>
                    <strong>"{r['span']['text']}"</strong><br>
                    <small style='color: #155724;'>Position: [{r['actual_start']}:{r['actual_end']}]</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<em style='color: #adb5bd;'>None</em>", unsafe_allow_html=True)
    
    with col_v2:
        st.markdown("#### ⚠️ Contradictory Evidence")
        if contra:
            for r in contra:
                st.markdown(f"""
                <div style='background: #fff3cd; padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #ffc107;'>
                    <strong>"{r['span']['text']}"</strong><br>
                    <small style='color: #856404;'>Position: [{r['actual_start']}:{r['actual_end']}]</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<em style='color: #adb5bd;'>None</em>", unsafe_allow_html=True)
    
    # Rejected Evidence
    if rejected:
        st.markdown("#### ❌ Rejected Evidence")
        for r in rejected:
            st.markdown(f"""
            <div style='background: #f8d7da; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #dc3545;'>
                <strong>"{r['span']['text']}"</strong><br>
                <small style='color: #721c24;'>Reason: {r.get('rejection_reason')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Unsupported Claims
    unsupported = validation.get("unsupported_claims") or explanation.get("unsupported_claims") or []
    st.markdown("### 🚨 Unsupported Claims")
    if unsupported:
        for c in unsupported:
            st.markdown(f"""
            <div style='background: #fff3cd; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #ffc107;'>
                <strong>⚠️ Unsupported:</strong> "{c}"
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: #d4edda; padding: 12px; border-radius: 8px; border-left: 4px solid #28a745;'>
            <strong>✅</strong> No unsupported claims detected
        </div>
        """, unsafe_allow_html=True)
    
    # Final Explanation
    st.markdown("### 💬 Final Explanation")
    gen = explanation.get("generation_backend", "local_template")
    st.markdown(f"<small style='color: #718096;'>Backend: <code>{gen}</code></small>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background: white; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; line-height: 1.6;'>
        {explanation.get("final_explanation") or "—"}
    </div>
    """, unsafe_allow_html=True)
    
    # Expandable Details
    with st.expander("🔧 Intermediate Processing Details"):
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
