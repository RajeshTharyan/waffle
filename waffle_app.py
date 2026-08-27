# waffle_app.py
# Streamlit UI for The Waffle Cube. Scoring lives in the `waffle` package.
# Authors: Haku Rajesh, Rajesh Tharyan, Insight Companion
# License: MIT

from __future__ import annotations

import streamlit as st

from waffle.embeddings import get_backend
from waffle.labels import (
    component_analyses,
    label_actionability,
    label_focus,
    label_substance,
    label_waffle,
    pick_score_tagline,
    pick_tagline,
)
from waffle.plotting import build_cube_figure
from waffle.scoring import compute_features

st.set_page_config(page_title="The Waffle Cube", page_icon="🧇", layout="wide")
st.title("🧇 The Waffle Cube")
st.markdown(
    '<div class="subtitle">Operationalising Obfuscation: An Empirical Approach to Waffle Intensity.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* Metric value and label sizes */
    div[data-testid="stMetricValue"] > div { font-size: 1.1rem; line-height: 1.2; color: #ffffff; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem; line-height: 1.1; color: #ffffff; }
    /* Description paragraph */
    .desc { font-size: 0.95rem; color: #ffffff; }
    /* Subtitle under title */
    .subtitle { font-size: 1.4rem; color: #ffffff; font-style: italic; margin: -0.2rem 0 0.4rem 0; }
    /* Results headline */
    .tagline { font-size: 1.4rem; color: #ffffff; font-weight: 700; margin: 0.25rem 0 0.4rem 0; }
    .headline-score { font-size: 1.35rem; font-weight: 700; margin: 0.1rem 0; }
    .headline { font-size: 1.2rem; font-weight: 700; margin: 0 0 0.6rem 0; color: #ffffff; }
    /* Full-bleed divider */
    hr.full-bleed { border: none; border-top: 2px solid #e0e0e0; margin: 0.4rem 0 0.9rem 0; width: 100vw; position: relative; left: 50%; right: 50%; margin-left: -50vw; margin-right: -50vw; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<hr class="full-bleed" />', unsafe_allow_html=True)
st.markdown(
    '<p class="desc">Measure <em>waffle</em> across <strong>Substance</strong>, <strong>Focus</strong>, and <strong>Actionability</strong>. Upload a document or paste text.</p>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload a .txt or .md document", type=["txt", "md"])
    prompt = st.text_input(
        "Prompt / Question (improves the Focus metric)",
        value="Summarise the strategy in three points.",
    )

with col2:
    sample = st.text_area(
        "Or paste text here",
        height=300,
        help="You can combine this with an uploaded file; both will be analysed.",
    )

text = ""
if uploaded is not None:
    try:
        text += uploaded.read().decode("utf-8", errors="ignore")
    except (OSError, AttributeError, UnicodeError):
        st.warning("Could not decode file as UTF-8.")

text += "\n" + sample if sample else ""
text = text.strip()

st.caption(f"Embeddings backend: **{get_backend()}**")


def _choose_tagline() -> str:
    recent = st.session_state.get("recent_taglines", [])
    selected, recent = pick_tagline(recent)
    st.session_state["recent_taglines"] = recent
    return selected


def _choose_tagline_for_score(score: float, avoid: str = "") -> str:
    recent = st.session_state.get("recent_verdict_taglines", [])
    selected, recent = pick_score_tagline(score, recent, avoid=avoid)
    st.session_state["recent_verdict_taglines"] = recent
    return selected


if st.button("Analyse") or (text and len(text) > 10):
    if not text:
        st.error("Please upload or paste some text.")
    else:
        feats = compute_features(text, prompt)
        s_label = label_substance(feats["S"])
        f_label = label_focus(feats["F"])
        a_label = label_actionability(feats["A"])
        w_label = label_waffle(feats["WaffleScore"])
        tagline_text = _choose_tagline()
        st.markdown(
            f'<div class="tagline">Congratulations — {tagline_text}. Waffle score: {w_label}.</div>',
            unsafe_allow_html=True,
        )
        verdict_tag = _choose_tagline_for_score(feats["WaffleScore"], avoid=tagline_text)
        st.markdown(f'<div class="headline">Verdict: {verdict_tag}</div>', unsafe_allow_html=True)
        left_col, right_col = st.columns([1, 1])
        with left_col:
            m1, m2 = st.columns(2)
            m1.metric("Meatiness Quotient (Substance, S)", f"{feats['S']:.2f} — {s_label}")
            m2.metric("Laser Aim (Focus, F)", f"{feats['F']:.2f} — {f_label}")
            m3, m4 = st.columns(2)
            m3.metric("Get‑Stuff‑Done Quotient (Actionability, A)", f"{feats['A']:.2f} — {a_label}")
            m4.metric("Waffle Score (↑ = more waffle)", f"{feats['WaffleScore']:.2f} — {w_label}")
            expl = component_analyses(feats)
            st.markdown("<hr style='margin: 0.6rem 0' />", unsafe_allow_html=True)
            st.markdown(f"<div class='desc'><strong>Substance:</strong> {expl['S']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='desc'><strong>Focus:</strong> {expl['F']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='desc'><strong>Actionability:</strong> {expl['A']}</div>", unsafe_allow_html=True)
            st.subheader("Feature Diagnostics")
            with st.expander("Show raw feature values"):
                st.json({k: v for k, v in feats.items() if k not in ["backend"]})

        with right_col:
            try:
                fig = build_cube_figure(feats["S"], feats["F"], feats["A"])
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info("Install plotly for a 3D scatter: `pip install plotly`.")
else:
    st.info("Upload or paste text, optionally add the prompt, then click **Analyse**.")
