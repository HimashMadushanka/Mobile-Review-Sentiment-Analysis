import os
import sys
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_sentiment_details


st.set_page_config(
    page_title="Sentiment Studio | Review Analytics",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, value in {
    "review_title": "",
    "review_text": "",
    "history": [],
    "last_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def load_example(title, text):
    st.session_state.review_title = title
    st.session_state.review_text = text


def save_analysis(review, sentiment, confidence, probabilities):
    result = {
        "review": review,
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": probabilities,
        "time": datetime.now().strftime("%H:%M:%S"),
    }
    st.session_state.last_result = result
    st.session_state.history.insert(
        0,
        {
            "Time": result["time"],
            "Sentiment": sentiment,
            "Confidence": f"{confidence * 100:.1f}%",
            "Review": review[:95] + ("..." if len(review) > 95 else ""),
        },
    )
    st.session_state.history = st.session_state.history[:10]


SENTIMENT_COLORS = {
    "Negative": "#fb7185",
    "Neutral": "#818cf8",
    "Positive": "#5eead4",
}


def probability_donut(probabilities):
    """Return a compact, accessible chart of the latest class probabilities."""
    chart_data = pd.DataFrame(
        {"Sentiment": list(probabilities), "Probability": list(probabilities.values())}
    )
    return (
        alt.Chart(chart_data)
        .mark_arc(innerRadius=62, outerRadius=92, cornerRadius=5)
        .encode(
            theta=alt.Theta("Probability:Q", stack=True),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=list(SENTIMENT_COLORS),
                    range=list(SENTIMENT_COLORS.values()),
                ),
                legend=alt.Legend(orient="bottom", labelColor="#f7f4ff", title=None),
            ),
            tooltip=[
                alt.Tooltip("Sentiment:N"),
                alt.Tooltip("Probability:Q", format=".1%"),
            ],
        )
        .properties(height=255)
        .configure_view(strokeOpacity=0)
        .configure(background="transparent")
    )


def session_sentiment_chart(history):
    """Return a chart showing the sentiment mix across this browser session."""
    counts = (
        pd.DataFrame(history)["Sentiment"]
        .value_counts()
        .reindex(["Negative", "Neutral", "Positive"], fill_value=0)
        .rename_axis("Sentiment")
        .reset_index(name="Reviews")
    )
    return (
        alt.Chart(counts)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Sentiment:N", sort=["Negative", "Neutral", "Positive"], axis=alt.Axis(labelColor="#f7f4ff", title=None)),
            y=alt.Y("Reviews:Q", axis=alt.Axis(labelColor="#f7f4ff", title=None, tickMinStep=1)),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(domain=list(SENTIMENT_COLORS), range=list(SENTIMENT_COLORS.values())),
                legend=None,
            ),
            tooltip=["Sentiment:N", "Reviews:Q"],
        )
        .properties(height=250)
        .configure_view(strokeOpacity=0)
        .configure(background="transparent")
    )


st.markdown(
    """
    <style>
        :root {--text:#ffffff; --muted:#e1d9f0; --panel:#1a1333; --panel-2:#251b46; --line:#5d477e; --accent:#c4b5fd;}
        .stApp {background:radial-gradient(circle at 12% 5%,rgba(168,85,247,.24),transparent 28%),radial-gradient(circle at 88% 14%,rgba(34,211,238,.18),transparent 25%),linear-gradient(135deg,#10091f 0%,#1c1135 50%,#081b2c 100%); color:var(--text);}
        .stApp p, .stApp label, .stApp li, .stApp strong {color:var(--text);}
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color:var(--muted) !important;}
        header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {background:#10091f !important;}
        [data-testid="stHeader"] button, [data-testid="stToolbar"] button {color:#ded5ff !important;}
        .block-container {max-width:1380px; padding:1.6rem 2.5rem 3.5rem;}
        [data-testid="stSidebar"] {background:linear-gradient(180deg,#130b26,#17112d); border-right:1px solid var(--line);}
        [data-testid="stSidebar"] * {color:var(--text);}
        [data-testid="stTabs"] [data-baseweb="tab-list"] {gap:.5rem; border-bottom:1px solid var(--line);}
        [data-testid="stTabs"] button {color:#e2d9f3 !important; font-weight:700; padding:.65rem .9rem;}
        [data-testid="stTabs"] button[aria-selected="true"] {color:#ffffff !important;}
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {background-color:#c4b5fd !important;}
        h1,h2,h3 {color:#f7f9ff; letter-spacing:-.03em;}
        .app-header {display:flex; justify-content:space-between; align-items:center; padding-bottom:1.8rem; border-bottom:1px solid var(--line); margin-bottom:2rem;}
        .logo {font-size:1.15rem; font-weight:850; color:#fff;}.logo-mark {display:inline-grid; place-items:center; width:27px; height:27px; border-radius:8px; margin-right:8px; color:#fff; background:linear-gradient(135deg,#c084fc,#22d3ee); box-shadow:0 7px 22px rgba(168,85,247,.42);}
        .status {padding:.42rem .72rem; border-radius:999px; color:#86efac; background:rgba(74,222,128,.10); border:1px solid rgba(134,239,172,.28); font-size:.8rem; font-weight:700;}
        .hero {padding:1rem 0 1.7rem;}.eyebrow {font-size:.75rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; color:#d8ccff;}.hero h1 {font-size:clamp(2.2rem,4vw,3.8rem); margin:.4rem 0 .65rem; line-height:1.03;}.hero p {max-width:720px; color:var(--muted); font-size:1.06rem; line-height:1.65;}
        .metric-card {padding:1.1rem 1.2rem; background:linear-gradient(135deg,rgba(47,34,86,.92),rgba(20,38,67,.92)); border:1px solid var(--line); border-radius:14px;}.metric-number {font-size:1.55rem; color:#fff; font-weight:850;}.metric-caption {font-size:.83rem; color:var(--muted); margin-top:.18rem;}
        .panel {padding:1.45rem; border-radius:17px; background:linear-gradient(135deg,rgba(41,29,75,.96),rgba(18,30,54,.96)); border:1px solid var(--line); box-shadow:0 14px 35px rgba(6,3,18,.18);}.panel-title {font-size:1.15rem; color:#fff; font-weight:800; margin-bottom:.25rem;}.panel-subtitle {font-size:.92rem; color:var(--muted); line-height:1.55; margin-bottom:1.25rem;}
        [data-testid="stForm"] {border:0; padding:0; background:transparent;} [data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea {background:#120c26!important; color:#f7f4ff!important; border:1px solid #5b4386!important; border-radius:9px!important;} [data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus {border-color:#a78bfa!important; box-shadow:0 0 0 1px #a78bfa!important;}
        [data-testid="stFormSubmitButton"] button,[data-testid="stSidebar"] button {border:0!important; color:#fff!important; border-radius:9px!important; font-weight:750!important; background:linear-gradient(100deg,#8b5cf6,#06b6d4)!important; box-shadow:0 8px 20px rgba(91,55,190,.28)!important;}
        .workflow-step {display:flex; gap:.85rem; padding:.9rem 0; border-bottom:1px solid rgba(196,181,253,.13);}.workflow-step:last-child{border-bottom:0;}.step-number{flex:0 0 27px;height:27px;border-radius:50%;display:grid;place-items:center;background:#4c3579;color:#ede9fe;font-size:.8rem;font-weight:800}.step-name{font-weight:750;color:#f5f3ff}.step-copy{font-size:.84rem;color:var(--muted);margin-top:.14rem;line-height:1.4}
        .result {min-height:210px; border-radius:16px; padding:1.55rem; color:#fff;}.positive{background:linear-gradient(135deg,#047857,#0d9488)}.neutral{background:linear-gradient(135deg,#4338ca,#6366f1)}.negative{background:linear-gradient(135deg,#be185d,#e11d48)}.result-tag{font-size:.74rem;letter-spacing:.13em;text-transform:uppercase;font-weight:800;opacity:.82}.result-value{font-size:2.55rem;font-weight:900;letter-spacing:-.05em;margin:.55rem 0 .45rem}.result-copy{line-height:1.55;opacity:.9}.confidence{display:inline-block;margin-top:1rem;padding:.45rem .68rem;border-radius:999px;background:rgba(255,255,255,.16);font-size:.88rem;font-weight:750}
        .probability-row{margin:.85rem 0}.probability-head{display:flex;justify-content:space-between;margin-bottom:.34rem;font-size:.9rem}.probability-track{height:10px;background:#100a21;border-radius:999px;overflow:hidden}.probability-fill{height:100%;border-radius:999px}.bar-negative{background:#fb7185}.bar-neutral{background:#818cf8}.bar-positive{background:#5eead4}
        [data-testid="stMetric"]{padding:1rem;border:1px solid var(--line);border-radius:12px;background:linear-gradient(135deg,#21173e,#19112e)}[data-testid="stMetricLabel"] p{color:#e9e2f6!important;}[data-testid="stMetricValue"]{color:#fff!important;}[data-testid="stExpander"] summary,[data-testid="stExpander"] summary span{color:#fff!important;}[data-testid="stAlert"] p{color:#fff!important;}[data-testid="stDataFrame"] *{color:#1c1630;}.empty-state{padding:2.7rem 1.5rem;text-align:center;border:1px dashed #8b6ab8;border-radius:16px;color:var(--muted);background:rgba(35,24,65,.55)}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Sentiment Studio")
    st.caption("Mobile review analytics")
    st.divider()
    st.markdown("### Quick start")
    st.button("Use positive example", use_container_width=True, on_click=load_example, args=("Excellent camera", "The camera is excellent, the battery lasts all day, and the phone is fast and easy to use."))
    st.button("Use neutral example", use_container_width=True, on_click=load_example, args=("Average phone", "The phone works as expected, but the battery life is average and the camera is acceptable."))
    st.button("Use negative example", use_container_width=True, on_click=load_example, args=("Poor battery", "The phone overheats, freezes constantly, and its battery drains within a few hours."))
    st.divider()
    st.markdown("### This session")
    st.metric("Reviews analyzed", len(st.session_state.history))
    st.caption("History is stored only while this app session stays open.")
    st.divider()
    st.markdown("### Input guidance")
    st.caption("Describe the battery, camera, speed, display, price, or build quality. Detailed reviews improve reliability.")

st.markdown('<div class="app-header"><div class="logo"><span class="logo-mark">S</span>Sentiment Studio</div><div class="status">Model online</div></div>', unsafe_allow_html=True)

tab_dashboard, tab_analysis, tab_history, tab_model = st.tabs(
    ["Dashboard", "Analyze review", "Session history", "Model information"]
)

with tab_dashboard:
    st.markdown("## Dashboard overview")
    st.caption("A visual summary of the current browser session.")
    latest_result = st.session_state.last_result
    review_count = len(st.session_state.history)
    average_confidence = (
        sum(float(item["Confidence"].rstrip("%")) for item in st.session_state.history)
        / review_count
        if review_count
        else 0
    )
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Reviews analyzed", review_count)
    metric_b.metric("Latest sentiment", latest_result["sentiment"] if latest_result else "—")
    metric_c.metric("Average confidence", f"{average_confidence:.1f}%" if review_count else "—")

    if latest_result:
        chart_col, insight_col = st.columns([1.15, 1], gap="large")
        with chart_col:
            st.markdown("### Latest sentiment mix")
            st.altair_chart(probability_donut(latest_result["probabilities"]), use_container_width=True)
        with insight_col:
            dominant = latest_result["sentiment"]
            score = latest_result["confidence"] * 100
            st.markdown("### Live insight")
            st.markdown(
                f'<div class="panel"><div class="panel-title">{dominant} review detected</div>'
                f'<div class="panel-subtitle">The model assigns <strong>{score:.1f}%</strong> confidence to the leading sentiment. Open <strong>Analyze review</strong> to submit another review.</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="empty-state"><h3>Your dashboard is ready</h3><p>Analyze your first review to unlock live sentiment charts and session insights.</p></div>',
            unsafe_allow_html=True,
        )

with tab_analysis:
    st.markdown('<section class="hero"><div class="eyebrow">Review intelligence platform</div><h1>Analyze mobile reviews with confidence.</h1><p>Write a customer review, run the BERT classifier, and explore clear sentiment probabilities in a focused workspace.</p></section>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.markdown('<div class="metric-card"><div class="metric-number">3</div><div class="metric-caption">Sentiment classes</div></div>', unsafe_allow_html=True)
    b.markdown('<div class="metric-card"><div class="metric-number">BERT</div><div class="metric-caption">Transformer classifier</div></div>', unsafe_allow_html=True)
    c.markdown(f'<div class="metric-card"><div class="metric-number">{len(st.session_state.history)}</div><div class="metric-caption">Session analyses</div></div>', unsafe_allow_html=True)
    d.markdown('<div class="metric-card"><div class="metric-number">Live</div><div class="metric-caption">Inference status</div></div>', unsafe_allow_html=True)

    st.markdown("### Review input")
    input_col, help_col = st.columns([1.65, 1], gap="large")
    with input_col:
        st.markdown('<div class="panel"><div class="panel-title">Create an analysis</div><div class="panel-subtitle">Add a full sentence or detailed review for the best model result.</div>', unsafe_allow_html=True)
        with st.form("review_form", clear_on_submit=False):
            review_title = st.text_input("Review title", placeholder="Example: Poor battery life", key="review_title")
            review_text = st.text_area("Review details", height=190, placeholder="Example: The battery drains quickly and the phone becomes hot while charging...", key="review_text")
            submitted = st.form_submit_button("Analyze review", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with help_col:
        st.markdown('<div class="panel"><div class="panel-title">How it works</div><div class="panel-subtitle">A simple three-step workflow</div><div class="workflow-step"><div class="step-number">1</div><div><div class="step-name">Write the review</div><div class="step-copy">Provide clear detail about the customer experience.</div></div></div><div class="workflow-step"><div class="step-number">2</div><div><div class="step-name">Run analysis</div><div class="step-copy">The BERT model evaluates context and wording.</div></div></div><div class="workflow-step"><div class="step-number">3</div><div><div class="step-name">Read the result</div><div class="step-copy">Compare confidence across all sentiment classes.</div></div></div></div>', unsafe_allow_html=True)

    if submitted:
        full_review = " ".join(part.strip() for part in [review_title, review_text] if part.strip())
        if not review_text.strip():
            st.warning("Enter a review before starting the analysis.")
        elif len(full_review.split()) < 5:
            st.warning("Please add more detail. Reviews with fewer than five words are often ambiguous.")
        else:
            try:
                with st.spinner("Loading the model and analyzing the review..."):
                    sentiment, confidence, probabilities = predict_sentiment_details(full_review)
                save_analysis(full_review, sentiment, confidence, probabilities)
            except (MemoryError, RuntimeError, OSError) as error:
                st.error(
                    "The sentiment model could not start. Close unused Python or "
                    "Streamlit processes, then restart the app and try again."
                )
                st.caption(f"Technical detail: {type(error).__name__}: {error}")

    result = st.session_state.last_result
    st.markdown("### Latest analysis")
    if result is None:
        st.markdown('<div class="empty-state"><h3>No review analyzed yet</h3><p>Complete the review form above to generate an interactive sentiment report.</p></div>', unsafe_allow_html=True)
    else:
        sentiment = result["sentiment"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]
        style, result_copy = {
            "Positive": ("positive", "The review contains predominantly favorable language."),
            "Neutral": ("neutral", "The review contains balanced or mixed language."),
            "Negative": ("negative", "The review contains predominantly unfavorable language."),
        }[sentiment]
        result_col, breakdown_col = st.columns([1, 1.15], gap="large")
        with result_col:
            st.markdown(f'<div class="result {style}"><div class="result-tag">Predicted sentiment</div><div class="result-value">{sentiment}</div><div class="result-copy">{result_copy}</div><div class="confidence">Confidence: {confidence * 100:.1f}%</div></div>', unsafe_allow_html=True)
            if confidence < 0.70:
                st.info("This prediction has low confidence. Add more specific product details for a clearer result.")
        with breakdown_col:
            st.markdown('<div class="panel"><div class="panel-title">Probability breakdown</div><div class="panel-subtitle">Confidence across all possible sentiment labels</div>', unsafe_allow_html=True)
            st.altair_chart(probability_donut(probabilities), use_container_width=True)
            bar_classes = {"Negative": "bar-negative", "Neutral": "bar-neutral", "Positive": "bar-positive"}
            for label in ["Negative", "Neutral", "Positive"]:
                score = probabilities[label] * 100
                st.markdown(f'<div class="probability-row"><div class="probability-head"><span>{label}</span><strong>{score:.1f}%</strong></div><div class="probability-track"><div class="probability-fill {bar_classes[label]}" style="width:{score:.2f}%"></div></div></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Class confidence")
        negative, neutral, positive = st.columns(3)
        negative.metric("Negative", f"{probabilities['Negative'] * 100:.1f}%")
        neutral.metric("Neutral", f"{probabilities['Neutral'] * 100:.1f}%")
        positive.metric("Positive", f"{probabilities['Positive'] * 100:.1f}%")
        with st.expander("View submitted review and chart"):
            chart = pd.DataFrame({"Probability": [probabilities["Negative"] * 100, probabilities["Neutral"] * 100, probabilities["Positive"] * 100]}, index=["Negative", "Neutral", "Positive"])
            st.bar_chart(chart, color="#6686ff", horizontal=True)
            st.caption("Review submitted to the model")
            st.write(result["review"])

with tab_history:
    st.markdown("## Session history")
    st.caption("The ten latest predictions from the active browser session.")
    if st.session_state.history:
        history_chart_col, history_table_col = st.columns([1, 1.35], gap="large")
        with history_chart_col:
            st.markdown("### Sentiment distribution")
            st.altair_chart(session_sentiment_chart(st.session_state.history), use_container_width=True)
        with history_table_col:
            st.markdown("### Recent analyses")
            history_frame = pd.DataFrame(st.session_state.history)
            st.dataframe(history_frame, use_container_width=True, hide_index=True)
        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()
    else:
        st.info("No session history yet. Analyze a review in the Analysis workspace.")

with tab_model:
    st.markdown("## Model information")
    st.markdown('<div class="panel"><div class="panel-title">BERT sentiment classifier</div><div class="panel-subtitle">The project model classifies a mobile-phone review into Negative, Neutral, or Positive sentiment.</div></div>', unsafe_allow_html=True)
    model_a, model_b, model_c = st.columns(3)
    model_a.metric("Classes", "3")
    model_b.metric("Maximum input", "128 tokens")
    model_c.metric("Output", "Probability distribution")
    st.markdown("### Tips for reliable results")
    st.markdown("- Use a complete review rather than one word.\n- Include product details and your reason for liking or disliking it.\n- Treat low-confidence results as uncertain rather than final.")
