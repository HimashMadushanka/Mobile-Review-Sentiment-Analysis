import io
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus

import joblib
import pandas as pd
import requests
import streamlit as st

# Setup Root Directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_preprocessing import clean_text

# Paths & Endpoints
MODEL_PATH = ROOT / "models" / "traditional_ml" / "best_tfidf_model.joblib"
VECTORIZER_PATH = ROOT / "models" / "traditional_ml" / "tfidf_vectorizer.joblib"
FIGURES_DIR = ROOT / "outputs" / "figures"
API_URL = "http://127.0.0.1:8000/predict"

# Page Configuration
st.set_page_config(
    page_title="Mobile Review Sentiment Analysis",
    page_icon="📱",
    layout="wide",
)

# Custom Styling for Clear Structured Layout
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Professional Headings & Labels */
    h1, h2, h3, h4 {
        color: #0f172a;
        font-weight: 700;
        letter-spacing: -0.3px;
    }

    .title-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 22px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
    }
    .title-banner-text {
        color: #ffffff !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px !important;
        margin: 0 !important;
        display: block !important;
    }

    /* Structured Clean Container Cards */
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }

    /* Textarea Clean Styling */
    .stTextArea textarea {
        border-radius: 8px !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
        background-color: #ffffff !important;
    }
    .stTextArea textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.12) !important;
    }

    /* Refined Corporate Sentiment Boxes */
    .sentiment-box {
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .sentiment-positive {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        color: #166534;
    }
    .sentiment-negative {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 5px solid #dc2626;
        color: #991b1b;
    }
    .sentiment-neutral {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 5px solid #d97706;
        color: #92400e;
    }

    /* Subtle Slate Aspect Badges */
    .aspect-badge {
        display: inline-block;
        background: #f8fafc;
        color: #475569;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 4px 4px 0 0;
        border: 1px solid #e2e8f0;
    }

    /* Button Styling */
    .stButton button {
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    return None, None


tfidf_model, tfidf_vectorizer = load_artifacts()


def extract_aspects(text: str):
    lower = text.lower()
    aspects = []
    if re.search(r"\b(battery|charge|charging|drain|mah)\b", lower):
        aspects.append("Battery")
    if re.search(r"\b(camera|photo|lens|zoom|video)\b", lower):
        aspects.append("Camera")
    if re.search(r"\b(screen|display|amoled|oled|resolution)\b", lower):
        aspects.append("Display")
    if re.search(r"\b(speed|lag|fast|processor|ram|performance)\b", lower):
        aspects.append("Performance")
    if re.search(r"\b(price|cost|money|value|worth)\b", lower):
        aspects.append("Price / Value")
    if re.search(r"\b(sound|speaker|audio|mic|volume)\b", lower):
        aspects.append("Audio")
    return aspects


# ==============================================================================
# SIDEBAR: CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.subheader("Model Configuration")

    model_option = st.radio(
        "Select Model Architecture:",
        options=["Fine-Tuned BERT", "TF-IDF (Baseline)"],
        index=0,
        help="BERT runs via FastAPI (82.6% Acc). TF-IDF runs locally on CPU (79.5% Acc).",
    )

    st.markdown("---")

    # API Health Check
    try:
        res = requests.get("http://127.0.0.1:8000/", timeout=1.0)
        api_status = res.status_code == 200
    except Exception:
        api_status = False

    st.markdown(
        f"**Backend Status:** {'🟢 Online' if api_status else '🔴 Offline'}"
    )
    if not api_status and "BERT" in model_option:
        st.caption("Start the backend with `python run.py`")


# ==============================================================================
# MAIN PAGE: HEADER & TABS
# ==============================================================================
st.markdown(
    """
    <div class="title-banner">
        <div class="title-banner-text">
            Mobile Review Sentiment Analysis
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(
    ["Analysis Studio", "Batch Processing", "Live Web Search"]
)

# ------------------------------------------------------------------------------
# TAB 1: ANALYSIS STUDIO
# ------------------------------------------------------------------------------
with tab1:
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("#### Input Review")

        # Quick preset selection
        st.caption("Select a sample to auto-fill:")
        p1, p2, p3 = st.columns(3)
        if p1.button("Positive Review", use_container_width=True):
            st.session_state["sample_text"] = "Great camera clarity and phenomenal 2-day battery life. The OLED screen is stunning."
        if p2.button("Neutral Review", use_container_width=True):
            st.session_state["sample_text"] = "Decent budget device. The processor is fast enough, but charging speed is quite slow."
        if p3.button("Negative Review", use_container_width=True):
            st.session_state["sample_text"] = "Terrible experience. The phone overheats constantly and the screen froze on day two."

        input_text = st.text_area(
            "Review text input:",
            value=st.session_state.get("sample_text", ""),
            placeholder="Type or paste a customer review...",
            height=130,
            label_visibility="collapsed",
        )

        submit_btn = st.button("Analyze Review", type="primary", use_container_width=True)

    with col_right:
        st.markdown("#### Analysis Results")

        if submit_btn:
            if not input_text.strip():
                st.warning("Please provide a review to analyze.")
            else:
                start_clock = time.perf_counter()
                predicted_label = None
                confidence_score = 0.0
                error_detail = None

                if "BERT" in model_option:
                    try:
                        api_res = requests.post(API_URL, json={"text": input_text}, timeout=10)
                        if api_res.status_code == 200:
                            payload = api_res.json()
                            predicted_label = str(payload["sentiment"]).capitalize()
                            confidence_score = float(payload["confidence"])
                        else:
                            error_detail = api_res.json().get("detail", "Prediction error.")
                    except Exception:
                        error_detail = "FastAPI backend is offline. Run `python run.py`."
                else:
                    if tfidf_model and tfidf_vectorizer:
                        clean_input = clean_text(input_text)
                        if not clean_input:
                            error_detail = "No valid text after preprocessing."
                        else:
                            feats = tfidf_vectorizer.transform([clean_input])
                            predicted_label = str(tfidf_model.predict(feats)[0]).capitalize()
                            if hasattr(tfidf_model, "predict_proba"):
                                confidence_score = float(max(tfidf_model.predict_proba(feats)[0]))
                    else:
                        error_detail = "TF-IDF model artifacts missing."

                latency_ms = (time.perf_counter() - start_clock) * 1000

                if error_detail:
                    st.error(error_detail)
                elif predicted_label:
                    theme = (
                        "sentiment-positive" if predicted_label == "Positive"
                        else "sentiment-negative" if predicted_label == "Negative"
                        else "sentiment-neutral"
                    )

                    # Result Card
                    st.markdown(
                        f"""
                        <div class="sentiment-box {theme}">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div style="font-size:22px; font-weight:800;">{predicted_label}</div>
                                <div style="font-size:16px; font-weight:700;">{confidence_score * 100:.1f}% Confidence</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Metrics Summary
                    m1, m2 = st.columns(2)
                    m1.metric("Active Model", "BERT" if "BERT" in model_option else "TF-IDF")
                    m2.metric("Inference Latency", f"{latency_ms:.1f} ms")

                    # Aspects
                    found_aspects = extract_aspects(input_text)
                    if found_aspects:
                        st.markdown("**Detected Topics:**")
                        badges = "".join([f'<span class="aspect-badge">{a}</span>' for a in found_aspects])
                        st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)
        else:
            st.info("Enter a review on the left and click **Analyze Review** to view output metrics.")


# ------------------------------------------------------------------------------
# TAB 2: BATCH PROCESSING
# ------------------------------------------------------------------------------
with tab2:
    st.markdown("#### Upload Dataset for Bulk Prediction")
    batch_file = st.file_uploader("Upload CSV file:", type=["csv"])

    if batch_file:
        try:
            df_batch = pd.read_csv(batch_file)
            st.write(f"Total Rows: **{len(df_batch):,}**")

            col_choice = st.selectbox("Select review text column:", options=df_batch.columns.tolist())

            if st.button("Execute Batch Prediction", type="primary"):
                if not (tfidf_model and tfidf_vectorizer):
                    st.error("TF-IDF model is required for fast batch processing.")
                else:
                    with st.spinner("Processing reviews..."):
                        cleaned_series = df_batch[col_choice].fillna("").astype(str).apply(clean_text)
                        transformed = tfidf_vectorizer.transform(cleaned_series)
                        df_batch["Predicted_Sentiment"] = [p.capitalize() for p in tfidf_model.predict(transformed)]

                        if hasattr(tfidf_model, "predict_proba"):
                            probs = tfidf_model.predict_proba(transformed)
                            df_batch["Confidence"] = [round(float(max(p)), 4) for p in probs]

                    st.success("Batch processing completed.")

                    # Summary Counts
                    counts = df_batch["Predicted_Sentiment"].value_counts()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Positive", counts.get("Positive", 0))
                    c2.metric("Neutral", counts.get("Neutral", 0))
                    c3.metric("Negative", counts.get("Negative", 0))

                    st.dataframe(df_batch[[col_choice, "Predicted_Sentiment", "Confidence"]].head(20), use_container_width=True)

                    # Export
                    out_buffer = io.StringIO()
                    df_batch.to_csv(out_buffer, index=False)
                    st.download_button(
                        "Download Analyzed CSV",
                        data=out_buffer.getvalue(),
                        file_name="sentiment_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as ex:
            st.error(f"Error reading CSV: {ex}")


# ------------------------------------------------------------------------------
# TAB 3: WEB SEARCH
# ------------------------------------------------------------------------------
with tab3:
    st.markdown("#### Test Real-World Reviews via Web Search")
    st.caption("Search for live mobile reviews to test with the Chrome Browser Extension.")

    with st.form("search_form"):
        search_query = st.text_input("Enter product name:", placeholder="e.g. Google Pixel 8 Pro customer reviews")
        search_submitted = st.form_submit_button("Search on Google")

    if search_submitted and search_query.strip():
        search_url = f"https://www.google.com/search?q={quote_plus(search_query.strip())}"
        st.link_button(f"Open Google Search for '{search_query.strip()}'", search_url)
