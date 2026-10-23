import sys
from pathlib import Path
from urllib.parse import quote_plus

import joblib
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_preprocessing import clean_text

MODEL_PATH = ROOT / "models" / "traditional_ml" / "best_tfidf_model.joblib"
VECTORIZER_PATH = ROOT / "models" / "traditional_ml" / "tfidf_vectorizer.joblib"
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Mobile Review Sentiment Analyzer", page_icon="📱", layout="centered")

st.title("📱 AI-Powered Review Sentiment Analyzer")
st.caption("Analyze mobile product reviews with Deep Learning (BERT) or Classical ML (TF-IDF).")


@st.cache_resource
def load_traditional_artifacts():
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    return None, None


tfidf_model, tfidf_vectorizer = load_traditional_artifacts()

sentiment_tab, search_tab = st.tabs(["💬 Sentiment Analyzer", "🔍 Google Search"])

with sentiment_tab:
    st.subheader("Analyze Review Sentiment")

    # Model Selection Switcher
    model_choice = st.radio(
        "Select Model Architecture:",
        options=[
            "BERT (Deep Learning / Transformer - High Accuracy)",
            "TF-IDF + Classical ML (Ultra Fast Baseline)",
        ],
        index=0,
        help="BERT connects to the FastAPI backend microservice. TF-IDF runs locally.",
    )

    review = st.text_area(
        "Enter customer review text:",
        placeholder="e.g., The battery life is phenomenal and the display is crystal clear, but the camera struggles in low light.",
        height=120,
    )

    if st.button("🚀 Predict Sentiment", type="primary"):
        if not review.strip():
            st.error("Please enter a review first.")
        else:
            if "BERT" in model_choice:
                # Call FastAPI backend
                try:
                    with st.spinner("Analyzing with fine-tuned BERT..."):
                        response = requests.post(
                            API_URL,
                            json={"text": review},
                            timeout=10,
                        )

                    if response.status_code == 200:
                        data = response.json()
                        sentiment = str(data["sentiment"]).capitalize()
                        confidence = float(data["confidence"])

                        # Visual result cards
                        col1, col2 = st.columns(2)
                        with col1:
                            if sentiment.lower() == "positive":
                                st.success(f"### Sentiment: {sentiment} 😊")
                            elif sentiment.lower() == "negative":
                                st.error(f"### Sentiment: {sentiment} 😞")
                            else:
                                st.info(f"### Sentiment: {sentiment} 😐")

                        with col2:
                            st.metric("Model Confidence", f"{confidence * 100:.2f}%")

                        st.progress(confidence)

                        with st.expander("ℹ️ Preprocessed Text (BERT Tokenizer input)"):
                            st.code(data.get("cleaned_text", ""))

                    else:
                        detail = response.json().get("detail", response.text)
                        st.error(f"API Error ({response.status_code}): {detail}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "⚠️ Unable to reach the FastAPI backend at `http://127.0.0.1:8000`. "
                        "Make sure your API is running (e.g. `python run.py`), or switch to the TF-IDF model."
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

            else:
                # Run Local TF-IDF model
                if tfidf_model is None or tfidf_vectorizer is None:
                    st.warning("TF-IDF model files not found in models/traditional_ml/. Please run notebooks 05 and 06.")
                else:
                    cleaned_review = clean_text(review)
                    if not cleaned_review:
                        st.warning("Review contains no valid words after cleaning.")
                    else:
                        features = tfidf_vectorizer.transform([cleaned_review])
                        prediction = str(tfidf_model.predict(features)[0]).capitalize()

                        confidence_text = None
                        confidence_val = None
                        if hasattr(tfidf_model, "predict_proba"):
                            probs = tfidf_model.predict_proba(features)[0]
                            max_prob = max(probs)
                            confidence_val = float(max_prob)
                            confidence_text = f"{confidence_val * 100:.2f}%"

                        col1, col2 = st.columns(2)
                        with col1:
                            if prediction.lower() == "positive":
                                st.success(f"### Sentiment: {prediction} 😊")
                            elif prediction.lower() == "negative":
                                st.error(f"### Sentiment: {prediction} 😞")
                            else:
                                st.info(f"### Sentiment: {prediction} 😐")

                        with col2:
                            if confidence_text:
                                st.metric("Model Confidence", confidence_text)

                        if confidence_val:
                            st.progress(confidence_val)

                        with st.expander("ℹ️ Preprocessed Text (TF-IDF input)"):
                            st.code(cleaned_review)

with search_tab:
    st.subheader("Live Product Review Search")
    st.write("Search Google for live mobile product reviews and test them with your models or browser extension.")

    with st.form("google_search_form"):
        search_query = st.text_input(
            "Search Google",
            placeholder="For example: iPhone 15 Pro customer reviews",
        )
        search_submitted = st.form_submit_button("Search Google")

    if search_submitted:
        if search_query.strip():
            st.session_state["google_search_query"] = search_query.strip()
        else:
            st.session_state.pop("google_search_query", None)
            st.warning("Enter a search term first.")

    saved_query = st.session_state.get("google_search_query")
    if saved_query:
        search_url = f"https://www.google.com/search?q={quote_plus(saved_query)}"
        st.link_button("🌐 Open Google Results in New Tab", search_url)
        st.caption(
            "Tip: Select any review text on the webpage to inspect sentiment instantly with your Chrome extension."
        )
