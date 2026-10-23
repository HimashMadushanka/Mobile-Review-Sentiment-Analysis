import sys
from pathlib import Path
from urllib.parse import quote_plus

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_preprocessing import clean_text

MODEL_PATH = ROOT / "models" / "traditional_ml" / "best_tfidf_model.joblib"
VECTORIZER_PATH = ROOT / "models" / "traditional_ml" / "tfidf_vectorizer.joblib"

st.set_page_config(page_title="Sentiment Analyzer", page_icon="\U0001F4F1")
st.title("AI-Powered Sentiment Analysis")
st.write("Analyze mobile product reviews or search the web for reviews.")

if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    st.warning("Run notebooks 05 and 06 to create the model and TF-IDF vectorizer.")
    st.stop()


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


model, vectorizer = load_artifacts()

sentiment_tab, search_tab = st.tabs(["Sentiment analyzer", "Google Search"])

with sentiment_tab:
    st.write("Enter a mobile product review to predict its sentiment.")
    review = st.text_area(
        "Enter your review",
        placeholder="The phone has a great camera and battery life.",
    )

    if st.button("Predict sentiment"):
        if not review.strip():
            st.error("Please enter a review.")
        else:
            cleaned_review = clean_text(review)
            if not cleaned_review:
                st.warning("Review contains no valid words or sentiment information after cleaning.")
            else:
                features = vectorizer.transform([cleaned_review])
                prediction = model.predict(features)[0]
                st.success(f"Predicted sentiment: {prediction}")

with search_tab:
    st.write("Search Google for product pages and customer reviews.")
    with st.form("google_search_form"):
        search_query = st.text_input(
            "Search Google",
            placeholder="For example: Samsung Galaxy S24 customer reviews",
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
        st.link_button("Open Google results", search_url)
        st.caption(
            "Google opens in a separate browser tab. Select review text on a webpage "
            "to use the browser extension's sentiment popup."
        )
