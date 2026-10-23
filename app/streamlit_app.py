import sys
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_preprocessing import clean_text

MODEL_PATH = ROOT / "models" / "traditional_ml" / "best_tfidf_model.joblib"
VECTORIZER_PATH = ROOT / "models" / "traditional_ml" / "tfidf_vectorizer.joblib"

st.set_page_config(page_title="Sentiment Analyzer", page_icon="\U0001F4F1")
st.title("AI-Powered Sentiment Analysis")
st.write("Enter a mobile product review to predict its sentiment.")

if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    st.warning("Run notebooks 05 and 06 to create the model and TF-IDF vectorizer.")
    st.stop()


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


model, vectorizer = load_artifacts()
review = st.text_area(
    "Enter your review",
    placeholder="The phone has a great camera and battery life.",
)

if st.button("Predict sentiment"):
    if not review.strip():
        st.error("Please enter a review.")
    else:
        features = vectorizer.transform([clean_text(review)])
        prediction = model.predict(features)[0]
        st.success(f"Predicted sentiment: {prediction}")
