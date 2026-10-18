import sys
from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_preprocessing import clean_text

MODEL_PATH = ROOT / "models" / "traditional_ml" / "best_tfidf_model.joblib"
if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "models" / "logistic_regression.pkl"

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬"
)

st.title("AI-Powered Sentiment Analysis")
st.write("Enter a mobile-product review to predict its sentiment.")

if not MODEL_PATH.exists():
    st.warning("Run Notebook 06 to train and save the model.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

review = st.text_area(
    "Enter your review",
    placeholder="The phone has a great camera and battery life."
)

if st.button("Predict Sentiment"):

    if not review.strip():
        st.error("Please enter a review.")

    else:
        cleaned_review = clean_text(review)
        prediction = model.predict([cleaned_review])[0]

        st.success(f"Predicted sentiment: {prediction}")