import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import streamlit as st

from src.predict import predict_sentiment


st.set_page_config(
    page_title="Mobile Review Sentiment Analysis",
    page_icon="📱",
    layout="centered"
)


st.title("📱 Mobile Review Sentiment Analysis")

st.write(
    "Enter a mobile phone review and the trained "
    "BERT model will automatically predict its sentiment."
)


review_title = st.text_input(
    "Review Title"
)


review_text = st.text_area(
    "Review Text",
    height=150,
    placeholder="Example: The phone camera is excellent..."
)


if st.button("Analyze Review"):

    if review_text.strip() == "":
        
        st.warning(
            "Please enter a review."
        )

    else:

        full_review = (
            review_title
            + " "
            + review_text
        )

        sentiment, confidence = predict_sentiment(
            full_review
        )

        st.subheader("Prediction")

        if sentiment == "Positive":

            st.success(
                f"😊 Positive"
            )

        elif sentiment == "Neutral":

            st.info(
                f"😐 Neutral"
            )

        else:

            st.error(
                f"😞 Negative"
            )

        st.write(
            f"Confidence: {confidence * 100:.2f}%"
        )

        st.subheader("Review")

        st.write(full_review)