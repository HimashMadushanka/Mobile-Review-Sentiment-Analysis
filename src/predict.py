from functools import lru_cache
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "bert_sentiment"


LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the model only when the first prediction is requested."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


def predict_sentiment_details(text):
    """Return the predicted sentiment, confidence, and all class probabilities."""

    # Delaying this import keeps the Streamlit interface available if PyTorch
    # needs time or resources to initialize.
    import torch

    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

    sentiment = LABELS[prediction]
    confidence = probabilities[0][prediction].item()
    class_probabilities = {
        LABELS[index]: probability
        for index, probability in enumerate(probabilities[0].tolist())
    }

    return sentiment, confidence, class_probabilities


def predict_sentiment(text):
    """Return only sentiment and confidence for backwards compatibility."""
    sentiment, confidence, _ = predict_sentiment_details(text)

    return sentiment, confidence
