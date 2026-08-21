import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


MODEL_PATH = "models/bert_sentiment"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


def predict_sentiment(text):

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

    labels = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    sentiment = labels[prediction]

    confidence = probabilities[0][prediction].item()

    return sentiment, confidence