
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preprocessing import preprocess_for_bert

MODEL_DIR = ROOT / "models" / "bert" / "fine-tuned"

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"BERT model folder not found: {MODEL_DIR}"
    )

# Load model and tokenizer
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR)
)
model.to(device)
model.eval()

# Load labels in their original training order
LABELS_FILE = MODEL_DIR / "label_encoder_classes.txt"

if not LABELS_FILE.exists():
    raise FileNotFoundError(
        "Create label_encoder_classes.txt in the model folder. "
        "It must contain one class label per line, in training order."
    )

LABELS = [
    line.strip()
    for line in LABELS_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

if len(LABELS) != model.config.num_labels:
    raise ValueError(
        "The number of labels does not match the model's output classes."
    )

app = FastAPI(title="Review Sentiment API")

# Enable CORS for browser extensions and frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


@app.get("/")
def home():
    return {"message": "Review Sentiment API is running"}


@app.post("/predict")
def predict(request: ReviewRequest):
    raw_text = request.text.strip()

    if not raw_text:
        raise HTTPException(
            status_code=400,
            detail="Please provide review text."
        )

    # Apply exact same preprocessing used during model fine-tuning
    cleaned_text = preprocess_for_bert(raw_text)

    if not cleaned_text:
        raise HTTPException(
            status_code=400,
            detail="Review text contains no valid words or sentiment information after cleaning (e.g. only URLs, HTML tags, or symbols)."
        )

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        class_id = int(torch.argmax(probabilities).item())

    return {
        "sentiment": LABELS[class_id],
        "confidence": round(
            float(probabilities[class_id].item()), 4
        ),
        "cleaned_text": cleaned_text,
    }
