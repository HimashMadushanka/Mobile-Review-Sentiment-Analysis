import html
import re
import unicodedata
from typing import Optional

import pandas as pd


VALID_SENTIMENTS = {"positive", "neutral", "negative"}
MISSING_TEXT_VALUES = {"", "nan", "none", "null", "n/a"}


def sanitize_text(text: Optional[str]) -> str:
    """Base text sanitization shared across all pipelines.

    - Handles None / NaN values safely.
    - Normalizes Unicode characters (NFKC) so accents and special spaces are standardized.
    - Normalizes typographic quotes (“”, ‘’) and dashes (—, –) to standard ASCII.
    - Unescapes HTML entities (e.g. &amp;, &quot;, &#39;).
    - Strips HTML tags (<br>, <p>, etc.).
    - Strips web URLs and email addresses.
    - Strips social mentions (@user) and unwraps hashtags (#phone -> phone).
    """
    if text is None or pd.isna(text):
        return ""

    text = str(text)

    # 1. Unicode normalization (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # 2. Normalize smart/curly quotes and dashes
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("—", " - ")
        .replace("–", " - ")
    )

    # 3. Decode HTML entities
    text = html.unescape(text)

    # 4. Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 5. Strip URLs and email addresses
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " ", text)

    # 6. Handle @mentions and #hashtags
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)

    return text


def preprocess_for_bert(text: Optional[str]) -> str:
    """Preprocess review text for BERT / Transformer inference & training.

    Ensures 100% parity with fine-tuned bert-base-uncased tokenizer rules:
    - Applies base sanitization (HTML, URLs, mentions, Unicode normalization).
    - Lowercases text to match bert-base-uncased vocabulary.
    - Preserves sentiment-critical punctuation: ! ? . , ' " - :
      (vital for attention mechanisms and emotional tone like excitement or complaints).
    - Strips residual unsupported symbols/artifacts.
    - Normalizes spaces before punctuation and collapses whitespace.
    """
    text = sanitize_text(text)
    if not text:
        return ""

    text = text.lower()
    # Retain word chars, spaces, and expressive sentiment punctuation
    text = re.sub(r"[^\w\s.,!?'\":-]", " ", text)
    # Remove detached period/punctuation artifacts resulting from stripped URLs/emails
    text = re.sub(r"\s+([.,!?:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_for_tfidf(text: Optional[str]) -> str:
    """Preprocess review text for traditional ML (TF-IDF vectorizer)."""
    return preprocess_for_bert(text)


def clean_text(text: Optional[str]) -> str:
    """Legacy alias matching notebook 02 and backward compatibility.

    Delegates to preprocess_for_bert to guarantee 100% parity across
    notebook training, evaluation, Streamlit, and FastAPI inference.
    """
    return preprocess_for_bert(text)


def prepare_dataset(df):
    required_columns = ["review_text", "sentiment"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    df = df.copy()

    # Normalize text and labels before filtering missing values. This catches
    # actual nulls, whitespace-only fields, and common string placeholders.
    df["review_text"] = df["review_text"].map(
        lambda value: unicodedata.normalize("NFKC", str(value)).strip()
        if not pd.isna(value) else pd.NA
    )
    df["sentiment"] = df["sentiment"].map(
        lambda value: str(value).strip().casefold()
        if not pd.isna(value) else pd.NA
    )

    review_is_missing = df["review_text"].isna() | df["review_text"].str.casefold().isin(MISSING_TEXT_VALUES)
    sentiment_is_valid = df["sentiment"].isin(VALID_SENTIMENTS)
    df = df.loc[~review_is_missing & sentiment_is_valid].copy()

    df["clean_text"] = df["review_text"].apply(clean_text)

    df = df[df["clean_text"] != ""]

    # Ratings should be usable numeric values on the usual 1-5 scale.
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df.loc[~df["rating"].between(1, 5), "rating"] = pd.NA

    # Keep repeated review wording: separate customers can write the same
    # short review, so text-only deduplication can discard valid examples.

    return df.reset_index(drop=True)
