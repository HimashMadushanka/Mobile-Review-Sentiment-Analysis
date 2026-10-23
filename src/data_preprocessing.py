import re
import unicodedata

import pandas as pd


VALID_SENTIMENTS = {"positive", "neutral", "negative"}
MISSING_TEXT_VALUES = {"", "nan", "none", "null", "n/a"}


def clean_text(text):
    """Apply the same cleanup used by notebook 02 before model training."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " ", text)
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
