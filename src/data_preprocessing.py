import html
import re
import unicodedata

import pandas as pd


VALID_SENTIMENTS = {"Positive", "Neutral", "Negative"}
MISSING_TEXT_VALUES = {"", "nan", "none", "null", "n/a"}


CONTRACTIONS = {
    r"can\'t": "cannot",
    r"won\'t": "will not",
    r"n\'t": " not",
    r"\'re": " are",
    r"\'s": " is",
    r"\'d": " would",
    r"\'ll": " will",
    r"\'t": " not",
    r"\'ve": " have",
    r"\'m": " am",
}


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # 1. Decode HTML entities first so characters like &nbsp; become real spaces
    text = html.unescape(text)

    # 2. Normalize Unicode (NFKC) and lowercase
    text = unicodedata.normalize("NFKC", text).casefold()

    # 3. Remove HTML markup, control characters, URLs, and handles
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f\u200b\ufeff\xa0]", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)

    # 4. Expand contractions to preserve sentiment negations (e.g., don't -> do not)
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)

    # 5. Compress elongated characters (e.g., 'sooooo' -> 'soo')
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 6. Pad sentiment-critical punctuation (! and ?) so they are captured as tokens
    text = re.sub(r"([!?]+)", r" \1 ", text)

    # 7. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


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
        lambda value: str(value).strip().casefold().title()
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
