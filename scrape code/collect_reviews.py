
import os
import gzip
import json
import requests
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================

URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/"
    "categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz"
)

# Save relative to the project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

COMPRESSED_FILE = os.path.join(
    DATA_DIR,
    "reviews_Cell_Phones_and_Accessories_5.json.gz"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "amazon_mobile_reviews.csv"
)

TARGET_ROWS = 20000

# ==========================================
# 2. DOWNLOAD DATASET
# ==========================================

print("Downloading Amazon mobile reviews...")

with requests.get(URL, stream=True, timeout=120) as response:
    response.raise_for_status()

    with open(COMPRESSED_FILE, "wb") as file:
        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):
            if chunk:
                file.write(chunk)

print("Download completed!")
print(
    "Compressed size:",
    round(os.path.getsize(COMPRESSED_FILE) / (1024 * 1024), 2),
    "MB"
)

# ==========================================
# 3. READ UP TO 20,000 UNIQUE REVIEWS
# ==========================================

reviews = []
seen_texts = set()

print("Reading reviews...")

with gzip.open(
    COMPRESSED_FILE,
    "rt",
    encoding="utf-8"
) as file:

    for line in file:
        if len(reviews) >= TARGET_ROWS:
            break

        try:
            review = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = review.get("reviewText")
        rating = review.get("overall")

        if text is None or rating is None:
            continue

        text = str(text).strip()

        if not text or text in seen_texts:
            continue

        try:
            rating = float(rating)
        except (ValueError, TypeError):
            continue

        if not 1 <= rating <= 5:
            continue

        seen_texts.add(text)

        reviews.append({
            "review_id": str(len(reviews) + 1),
            "product_name": review.get("asin", ""),
            "review_text": text,
            "rating": rating
        })

if not reviews:
    raise ValueError("No valid reviews were collected.")

# ==========================================
# 4. CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(reviews)

# ==========================================
# 5. CREATE SENTIMENT LABELS
# ==========================================

def get_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    return "Positive"

df["sentiment"] = df["rating"].apply(get_sentiment)

# ==========================================
# 6. KEEP FIVE REQUIRED COLUMNS
# ==========================================

df = df[
    [
        "review_id",
        "product_name",
        "review_text",
        "rating",
        "sentiment"
    ]
]

# ==========================================
# 7. SAVE CSV
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n===================================")
print("DATA COLLECTION COMPLETED")
print("===================================")

print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("Columns:", df.columns.tolist())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nFirst five reviews:")
print(df.head())

print("\nSaved to:")
print(OUTPUT_FILE)
