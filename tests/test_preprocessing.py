import unittest
from src.data_preprocessing import (
    sanitize_text,
    preprocess_for_bert,
    preprocess_for_tfidf,
    clean_text,
    prepare_dataset,
)
import pandas as pd


class TestPreprocessing(unittest.TestCase):
    def test_html_entities_and_tags(self):
        dirty = "<p>The screen is &quot;great&quot; and vibrant!<br></p>"
        expected = 'the screen is "great" and vibrant!'
        self.assertEqual(preprocess_for_bert(dirty), expected)

    def test_urls_and_emails(self):
        dirty = "Check details at https://example.com/phone or email support@domain.com. Terrible battery!"
        expected = "check details at or email. terrible battery!"
        self.assertEqual(preprocess_for_bert(dirty), expected)

    def test_mentions_and_hashtags(self):
        dirty = "Shoutout to @brand for this #amazing phone!"
        expected = "shoutout to for this amazing phone!"
        self.assertEqual(preprocess_for_bert(dirty), expected)

    def test_punctuation_preservation_for_sentiment(self):
        dirty = "Don't buy this phone! It's terrible... Why does it overheat?!"
        expected = "don't buy this phone! it's terrible... why does it overheat?!"
        self.assertEqual(preprocess_for_bert(dirty), expected)

    def test_unicode_normalization(self):
        dirty = "“Amazing\u00a0camera” — best phone ever!"
        expected = '"amazing camera" - best phone ever!'
        self.assertEqual(preprocess_for_bert(dirty), expected)

    def test_clean_text_backward_compatibility(self):
        sample = "Great product! Would buy again: https://amazon.com"
        self.assertEqual(clean_text(sample), preprocess_for_bert(sample))
        self.assertEqual(preprocess_for_tfidf(sample), preprocess_for_bert(sample))

    def test_empty_and_noise_only(self):
        self.assertEqual(preprocess_for_bert(""), "")
        self.assertEqual(preprocess_for_bert(None), "")
        self.assertEqual(preprocess_for_bert("https://example.com <br/> @someone"), "")

    def test_prepare_dataset_valid(self):
        raw_df = pd.DataFrame({
            "review_text": ["Super fast phone with good camera!", "Terrible display quality."],
            "sentiment": ["positive", "negative"],
            "rating": [5, 1],
        })
        cleaned = prepare_dataset(raw_df)
        self.assertEqual(len(cleaned), 2)
        self.assertIn("clean_text", cleaned.columns)
        self.assertEqual(cleaned.loc[0, "sentiment"], "positive")

    def test_prepare_dataset_missing_column(self):
        raw_df = pd.DataFrame({"review_text": ["Good"]})
        with self.assertRaises(ValueError):
            prepare_dataset(raw_df)

    def test_prepare_dataset_filters_invalid_sentiment_and_empty_text(self):
        raw_df = pd.DataFrame({
            "review_text": [
                "Valid review",
                "Another valid review",
                "https://empty.com",
                None,
            ],
            "sentiment": [
                "positive",
                "invalid_sentiment",
                "positive",
                "positive",
            ],
        })
        cleaned = prepare_dataset(raw_df)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["review_text"], "Valid review")


if __name__ == "__main__":
    unittest.main()

