import unittest
from fastapi.testclient import TestClient
from app.api import app


class TestSentimentAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
        self.assertEqual(response.json()["message"], "Review Sentiment API is running")

    def test_predict_positive_review(self):
        response = self.client.post(
            "/predict",
            json={"text": "Outstanding camera quality and superb battery life! Highly recommend."}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("sentiment", data)
        self.assertIn("confidence", data)
        self.assertIn("cleaned_text", data)
        self.assertEqual(data["sentiment"], "positive")
        self.assertGreater(data["confidence"], 0.7)

    def test_predict_negative_review(self):
        response = self.client.post(
            "/predict",
            json={"text": "Horrible phone. The screen broke immediately and the battery drains in one hour."}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["sentiment"], "negative")
        self.assertGreater(data["confidence"], 0.7)

    def test_predict_empty_text(self):
        response = self.client.post(
            "/predict",
            json={"text": "   "}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())

    def test_predict_noise_only_text(self):
        response = self.client.post(
            "/predict",
            json={"text": "https://example.com @user <br/>"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())


if __name__ == "__main__":
    unittest.main()
