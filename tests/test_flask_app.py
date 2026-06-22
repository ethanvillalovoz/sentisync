import unittest

from flask_app import app as app_module


class DummyMatrix:
    def __init__(self, row_count):
        self.row_count = row_count

    def toarray(self):
        return [[0.0] for _ in range(self.row_count)]


class DummyVectorizer:
    def transform(self, comments):
        return DummyMatrix(len(comments))


class DummyModel:
    def predict(self, rows):
        return [1 for _ in rows]


class FlaskAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_predict_requires_comments(self):
        response = self.client.post("/predict", json={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_predict_uses_configured_model_and_vectorizer(self):
        original_model = app_module.model
        original_vectorizer = app_module.vectorizer
        app_module.model = DummyModel()
        app_module.vectorizer = DummyVectorizer()

        try:
            response = self.client.post(
                "/predict",
                json={"comments": ["This is not bad", "Really useful walkthrough"]},
            )
        finally:
            app_module.model = original_model
            app_module.vectorizer = original_vectorizer

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]["sentiment"], 1)

    def test_generate_chart_requires_sentiment_counts(self):
        response = self.client.post("/generate_chart", json={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_preprocess_comment_preserves_negation_terms(self):
        processed = app_module.preprocess_comment("This is NOT bad!!!")

        self.assertIn("not", processed)


if __name__ == "__main__":
    unittest.main()
