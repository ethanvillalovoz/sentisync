import unittest
from unittest.mock import patch

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
        return [1 for _ in range(rows.row_count)]


class DummyYouTubeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "items": [
                {
                    "snippet": {
                        "topLevelComment": {
                            "snippet": {
                                "textOriginal": "A useful explanation",
                                "publishedAt": "2026-06-01T10:00:00Z",
                                "authorChannelId": {"value": "author-1"},
                            }
                        }
                    }
                }
            ]
        }


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

    def test_predict_rejects_oversized_batches(self):
        response = self.client.post(
            "/predict",
            json={"comments": ["valid comment"] * (app_module.MAX_COMMENTS + 1)},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("at most", response.get_json()["error"])

    def test_predict_preserves_text_as_json_data(self):
        original_model = app_module.model
        original_vectorizer = app_module.vectorizer
        app_module.model = DummyModel()
        app_module.vectorizer = DummyVectorizer()
        dangerous_text = '<img src=x onerror="alert(1)"> useful analysis'

        try:
            response = self.client.post("/predict", json={"comments": [dangerous_text]})
        finally:
            app_module.model = original_model
            app_module.vectorizer = original_vectorizer

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()[0]["comment"], dangerous_text)

    def test_youtube_comments_validates_video_id_before_configuration(self):
        response = self.client.post(
            "/youtube/comments",
            json={"video_id": "too-short"},
        )

        self.assertEqual(response.status_code, 400)

    def test_youtube_comments_requires_server_side_key(self):
        response = self.client.post(
            "/youtube/comments",
            json={"video_id": "dQw4w9WgXcQ", "max_results": 10},
        )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("key", response.get_json()["error"].lower())

    @patch.dict("os.environ", {"YOUTUBE_API_KEY": "test-secret"})
    @patch("flask_app.app.requests.get", return_value=DummyYouTubeResponse())
    def test_youtube_comments_normalizes_api_response(self, request_get):
        response = self.client.post(
            "/youtube/comments",
            json={"video_id": "dQw4w9WgXcQ", "max_results": 10},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["comments"][0]["author_id"], "author-1")
        self.assertEqual(request_get.call_args.kwargs["timeout"], 10)

    def test_generate_chart_requires_sentiment_counts(self):
        response = self.client.post("/generate_chart", json={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_preprocess_comment_preserves_negation_terms(self):
        processed = app_module.preprocess_comment("This is NOT bad!!!")

        self.assertIn("not", processed)


if __name__ == "__main__":
    unittest.main()
