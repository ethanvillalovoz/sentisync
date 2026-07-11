import os
import io
import pickle
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use('Agg')

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from wordcloud import WordCloud

load_dotenv()

# ================================
# Flask App Initialization
# ================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1_000_000


def get_allowed_origins():
    configured = os.environ.get(
        "SENTISYNC_ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


CORS(
    app,
    resources={r"/*": {"origins": get_allowed_origins()}},
    methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

SENTIMENT_STOPWORD_EXCEPTIONS = {'not', 'but', 'however', 'no', 'yet'}
lemmatizer = WordNetLemmatizer()
MAX_COMMENTS = 500
MAX_COMMENT_LENGTH = 2_000
YOUTUBE_VIDEO_ID_PATTERN = re.compile(r"^[\w-]{11}$")


def get_sentiment_stopwords():
    """Return English stopwords while preserving sentiment-bearing words."""
    try:
        return set(stopwords.words('english')) - SENTIMENT_STOPWORD_EXCEPTIONS
    except LookupError:
        app.logger.warning("NLTK stopwords corpus is unavailable; continuing without stopword removal")
        return set()


def lemmatize_words(words):
    """Lemmatize tokens, falling back to original tokens if WordNet is unavailable."""
    try:
        return [lemmatizer.lemmatize(word) for word in words]
    except LookupError:
        app.logger.warning("NLTK wordnet corpus is unavailable; continuing without lemmatization")
        return words


# ================================
# Preprocessing Function
# ================================
def preprocess_comment(comment):
    """
    Apply preprocessing transformations to a comment:
    - Lowercase
    - Strip whitespace
    - Remove newlines
    - Remove non-alphanumeric characters (except punctuation)
    - Remove stopwords (keep some for sentiment)
    - Lemmatize words
    """
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = get_sentiment_stopwords()
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        comment = ' '.join(lemmatize_words(comment.split()))
        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# ================================
# Model Loading
# ================================
def load_model(model_path, vectorizer_path):
    """
    Load trained model and vectorizer from disk.
    """
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        return model, vectorizer
    except Exception:
        raise

MODEL_PATH = os.environ.get("SENTISYNC_MODEL_PATH", "./lgbm_model.pkl")
VECTORIZER_PATH = os.environ.get("SENTISYNC_VECTORIZER_PATH", "./tfidf_vectorizer.pkl")
model = None
vectorizer = None


def get_model_and_vectorizer():
    """Load model artifacts on first use so health checks do not require native ML libraries."""
    global model, vectorizer

    if model is not None and vectorizer is not None:
        return model, vectorizer

    try:
        model, vectorizer = load_model(MODEL_PATH, VECTORIZER_PATH)
        return model, vectorizer
    except Exception as exc:
        app.logger.exception("Failed to load sentiment model artifacts")
        raise RuntimeError(
            "Sentiment model artifacts could not be loaded. "
            "Check SENTISYNC_MODEL_PATH, SENTISYNC_VECTORIZER_PATH, and native LightGBM dependencies."
        ) from exc


def predict_sentiments(comments):
    """Transform comments and return model predictions."""
    active_model, active_vectorizer = get_model_and_vectorizer()
    transformed_comments = active_vectorizer.transform(comments)
    predictions = active_model.predict(transformed_comments)
    return predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)


def validate_comments(comments):
    if not isinstance(comments, list) or not comments:
        raise ValueError("comments must be a non-empty list")
    if len(comments) > MAX_COMMENTS:
        raise ValueError(f"comments must contain at most {MAX_COMMENTS} items")
    if any(
        not isinstance(comment, str)
        or not comment.strip()
        or len(comment) > MAX_COMMENT_LENGTH
        for comment in comments
    ):
        raise ValueError(
            f"each comment must be a string between 1 and {MAX_COMMENT_LENGTH} characters"
        )
    return comments


def prediction_error(exc):
    app.logger.exception("Sentiment prediction failed")
    return jsonify({"error": "Sentiment prediction is temporarily unavailable."}), 503

# ================================
# API Endpoints
# ================================

@app.route('/')
def home():
    return jsonify({"service": "sentisync", "health": "/health"})

@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "sentisync"})


@app.errorhandler(413)
def payload_too_large(_error):
    return jsonify({"error": "Request payload exceeds the 1 MB limit."}), 413


@app.route('/youtube/comments', methods=['POST'])
def youtube_comments():
    """Fetch top-level comments without exposing the YouTube API key to the extension."""
    data = request.get_json(silent=True) or {}
    video_id = data.get('video_id', '')
    max_results = data.get('max_results', 200)

    if not isinstance(video_id, str) or not YOUTUBE_VIDEO_ID_PATTERN.fullmatch(video_id):
        return jsonify({"error": "video_id must be an 11-character YouTube video ID."}), 400
    if not isinstance(max_results, int) or not 1 <= max_results <= MAX_COMMENTS:
        return jsonify({"error": f"max_results must be between 1 and {MAX_COMMENTS}."}), 400

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        return jsonify({"error": "YouTube comment retrieval is not configured."}), 503

    comments = []
    page_token = None

    try:
        while len(comments) < max_results:
            page_size = min(100, max_results - len(comments))
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": page_size,
                "textFormat": "plainText",
                "key": api_key,
            }
            if page_token:
                params["pageToken"] = page_token

            response = requests.get(
                "https://www.googleapis.com/youtube/v3/commentThreads",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()

            for item in payload.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append(
                    {
                        "text": snippet.get("textOriginal", ""),
                        "timestamp": snippet.get("publishedAt"),
                        "author_id": snippet.get("authorChannelId", {}).get("value", "unknown"),
                    }
                )

            page_token = payload.get("nextPageToken")
            if not page_token:
                break
    except (requests.RequestException, KeyError, TypeError, ValueError):
        app.logger.exception("YouTube comment retrieval failed")
        return jsonify({"error": "YouTube comments could not be retrieved."}), 502

    return jsonify({"comments": comments, "count": len(comments)})

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    """
    Predict sentiment for a list of comments with timestamps.
    Returns original comment, predicted sentiment, and timestamp.
    """
    data = request.get_json(silent=True) or {}
    try:
        comments_data = data.get('comments')
        if not isinstance(comments_data, list) or not comments_data:
            raise ValueError("comments must be a non-empty list")
        if len(comments_data) > MAX_COMMENTS:
            raise ValueError(f"comments must contain at most {MAX_COMMENTS} items")

        comments = validate_comments([item['text'] for item in comments_data])
        timestamps = [item['timestamp'] for item in comments_data]
        author_ids = [item.get('author_id', 'unknown') for item in comments_data]
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        predictions = predict_sentiments(preprocessed_comments)
        predictions = [str(pred) for pred in predictions]
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return prediction_error(exc)

    response = [
        {
            "comment": comment,
            "sentiment": sentiment,
            "timestamp": timestamp,
            "author_id": author_id,
        }
        for comment, sentiment, timestamp, author_id in zip(
            comments, predictions, timestamps, author_ids
        )
    ]
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment for a list of comments.
    Returns original comment and predicted sentiment.
    """
    data = request.get_json(silent=True) or {}
    try:
        comments = validate_comments(data.get('comments'))
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        predictions = predict_sentiments(preprocessed_comments)
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return prediction_error(exc)

    response = [
        {"comment": comment, "sentiment": sentiment}
        for comment, sentiment in zip(comments, predictions)
    ]
    return jsonify(response)

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    """
    Generate a pie chart of sentiment counts.
    Expects sentiment_counts dict in request.
    Returns PNG image.
    """
    try:
        data = request.get_json(silent=True) or {}
        sentiment_counts = data.get('sentiment_counts')
        if not isinstance(sentiment_counts, dict):
            raise ValueError("sentiment_counts must be an object")

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if any(value < 0 for value in sizes) or sum(sizes) == 0:
            raise ValueError("sentiment counts must be non-negative and sum above zero")
        colors = ['#47776d', '#a2a29c', '#b26c62']

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': '#1d1d1f'}
        )
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Chart generation failed")
        return jsonify({"error": "Chart generation is temporarily unavailable."}), 503

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    """
    Generate a word cloud from a list of comments.
    Returns PNG image.
    """
    try:
        data = request.get_json(silent=True) or {}
        comments = validate_comments(data.get('comments'))

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        text = ' '.join(preprocessed_comments)
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='#f7f7f5',
            colormap='Greens',
            stopwords=get_sentiment_stopwords(),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Word-cloud generation failed")
        return jsonify({"error": "Word-cloud generation is temporarily unavailable."}), 503

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    """
    Generate a line graph showing monthly sentiment percentages over time.
    Expects sentiment_data list in request.
    Returns PNG image.
    """
    try:
        data = request.get_json(silent=True) or {}
        sentiment_data = data.get('sentiment_data')
        if not isinstance(sentiment_data, list) or not sentiment_data:
            raise ValueError("sentiment_data must be a non-empty list")
        if len(sentiment_data) > MAX_COMMENTS:
            raise ValueError(f"sentiment_data must contain at most {MAX_COMMENTS} items")

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        monthly_counts = df.resample('ME')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: '#b26c62', 0: '#a2a29c', 1: '#47776d'}
        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Trend graph generation failed")
        return jsonify({"error": "Trend generation is temporarily unavailable."}), 503

# ================================
# Main Entrypoint
# ================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", "8080"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug)
