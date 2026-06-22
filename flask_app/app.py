import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server-side image generation

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.dates as mdates
import pickle

# ================================
# Flask App Initialization
# ================================
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

SENTIMENT_STOPWORD_EXCEPTIONS = {'not', 'but', 'however', 'no', 'yet'}
lemmatizer = WordNetLemmatizer()


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
    dense_comments = transformed_comments.toarray()
    predictions = active_model.predict(dense_comments)
    return predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)

# ================================
# API Endpoints
# ================================

@app.route('/')
def home():
    """Basic health check endpoint."""
    return "Welcome to our flask api"

@app.route('/health')
def health():
    """Machine-readable health check endpoint."""
    return jsonify({"status": "ok", "service": "sentisync"})

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    """
    Predict sentiment for a list of comments with timestamps.
    Returns original comment, predicted sentiment, and timestamp.
    """
    data = request.get_json(silent=True) or {}
    comments_data = data.get('comments')
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        predictions = predict_sentiments(preprocessed_comments)
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment for a list of comments.
    Returns original comment and predicted sentiment.
    """
    data = request.get_json(silent=True) or {}
    comments = data.get('comments')
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        predictions = predict_sentiments(preprocessed_comments)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

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
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Draw pie as a circle

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    """
    Generate a word cloud from a list of comments.
    Returns PNG image.
    """
    try:
        data = request.get_json(silent=True) or {}
        comments = data.get('comments')
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        text = ' '.join(preprocessed_comments)
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=get_sentiment_stopwords(),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

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
        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: 'red', 0: 'gray', 1: 'green'}
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
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

# ================================
# Main Entrypoint
# ================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", "8080"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug)
