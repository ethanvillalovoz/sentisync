FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libpng-dev \
    libfreetype6-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency metadata first for better Docker layer caching
COPY requirements-api.txt setup.py ./
COPY src ./src

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt \
    && python3 -m nltk.downloader stopwords wordnet

# Copy the application, model artifacts, extension, and docs
COPY . /app

EXPOSE 8080

# Run the Flask app behind a production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "flask_app.app:app"]
