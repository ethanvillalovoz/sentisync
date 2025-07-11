# ====================================
# Imports and Configuration
# ====================================

import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import logging

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ====================================
# Logging Configuration
# ====================================

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

# Console handler for logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler for logging errors
file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel(logging.ERROR)

# Common formatter for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ====================================
# NLTK Setup
# ====================================

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# ====================================
# Text Preprocessing Functions
# ====================================

def preprocess_comment(comment):
    """
    Perform text normalization on a single comment:
    - Lowercasing
    - Whitespace cleanup
    - Removing special characters
    - Stopword filtering (retain sentiment-heavy words)
    - Lemmatization

    Args:
        comment (str): The raw text comment.

    Returns:
        str: The cleaned comment.
    """
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Strip leading/trailing whitespace
        comment = comment.strip()

        # Replace newline characters with space
        comment = re.sub(r'\n', ' ', comment)

        # Remove all non-alphanumeric characters except common punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain negation and contrastive words
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize each word
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment  # Return unprocessed comment as fallback


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text preprocessing to all comments in the 'clean_comment' column.

    Args:
        df (pd.DataFrame): DataFrame containing a 'clean_comment' column.

    Returns:
        pd.DataFrame: DataFrame with normalized 'clean_comment'.
    """
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug('Text normalization completed.')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

# ====================================
# Data Saving Function
# ====================================

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save the cleaned train and test datasets to the interim directory.

    Args:
        train_data (pd.DataFrame): Cleaned training data.
        test_data (pd.DataFrame): Cleaned testing data.
        data_path (str): Base path where /interim directory will be created.
    """
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")

        os.makedirs(interim_data_path, exist_ok=True)  # Safe creation

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)

        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

# ====================================
# Main Pipeline
# ====================================

def main():
    """
    Run the full preprocessing pipeline:
    1. Load raw data
    2. Normalize text
    3. Save processed data
    """
    try:
        logger.debug("Starting data preprocessing...")

        # Load raw data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Raw train and test data loaded successfully.')

        # Normalize comment text
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save the preprocessed data
        save_data(train_processed_data, test_processed_data, data_path='./data')

    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

# ====================================
# Script Entry Point
# ====================================

if __name__ == '__main__':
    main()
