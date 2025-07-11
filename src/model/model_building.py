# ======================================
# Imports and Logger Configuration
# ======================================

import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logger for model building process
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

# Console log handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File log handler (only logs errors)
file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

# Common log format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ======================================
# Utility Functions
# ======================================

def load_params(params_path: str) -> dict:
    """
    Load model and preprocessing parameters from a YAML file.

    Args:
        params_path (str): Path to the YAML configuration file.

    Returns:
        dict: Dictionary of parameters.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML parsing error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading parameters: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load cleaned training data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame with NaNs filled as empty strings.
    """
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('CSV parsing error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading data: %s', e)
        raise


def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """
    Apply TF-IDF transformation with n-grams on cleaned comments.

    Args:
        train_data (pd.DataFrame): Preprocessed training data.
        max_features (int): Maximum number of features to retain.
        ngram_range (tuple): N-gram range for vectorization.

    Returns:
        tuple: Transformed TF-IDF features and labels.
    """
    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        # Fit and transform the comments
        X_train_tfidf = vectorizer.fit_transform(X_train)
        logger.debug(f"TF-IDF transformation complete. Train shape: {X_train_tfidf.shape}")

        # Save the vectorizer for future inference
        with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
        logger.debug('TF-IDF vectorizer saved.')

        return X_train_tfidf, y_train
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise


def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
    """
    Train a LightGBM classifier using the provided training data.

    Args:
        X_train (np.ndarray): TF-IDF feature matrix.
        y_train (np.ndarray): Corresponding labels.
        learning_rate (float): Learning rate for boosting.
        max_depth (int): Maximum depth of each tree.
        n_estimators (int): Number of boosting iterations.

    Returns:
        lgb.LGBMClassifier: Trained LightGBM model.
    """
    try:
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            is_unbalance=True,
            class_weight='balanced',
            reg_alpha=0.1,
            reg_lambda=0.1,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        model.fit(X_train, y_train)
        logger.debug('LightGBM model training completed.')
        return model
    except Exception as e:
        logger.error('Error during LightGBM model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """
    Save the trained model to disk using pickle.

    Args:
        model: Trained model object.
        file_path (str): Output path to save the model.
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def get_root_directory() -> str:
    """
    Resolve the project root directory (two levels up from script location).

    Returns:
        str: Absolute path to the root directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

# ======================================
# Main Orchestration
# ======================================

def main():
    """
    Main pipeline:
    1. Load parameters from config.
    2. Load cleaned training data.
    3. Apply TF-IDF feature extraction.
    4. Train a LightGBM model.
    5. Save both model and TF-IDF vectorizer.
    """
    try:
        logger.debug("Starting model building pipeline...")

        # Get root path and load configuration
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, 'params.yaml'))

        # Extract parameters for feature engineering and model training
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])
        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        # Load preprocessed training data
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

        # Apply TF-IDF transformation
        X_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)

        # Train LightGBM model
        best_model = train_lgbm(X_train_tfidf, y_train, learning_rate, max_depth, n_estimators)

        # Save trained model
        save_model(best_model, os.path.join(root_dir, 'lgbm_model.pkl'))

        logger.debug("Model building pipeline completed successfully.")
    except Exception as e:
        logger.error('Failed to complete model building process: %s', e)
        print(f"Error: {e}")

# ======================================
# Entry Point
# ======================================

if __name__ == '__main__':
    main()
