# =====================================
# Imports and Logger Configuration
# =====================================

import os
import json
import pickle
import logging
import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()  # This loads environment variables from .env

# Configure logger
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

# Stream handler for debugging output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler for error logging
file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

# Formatter for log output
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# =====================================
# Utility Functions
# =====================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data and fill missing values with empty strings."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise

def load_model(model_path: str):
    """Load a trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load a TF-IDF vectorizer from a pickle file."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML configuration file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Generate predictions, classification report, and confusion matrix.

    Returns:
        Tuple: (classification_report_dict, confusion_matrix_array)
    """
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.debug('Model evaluation completed.')
        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def log_confusion_matrix(cm, dataset_name: str):
    """Log a confusion matrix as an MLflow artifact."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        cm_file_path = f'confusion_matrix_{dataset_name}.png'
        plt.savefig(cm_file_path)
        mlflow.log_artifact(cm_file_path)
        plt.close()
        logger.debug(f'Confusion matrix for {dataset_name} logged as artifact.')
    except Exception as e:
        logger.error(f"Failed to log confusion matrix: {e}")
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the run ID and model path to a JSON file for reproducibility."""
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

# =====================================
# Main Evaluation Pipeline
# =====================================

def main():
    # Configure MLflow connection and experiment name
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment('dvc-pipeline-runs')

    with mlflow.start_run() as run:
        try:
            # Get root path and load params
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # Log all parameters in MLflow
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Load trained model and vectorizer
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Load preprocessed test data
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            # Vectorize test data
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Use a few rows for model signature inference
            input_example = pd.DataFrame(X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out())
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))

            # Log the model to MLflow with signature and input example
            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=signature,
                input_example=input_example
            )

            # Save run ID and model info to a local JSON file
            save_model_info(run.info.run_id, "lgbm_model", 'experiment_info.json')

            # Log the TF-IDF vectorizer file
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Evaluate model and collect metrics
            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            # Log classification metrics
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # Log confusion matrix visualization
            log_confusion_matrix(cm, "Test Data")

            # Add metadata tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

            logger.debug("Model evaluation pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")

# =====================================
# Script Entry Point
# =====================================

if __name__ == '__main__':
    main()
