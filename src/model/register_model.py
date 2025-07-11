# ======================================
# Model Registration Script with MLflow
# ======================================

import os
import json
import mlflow
import logging
from dotenv import load_dotenv

load_dotenv()  # This loads environment variables from .env

# ======================================
# MLflow Tracking Configuration
# ======================================

# Set the tracking URI for MLflow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

# ======================================
# Logging Configuration
# ======================================

logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

# Console handler for real-time logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler for persistent error logging
file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel(logging.ERROR)

# Common formatter for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ======================================
# Utility Functions
# ======================================

def load_model_info(file_path: str) -> dict:
    """
    Load the run ID and model path info from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing run metadata.

    Returns:
        dict: Dictionary with keys 'run_id' and 'model_path'.
    """
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """
    Register a model in the MLflow Model Registry and transition to Staging.

    Args:
        model_name (str): Desired name of the registered model.
        model_info (dict): Dictionary with 'run_id' and 'model_path'.

    Steps:
        - Constructs model URI using run ID
        - Registers the model in MLflow
        - Transitions the model to the 'Staging' stage
    """
    try:
        # Construct full model URI using run ID and artifact path
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        # Initialize MLflow client for stage transition
        client = mlflow.tracking.MlflowClient()

        # Transition the model version to 'Staging'
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.debug(f'Model "{model_name}" version {model_version.version} registered and moved to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

# ======================================
# Main Entry Point
# ======================================

def main():
    """
    Orchestrates model registration by:
    - Loading model metadata from JSON
    - Registering the model to MLflow
    - Transitioning it to Staging
    """
    try:
        # Path to the JSON file saved during evaluation
        model_info_path = 'experiment_info.json'

        # Load metadata containing run ID and artifact path
        model_info = load_model_info(model_info_path)

        # Define model name to be registered in MLflow
        model_name = "yt_chrome_plugin_model"

        # Register and transition the model
        register_model(model_name, model_info)

    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

# ======================================
# Script Execution
# ======================================

if __name__ == '__main__':
    main()
