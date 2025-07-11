# ================================
# Imports and Configuration
# ================================

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# ================================
# Logger Setup
# ================================

# Create a logger object for this script
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)  # Set logging level for logger

# Stream handler for console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler for error logs
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

# Define a consistent formatter for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ================================
# Utility Functions
# ================================

def load_params(params_path: str) -> dict:
    """
    Load configuration parameters from a YAML file.

    Args:
        params_path (str): Path to the YAML config file.

    Returns:
        dict: Dictionary of loaded parameters.
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

def load_data(data_url: str) -> pd.DataFrame:
    """
    Load dataset from a remote CSV file.

    Args:
        data_url (str): URL of the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded successfully from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('CSV parsing error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing missing values, duplicates, and empty strings.

    Args:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df.dropna(inplace=True)  # Remove rows with any NaN values
        df.drop_duplicates(inplace=True)  # Remove duplicate rows
        df = df[df['clean_comment'].str.strip() != '']  # Remove empty string rows from 'clean_comment' column
        logger.debug('Data preprocessing complete.')
        return df
    except KeyError as e:
        logger.error('Missing expected column: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save training and testing datasets as CSV files in a 'raw' folder.

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        data_path (str): Path to the directory where data will be saved.
    """
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)  # Create folder if not exists

        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Error saving data: %s', e)
        raise

# ================================
# Main Data Ingestion Pipeline
# ================================

def main():
    """
    Execute the end-to-end data ingestion pipeline:
    1. Load params
    2. Load data
    3. Preprocess
    4. Split
    5. Save
    """
    try:
        # Path to params.yaml located two directories up
        params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        test_size = params['data_ingestion']['test_size']

        # Load dataset
        df = load_data(data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')

        # Clean and prepare data
        final_df = preprocess_data(df)

        # Split into training and test sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        # Save cleaned and split data to file system
        save_data(train_data, test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))

    except Exception as e:
        logger.error('Failed to complete data ingestion pipeline: %s', e)
        print(f"Error: {e}")

# ================================
# Entry Point
# ================================

if __name__ == '__main__':
    main()
