import logging
import os
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from . import config

# Setup logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dataset(dataset_type: str) -> pd.DataFrame:
    """Load either heart disease or diabetes dataset."""
    if dataset_type == "heart":
        path = config.HEART_DATA_PATH
    elif dataset_type == "diabetes":
        path = config.DIABETES_DATA_PATH
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    logger.info(f"Loading {dataset_type} dataset from {path}")
    return pd.read_csv(path)

def prepare_data(df: pd.DataFrame, target_col: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features and target variables."""
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    return X, y

def split_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and testing sets."""
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y
    )