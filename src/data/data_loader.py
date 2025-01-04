import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging
from ..config import HEART_DATA_PATH, DIABETES_DATA_PATH, TEST_SIZE, RANDOM_SEED

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles data loading and basic preprocessing for both datasets."""
    
    def __init__(self):
        self.supported_datasets = {
            'heart': HEART_DATA_PATH,
            'diabetes': DIABETES_DATA_PATH
        }

    def load_data(self, dataset_type: str) -> pd.DataFrame:
        """Load specified dataset."""
        if dataset_type not in self.supported_datasets:
            raise ValueError(f"Unsupported dataset type. Choose from: {list(self.supported_datasets.keys())}")
        
        path = self.supported_datasets[dataset_type]
        print("path", path)
        logger.info(f"Loading {dataset_type} dataset from {path}")
        return pd.read_csv(path)

    def prepare_data(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target variables."""
        print('target_col', target_col)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        X = df.drop(columns=[target_col])
        y = df[target_col].values
        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        print("X", X)
        return train_test_split(X, y, test_size=TEST_SIZE, stratify=y)

    def load_and_prepare_data(self, dataset_type: str, target_col: str = 'target') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Complete pipeline for loading and preparing data."""
        print("dataset_type", dataset_type)
        df = self.load_data(dataset_type)
        X, y = self.prepare_data(df, target_col)
        return self.split_data(X, y)