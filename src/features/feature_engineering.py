from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import Optional, Union
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Handles feature engineering and preprocessing."""
    
    def __init__(self):
        self.pipeline = None
        self.scaler_type = None

    def create_pipeline(self, scaler_type: str = 'standard') -> Pipeline:
        """Create preprocessing pipeline."""
        if scaler_type not in ['standard', 'minmax', None]:
            raise ValueError("Scaler type must be 'standard', 'minmax', or None")
        
        steps = [
            ('imputer', SimpleImputer(strategy='median'))
        ]
        
        if scaler_type == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif scaler_type == 'minmax':
            steps.append(('scaler', MinMaxScaler()))
            
        self.pipeline = Pipeline(steps)
        self.scaler_type = scaler_type
        return self.pipeline

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fit pipeline and transform data."""
        if self.pipeline is None:
            self.create_pipeline()
        return self.pipeline.fit_transform(X)

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform data using fitted pipeline."""
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
        return self.pipeline.transform(X)