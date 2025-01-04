from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from typing import Any, Dict
import joblib
import logging
from pathlib import Path
from ..config import MODELS_DIR, MODEL_CONFIGS, RANDOM_SEED

logger = logging.getLogger(__name__)

class ModelFactory:
    """Creates and manages machine learning models."""
    
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'svc': SVC,
            'knn': KNeighborsClassifier,
            'mlp': MLPClassifier,
            'naive_bayes': GaussianNB
        }

    def create_model(self, model_type: str, **kwargs) -> Any:
        """Create a new model instance."""
        if model_type not in self.models:
            raise ValueError(f"Unsupported model type. Choose from: {list(self.models.keys())}")
        print('model_type', model_type)
        # Get default parameters from config
        default_params = MODEL_CONFIGS.get(model_type, {}).get('params', {})
        print('default_params', default_params)
        # Combine with any additional parameters
        params = {**default_params, **kwargs}
        print('params', params)
        logger.info(f"Creating {model_type} model with parameters: {params}")
        print('self.models[model_type]', self.models[model_type])
        return self.models[model_type](**params)

    def save_model(self, model: Any, name: str) -> None:
        """Save model to disk."""
        Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
        path = Path(MODELS_DIR) / f"{name}.joblib"
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, name: str) -> Any:
        """Load model from disk."""
        path = Path(MODELS_DIR) / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"No model found at {path}")
        logger.info(f"Loading model from {path}")
        return joblib.load(path)
