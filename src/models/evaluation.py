from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
from typing import Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation and performance metrics."""

    @staticmethod
    def evaluate(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Calculate various model performance metrics."""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }

        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)

        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics

    @staticmethod
    def get_confusion_matrix(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """Generate confusion matrix."""
        y_pred = model.predict(X_test)
        return confusion_matrix(y_test, y_pred)

    @staticmethod
    def get_classification_report(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> str:
        """Generate detailed classification report."""
        y_pred = model.predict(X_test)
        return classification_report(y_test, y_pred)