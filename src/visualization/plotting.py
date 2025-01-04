# src/visualization/plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd
from typing import Any, Optional
import logging
from pathlib import Path
from ..config import RANDOM_SEED

logger = logging.getLogger(__name__)

class Plotter:
    """Handles all visualization tasks."""
    
    def __init__(self):
        # Set the style using seaborn's set_style instead of plt.style.use
        sns.set_style("whitegrid")
        # Set random seed for reproducibility
        np.random.seed(RANDOM_SEED)
        # Set default figure size
        plt.rcParams['figure.figsize'] = [10, 6]
        # Improve font readability
        plt.rcParams['font.size'] = 12

    def plot_confusion_matrix(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                            save_path: Optional[str] = None) -> None:
        """Plot confusion matrix."""
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   square=True, cbar_kws={'shrink': .8})
        plt.title('Confusion Matrix', pad=20)
        plt.ylabel('True Label', labelpad=10)
        plt.xlabel('Predicted Label', labelpad=10)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Confusion matrix plot saved to {save_path}")
        plt.show()
        plt.close()

    def plot_roc_curve(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      save_path: Optional[str] = None) -> None:
        """Plot ROC curve."""
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model doesn't support probability predictions. Skipping ROC curve.")
            return

        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"ROC curve plot saved to {save_path}")
        plt.show()
        plt.close()

    def plot_feature_importance(self, model: Any, feature_names: list,
                              save_path: Optional[str] = None) -> None:
        """Plot feature importance."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning("Model doesn't support feature importance visualization")
            return

        # Create DataFrame for better plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.tail(10), x='Importance', y='Feature', 
                   palette='viridis')
        plt.title('Top 10 Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Feature importance plot saved to {save_path}")
        plt.show()
        plt.close()

    def plot_distribution(self, data: pd.DataFrame, column: str,
                         save_path: Optional[str] = None) -> None:
        """Plot distribution of a feature."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x=column, kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Distribution plot saved to {save_path}")
        plt.show()
        plt.close()

    def plot_correlation_matrix(self, data: pd.DataFrame,
                              save_path: Optional[str] = None) -> None:
        """Plot correlation matrix of features."""
        plt.figure(figsize=(12, 8))
        corr_matrix = data.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', square=True)
        plt.title('Feature Correlation Matrix')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Correlation matrix plot saved to {save_path}")
        plt.show()
        plt.close()