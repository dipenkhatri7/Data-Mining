import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Data paths
DATA_DIR = os.getenv('DATA_DIR', 'data/processed')
HEART_DATA_PATH = os.path.join(DATA_DIR, "heart_cleveland_upload.csv")
DIABETES_DATA_PATH = os.path.join(DATA_DIR, "diabetes.csv")

# Model parameters
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
MODELS_DIR = os.getenv('MODELS_DIR', 'models/saved')

# Training parameters
TEST_SIZE = 0.2
CV_FOLDS = 5

MODEL_CONFIGS = {
    'logistic_regression': {
        'model_type': 'LogisticRegression',
        'params': {
            'C': 1.0,
            'max_iter': 1000,
        }
    },
    'random_forest': {
        'model_type': 'RandomForestClassifier',
        'params': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
        }
    },
}