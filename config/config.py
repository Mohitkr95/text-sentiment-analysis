import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Data settings
DATASET_PATH = DATA_DIR / "Restaurant_Reviews.tsv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2

# Text preprocessing settings
MIN_WORD_LENGTH = 2
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95

# Model settings
CROSS_VALIDATION_FOLDS = 5

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create directories if they don't exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True) 