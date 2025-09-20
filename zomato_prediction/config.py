import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data file paths
RAW_DATA_FILE = RAW_DATA_DIR / "zomato_raw.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "zomato_master_processed.parquet"
GEO_FEATURES_FILE = PROCESSED_DATA_DIR / "zomato_geo_features_final.parquet"
NLP_FEATURES_FILE = PROCESSED_DATA_DIR / "zomato_nlp_features_final.parquet"

# Model settings
MODEL_FILE = MODELS_DIR / "best_model.pkl"
MODEL_NAME = "zomato_rating_predictor"
MODEL_VERSION = "v1.0"

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = "zomato_rating_prediction"

# Geographic settings (UPDATE THESE BASED ON YOUR NOTEBOOK 04)
BANGALORE_CENTER_LAT = 12.9716  # Update with your city center coordinates
BANGALORE_CENTER_LON = 77.5946
BANGALORE_BOUNDS = {
    "lat_min": 12.8,  # Update with your analysis
    "lat_max": 13.1,
    "lon_min": 77.4,
    "lon_max": 77.8,
}

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"

# Redis settings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_CACHE_TTL = 3600  # 1 hour cache

# Model training settings
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# TODO: Update these parameters based on your best results from notebook 06
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_SEED,
}

# Validation settings - TODO: Update based on your data analysis
REQUIRED_FIELDS = [
    "restaurant_name",
    "location",
    "cuisines",
    # TODO: Add other required fields from your analysis
]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
