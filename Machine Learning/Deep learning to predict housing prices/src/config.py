
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "housing.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "model.keras"

SCALER_PATH = MODEL_DIR / "scaler.pkl"

HISTORY_PATH = MODEL_DIR / "history.pkl"

TEST_SIZE = 0.15

VAL_SIZE = 0.15

RANDOM_STATE = 42

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 0.001

INPUT_DIM = 7

TARGET = "median_house_value"

DROP_COLUMNS = [
    "ocean_proximity",
    "total_bedrooms"
]
