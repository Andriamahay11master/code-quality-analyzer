from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
DOCS_DIR = BASE_DIR / "docs"

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

TARGET_COLUMN = "quality_label"

MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "random_state": RANDOM_STATE,
    },
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    },
}

FILE_NAMES = {
    "cleaned_data": PROCESSED_DATA_DIR / "cleaned_dataset.csv",
    "trained_model": MODELS_DIR / "trained_model.pkl",
    "label_encoder": MODELS_DIR / "label_encoder.pkl",
    "metrics_report": METRICS_DIR / "metrics.csv",
}