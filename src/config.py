"""Configuration constants for the SafeDrive AI project.

SafeDrive AI: AI-Powered Road Accident Risk Prediction and Prevention System
Department of AI & DS Engineering, YSPM's Yashoda Technical Campus, Satara.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

DATASET_PATH = PROCESSED_DATA_DIR / "safedrive_accident_data.csv"
MODEL_PATH = MODELS_DIR / "accident_risk_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_SAMPLES = 6000
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "hour",                 # 0-23
    "visibility_km",        # 0.2 - 10
    "traffic_density",      # 0-100 (%)
    "avg_speed_kmph",       # 10-130
    "speed_limit_kmph",     # 30-100
    "accident_history_5yr", # count of previous accidents on segment
    "vehicle_count",        # vehicles observed in the interval
]

CATEGORICAL_FEATURES = [
    "weather",
    "road_condition",
    "road_type",
    "lighting",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "risk_level"

RISK_CLASSES = ["Low", "Medium", "High"]
CLASS_MAPPING = {name: idx for idx, name in enumerate(RISK_CLASSES)}

WEATHER_OPTIONS = ["Clear", "Cloudy", "Rain", "Fog", "Snow"]
ROAD_CONDITION_OPTIONS = ["Dry", "Wet", "Icy", "Flooded"]
ROAD_TYPE_OPTIONS = ["Highway", "Urban", "Rural"]
LIGHTING_OPTIONS = ["Daylight", "Streetlights", "Dark"]

CATEGORY_OPTIONS = {
    "weather": WEATHER_OPTIONS,
    "road_condition": ROAD_CONDITION_OPTIONS,
    "road_type": ROAD_TYPE_OPTIONS,
    "lighting": LIGHTING_OPTIONS,
}

# Risk-score thresholds used when generating the reference dataset.
# Chosen from the score distribution to give a useful Low/Medium/High balance.
RISK_THRESHOLDS = (27.0, 39.0)

# ---------------------------------------------------------------------------
# Demo road segments (Satara district, Maharashtra) used for the map view
# ---------------------------------------------------------------------------
LOCATIONS = [
    {"name": "Satara–Pune Highway (NH-48)", "lat": 17.6805, "lng": 73.9969, "road_type": "Highway", "speed_limit": 100},
    {"name": "Satara–Kolhapur Road (NH-166)", "lat": 17.1200, "lng": 74.2100, "road_type": "Highway", "speed_limit": 90},
    {"name": "Karad–Satara Road", "lat": 17.2900, "lng": 74.1800, "road_type": "Highway", "speed_limit": 90},
    {"name": "Wai–Satara Road", "lat": 17.9500, "lng": 73.9300, "road_type": "Urban", "speed_limit": 60},
    {"name": "Mahabaleshwar Ghat Section", "lat": 17.9500, "lng": 73.6500, "road_type": "Highway", "speed_limit": 50},
    {"name": "Satara City – Powar Peth", "lat": 17.6800, "lng": 74.0000, "road_type": "Urban", "speed_limit": 50},
    {"name": "Phaltan Road", "lat": 17.9900, "lng": 74.4200, "road_type": "Rural", "speed_limit": 70},
    {"name": "Koregaon Road", "lat": 17.7000, "lng": 74.1700, "road_type": "Rural", "speed_limit": 70},
    {"name": "Shiroli Phata Junction", "lat": 17.6200, "lng": 73.9800, "road_type": "Urban", "speed_limit": 60},
    {"name": "Akluj–Satara Road", "lat": 17.7700, "lng": 74.6000, "road_type": "Rural", "speed_limit": 70},
    {"name": "Medha Ghat Curve", "lat": 17.7200, "lng": 73.7000, "road_type": "Highway", "speed_limit": 50},
    {"name": "Vaduj–Satara Road", "lat": 17.8800, "lng": 74.2000, "road_type": "Rural", "speed_limit": 70},
]
