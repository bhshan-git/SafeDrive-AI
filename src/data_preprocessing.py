"""Data cleaning and preparation utilities for SafeDrive AI.

Implements the *Data Preprocessing* stage of the system block diagram:
cleaning, missing-value handling and feature preparation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORY_OPTIONS,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

_REQUIRED_COLUMNS = set(FEATURE_COLUMNS)


def load_dataset(path) -> pd.DataFrame:
    """Load the reference CSV dataset from *path*."""
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` when required feature columns are missing."""
    missing = sorted(_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw dataframe: coerce types and fill missing values.

    * Numeric columns -> ``to_numeric`` then median imputation.
    * Categorical columns -> strip whitespace, map empty strings to NaN,
      then mode imputation (falling back to the first configured option).
    """
    df = df.copy()
    validate_schema(df)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median = df[col].median()
        df[col] = df[col].fillna(median if pd.notna(median) else 0)

    for col in CATEGORICAL_FEATURES:
        series = df[col].astype("string").str.strip()
        series = series.mask(series.str.lower().isin({"", "nan", "none", "null", "unknown"}))
        modes = series.mode(dropna=True)
        fallback = CATEGORY_OPTIONS.get(col, ["Unknown"])[0]
        series = series.fillna(modes.iloc[0] if not modes.empty else fallback)
        df[col] = series.astype(str)

    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series | None]:
    """Return the feature matrix and (if present) the target vector."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build the sklearn preprocessing pipeline used during training.

    Numeric features are median-imputed and standardised. Categorical features
    are most-frequent-imputed and one-hot encoded with a fixed category list so
    that training and inference use exactly the same columns.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=[CATEGORY_OPTIONS[col] for col in CATEGORICAL_FEATURES],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_single_record(payload: dict) -> pd.DataFrame:
    """Convert one API payload dictionary into a clean 1-row feature frame.

    Missing keys are filled with safe defaults so the API is tolerant of
    partially specified requests.
    """
    defaults = {
        "hour": 12,
        "visibility_km": 9.0,
        "traffic_density": 40.0,
        "avg_speed_kmph": 60.0,
        "speed_limit_kmph": 60,
        "accident_history_5yr": 3,
        "vehicle_count": 300,
        "weather": "Clear",
        "road_condition": "Dry",
        "road_type": "Urban",
        "lighting": "Daylight",
    }
    record = {key: payload.get(key, default) for key, default in defaults.items()}
    return clean_dataframe(pd.DataFrame([record]))


def describe_dataset(df: pd.DataFrame) -> dict:
    """Return summary statistics used by the dashboard's data overview."""
    clean = clean_dataframe(df)
    total = len(clean)

    def counts(col: str) -> dict:
        return {str(k): int(v) for k, v in clean[col].value_counts().items()}

    return {
        "total_records": total,
        "total_locations": int(clean["location"].nunique()) if "location" in clean else 0,
        "date_range": {
            "from": str(clean["date"].min()) if "date" in clean else None,
            "to": str(clean["date"].max()) if "date" in clean else None,
        },
        "risk_distribution": counts(TARGET_COLUMN) if TARGET_COLUMN in clean else {},
        "weather_distribution": counts("weather"),
        "road_condition_distribution": counts("road_condition"),
        "location_distribution": counts("location"),
        "averages": {
            col: round(float(clean[col].mean()), 2) for col in NUMERIC_FEATURES
        },
    }
