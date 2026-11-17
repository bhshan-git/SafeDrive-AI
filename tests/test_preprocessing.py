"""Unit tests for the data preprocessing stage."""

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.data_preprocessing import (
    clean_dataframe,
    describe_dataset,
    prepare_single_record,
    validate_schema,
)


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "hour": 8, "visibility_km": 8.0, "traffic_density": 40.0,
        "avg_speed_kmph": 55.0, "speed_limit_kmph": 60,
        "accident_history_5yr": 2, "vehicle_count": 300,
        "weather": "Clear", "road_condition": "Dry",
        "road_type": "Urban", "lighting": "Daylight",
    }])


def test_validate_schema_accepts_valid_frame():
    validate_schema(_valid_frame())


def test_validate_schema_rejects_missing_column():
    df = _valid_frame().drop(columns=["weather"])
    with pytest.raises(ValueError, match="weather"):
        validate_schema(df)


def test_clean_dataframe_fills_missing_numeric_with_median():
    df = pd.concat([_valid_frame(), _valid_frame()], ignore_index=True)
    df.loc[0, "visibility_km"] = np.nan
    cleaned = clean_dataframe(df)
    assert cleaned["visibility_km"].notna().all()
    assert cleaned.loc[0, "visibility_km"] == pytest.approx(8.0)


def test_clean_dataframe_fills_missing_categorical_with_mode():
    df = pd.concat([_valid_frame(), _valid_frame()], ignore_index=True)
    df.loc[0, "weather"] = "   "
    cleaned = clean_dataframe(df)
    assert cleaned["weather"].tolist() == ["Clear", "Clear"]


def test_prepare_single_record_returns_all_features():
    record = prepare_single_record({"hour": 23, "weather": "Fog"})
    assert record.shape == (1, len(FEATURE_COLUMNS))
    assert list(record.columns) == FEATURE_COLUMNS
    assert record.loc[0, "hour"] == 23
    assert record.loc[0, "weather"] == "Fog"


def test_describe_dataset_returns_expected_keys():
    frame = _valid_frame()
    frame["risk_level"] = "Low"
    frame["location"] = "Test Road"
    frame["date"] = "2025-01-01"
    frame["latitude"], frame["longitude"] = 17.6, 74.0
    summary = describe_dataset(frame)
    assert summary["total_records"] == 1
    assert "risk_distribution" in summary
    assert "weather_distribution" in summary
