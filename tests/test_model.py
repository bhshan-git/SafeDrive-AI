"""Tests for the persisted model artifact and metadata."""

import json

import joblib
import pytest

from src.config import METADATA_PATH, MODEL_PATH, RISK_CLASSES
from src.data_preprocessing import prepare_single_record


@pytest.fixture(scope="module")
def artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        pytest.skip("Model artifacts missing — run scripts/train_model.py first.")
    return joblib.load(MODEL_PATH), json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def test_model_loads_and_predicts(artifacts):
    model, _ = artifacts
    record = prepare_single_record({
        "hour": 3, "visibility_km": 1.0, "traffic_density": 80,
        "avg_speed_kmph": 90, "speed_limit_kmph": 60,
        "accident_history_5yr": 15, "vehicle_count": 700,
        "weather": "Fog", "road_condition": "Wet",
        "road_type": "Highway", "lighting": "Dark",
    })
    prediction = int(model.predict(record)[0])
    assert prediction in {0, 1, 2}

    probabilities = model.predict_proba(record)[0]
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-4)


def test_metadata_is_complete(artifacts):
    _, metadata = artifacts
    assert metadata["classes"] == RISK_CLASSES
    assert metadata["best_model"]
    assert metadata["model_comparison"]
    assert metadata["feature_importance"]
    assert metadata["train_size"] > 0


def test_recommendations_are_non_empty():
    from src.recommendation import build_recommendations

    advice = build_recommendations(
        {"weather": "Rain", "road_condition": "Wet", "avg_speed_kmph": 90,
         "speed_limit_kmph": 60, "visibility_km": 2, "traffic_density": 90,
         "accident_history_5yr": 12, "lighting": "Dark"},
        "High", {"Low": 0, "Medium": 0.1, "High": 0.9},
    )
    assert len(advice) >= 4
    assert any("speed" in a.lower() for a in advice)
