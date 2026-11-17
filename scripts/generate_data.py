"""Generate the reference road-accident dataset used by SafeDrive AI.

The synopsis does not ship a real accident dataset, so this script creates a
synthetic but realistic dataset for the academic prototype. Every row describes
one observation of a road segment during a one-hour interval. The target
``risk_level`` is derived from a transparent weighted risk-score function so
that the Machine Learning model has genuine, learnable patterns.

Run from the project root:
    python scripts/generate_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    CATEGORY_OPTIONS,
    DATASET_PATH,
    LOCATIONS,
    N_SAMPLES,
    RANDOM_STATE,
    RISK_THRESHOLDS,
)

# Relative contribution of each factor to the underlying risk score.
WEIGHTS = {
    "speed_excess": 0.22,
    "visibility": 0.16,
    "weather": 0.14,
    "road_condition": 0.13,
    "traffic": 0.12,
    "history": 0.10,
    "lighting": 0.08,
    "road_type": 0.05,
}

WEATHER_RISK = {"Clear": 0.00, "Cloudy": 0.15, "Rain": 0.50, "Fog": 0.75, "Snow": 0.85}
ROAD_RISK = {"Dry": 0.00, "Wet": 0.40, "Icy": 0.80, "Flooded": 0.90}
LIGHT_RISK = {"Daylight": 0.00, "Streetlights": 0.30, "Dark": 0.70}
ROADTYPE_RISK = {"Highway": 0.30, "Urban": 0.50, "Rural": 0.35}


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Return a synthetic accident-risk DataFrame."""
    rng = np.random.default_rng(seed)

    loc_idx = rng.integers(0, len(LOCATIONS), n_samples)
    locations = [LOCATIONS[i] for i in loc_idx]

    speed_limit = np.array([loc["speed_limit"] for loc in locations], dtype=float)
    road_type_default = [loc["road_type"] for loc in locations]

    hour = rng.integers(0, 24, n_samples)
    weather = rng.choice(CATEGORY_OPTIONS["weather"], n_samples, p=[0.42, 0.22, 0.18, 0.12, 0.06])
    road_condition = np.array(
        [
            rng.choice(
                CATEGORY_OPTIONS["road_condition"],
                p=[0.55, 0.28, 0.07, 0.10] if w in ("Rain", "Snow") else [0.75, 0.18, 0.03, 0.04],
            )
            for w in weather
        ]
    )
    decided_road_type = np.array(
        [
            rt if rng.random() < 0.90 else rng.choice(CATEGORY_OPTIONS["road_type"])
            for rt in road_type_default
        ]
    )

    # Visibility is lower in fog/snow/heavy rain.
    visibility_base = {
        "Clear": 9.0, "Cloudy": 7.5, "Rain": 5.0, "Fog": 1.8, "Snow": 3.0,
    }
    visibility = np.clip(
        np.array([visibility_base[w] for w in weather]) + rng.normal(0, 0.9, n_samples),
        0.2, 10.0,
    )

    traffic_density = np.clip(rng.normal(48, 22, n_samples), 2, 100)
    # Evening rush hour increases traffic.
    rush = (hour >= 17) & (hour <= 19)
    traffic_density = np.clip(traffic_density + rush * rng.normal(18, 5, n_samples), 2, 100)

    vehicle_count = np.clip((traffic_density * rng.uniform(6, 12, n_samples)).astype(int), 20, 1200)

    # Speed is high on highways, low in rain/fog and dense traffic.
    speed_base = np.where(decided_road_type == "Highway", 0.82, np.where(decided_road_type == "Urban", 0.62, 0.70))
    speed_base = speed_base + rng.normal(0, 0.10, n_samples)
    weather_factor = np.array([{"Clear": 1.0, "Cloudy": 0.97, "Rain": 0.82, "Fog": 0.72, "Snow": 0.68}[w] for w in weather])
    traffic_factor = 1.0 - 0.30 * (traffic_density / 100.0)
    avg_speed = np.clip(speed_limit * speed_base * weather_factor * traffic_factor, 8, 130)

    accident_history = np.clip(rng.gamma(shape=2.2, scale=5.0, size=n_samples), 0, 40).astype(int)
    # Historically dangerous segments tend to remain dangerous.
    accident_history = accident_history + np.array([loc["lat"] > 17.9 for loc in locations]).astype(int) * rng.integers(0, 6, n_samples)
    accident_history = np.clip(accident_history, 0, 40)

    night = (hour < 6) | (hour >= 20)
    lighting = np.array(
        [
            "Daylight" if 7 <= h < 18 else ("Streetlights" if rng.random() < 0.65 else "Dark")
            for h in hour
        ]
    )
    lighting = np.array(
        [
            "Dark" if (n and rng.random() < 0.85) else light
            for n, light in zip(night, lighting)
        ]
    )

    # ------------------------------------------------------------------
    # Transparent risk score (0-100)
    # ------------------------------------------------------------------
    speed_excess = np.clip((avg_speed - speed_limit) / 30.0 + 0.25, 0.0, 1.0)
    visibility_risk = np.clip((10.0 - visibility) / 9.0, 0.0, 1.0)

    score = np.zeros(n_samples, dtype=float)
    score += WEIGHTS["speed_excess"] * speed_excess
    score += WEIGHTS["visibility"] * visibility_risk
    score += WEIGHTS["weather"] * np.array([WEATHER_RISK[w] for w in weather])
    score += WEIGHTS["road_condition"] * np.array([ROAD_RISK[rc] for rc in road_condition])
    score += WEIGHTS["traffic"] * (traffic_density / 100.0)
    score += WEIGHTS["history"] * np.clip(accident_history / 25.0, 0, 1)
    score += WEIGHTS["lighting"] * np.array([LIGHT_RISK[l] for l in lighting])
    score += WEIGHTS["road_type"] * np.array([ROADTYPE_RISK[rt] for rt in decided_road_type])
    score *= 100.0
    score += rng.normal(0, 6.0, n_samples)  # realistic noise
    score = np.clip(score, 0, 100)

    low, high = RISK_THRESHOLDS
    risk_level = np.where(score < low, "Low", np.where(score < high, "Medium", "High"))

    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, n_samples), unit="D")

    df = pd.DataFrame(
        {
            "record_id": [f"SD-{i:05d}" for i in range(1, n_samples + 1)],
            "date": dates.strftime("%Y-%m-%d"),
            "location": [loc["name"] for loc in locations],
            "latitude": [loc["lat"] for loc in locations],
            "longitude": [loc["lng"] for loc in locations],
            "hour": hour,
            "visibility_km": np.round(visibility, 2),
            "traffic_density": np.round(traffic_density, 1),
            "avg_speed_kmph": np.round(avg_speed, 1),
            "speed_limit_kmph": speed_limit.astype(int),
            "accident_history_5yr": accident_history,
            "vehicle_count": vehicle_count,
            "weather": weather,
            "road_condition": road_condition,
            "road_type": decided_road_type,
            "lighting": lighting,
            "risk_level": risk_level,
        }
    )

    # Inject a small amount of missing data to exercise the preprocessing stage.
    for col, frac in (("visibility_km", 0.015), ("weather", 0.010), ("traffic_density", 0.010)):
        missing_idx = rng.choice(n_samples, size=int(n_samples * frac), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


def main() -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(DATASET_PATH, index=False)
    print(f"Generated {len(df):,} records -> {DATASET_PATH}")
    print("\nRisk distribution:")
    print(df["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).to_string())


if __name__ == "__main__":
    main()
