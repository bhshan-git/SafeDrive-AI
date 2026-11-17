"""Rule-based safety recommendations for SafeDrive AI.

The Machine Learning model estimates *how risky* a situation is; this module
explains *why* and suggests *what to do about it* in plain language. Keeping
the rules separate from the model makes the system easier to audit and extend.
"""

from __future__ import annotations

from typing import Any

WEATHER_ADVICE = {
    "Rain": "Reduce speed and increase following distance on wet surfaces.",
    "Fog": "Use low-beam headlights and fog lamps; keep a safe following distance.",
    "Snow": "Drive slowly, avoid sudden braking and allow extra stopping distance.",
    "Cloudy": "Stay alert; conditions may change quickly.",
}

CONDITION_ADVICE = {
    "Wet": "Expect reduced tyre grip; brake gently and avoid sharp turns.",
    "Icy": "Avoid unnecessary travel; if you must drive, use extreme caution.",
    "Flooded": "Do not enter waterlogged stretches; find an alternate route.",
}

GENERAL_ADVICE = {
    "Low": "Conditions look manageable. Continue to follow traffic rules and stay attentive.",
    "Medium": "Exercise extra caution, obey speed limits and monitor road conditions.",
    "High": "High risk detected. Consider delaying travel, taking an alternate route, or informing traffic authorities.",
}


def build_recommendations(record: dict[str, Any], risk: str, probabilities: dict[str, float]) -> list[str]:
    """Return an ordered list of personalised safety recommendations."""
    advice: list[str] = [GENERAL_ADVICE.get(risk, GENERAL_ADVICE["Medium"])]

    weather = str(record.get("weather", ""))
    if weather in WEATHER_ADVICE:
        advice.append(WEATHER_ADVICE[weather])

    condition = str(record.get("road_condition", ""))
    if condition in CONDITION_ADVICE:
        advice.append(CONDITION_ADVICE[condition])

    try:
        speed = float(record.get("avg_speed_kmph", 0))
        limit = float(record.get("speed_limit_kmph", 0))
    except (TypeError, ValueError):
        speed, limit = 0.0, 0.0
    if limit and speed > limit:
        advice.append(f"Reduce speed from {speed:.0f} km/h to the posted limit of {limit:.0f} km/h.")

    try:
        visibility = float(record.get("visibility_km", 10))
    except (TypeError, ValueError):
        visibility = 10.0
    if visibility < 3:
        advice.append(f"Visibility is only {visibility:.1f} km — slow down and use proper lighting.")

    try:
        traffic = float(record.get("traffic_density", 0))
    except (TypeError, ValueError):
        traffic = 0.0
    if traffic > 75:
        advice.append("Traffic density is high; maintain lane discipline and avoid overtaking.")

    try:
        history = int(float(record.get("accident_history_5yr", 0)))
    except (TypeError, ValueError):
        history = 0
    if history >= 10:
        advice.append(f"This segment has {history} accidents in 5 years — treat it as a known black spot.")

    if str(record.get("lighting", "")) == "Dark":
        advice.append("Poor lighting: use high beam responsibly and watch for pedestrians.")

    # De-duplicate while preserving order.
    seen: set[str] = set()
    return [item for item in advice if not (item in seen or seen.add(item))]


def key_factors(record: dict[str, Any]) -> list[dict[str, str]]:
    """Return the human-readable factors contributing most to this reading."""
    factors: list[dict[str, str]] = []

    try:
        speed, limit = float(record["avg_speed_kmph"]), float(record["speed_limit_kmph"])
        if speed > limit:
            factors.append({"factor": "Overspeeding", "detail": f"{speed:.0f} km/h in a {limit:.0f} km/h zone"})
    except (KeyError, TypeError, ValueError):
        pass

    try:
        vis = float(record["visibility_km"])
        if vis < 5:
            factors.append({"factor": "Low visibility", "detail": f"{vis:.1f} km"})
    except (KeyError, TypeError, ValueError):
        pass

    condition = str(record.get("road_condition", ""))
    if condition in ("Wet", "Icy", "Flooded"):
        factors.append({"factor": "Poor road condition", "detail": condition})

    weather = str(record.get("weather", ""))
    if weather in ("Rain", "Fog", "Snow"):
        factors.append({"factor": "Adverse weather", "detail": weather})

    try:
        traffic = float(record["traffic_density"])
        if traffic > 70:
            factors.append({"factor": "Heavy traffic", "detail": f"{traffic:.0f}% density"})
    except (KeyError, TypeError, ValueError):
        pass

    return factors[:5]
