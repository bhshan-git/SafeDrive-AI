"""SafeDrive AI — native Streamlit dashboard.

Deployed on Streamlit Cloud from:
    https://github.com/bhshan-git/SafeDrive-AI

Run locally with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    CATEGORY_OPTIONS,
    DATASET_PATH,
    METADATA_PATH,
    MODEL_PATH,
    RISK_CLASSES,
)
from src.data_preprocessing import (  # noqa: E402
    clean_dataframe,
    describe_dataset,
    prepare_single_record,
)
from src.recommendation import build_recommendations, key_factors  # noqa: E402

st.set_page_config(
    page_title="SafeDrive AI · Road Accident Risk Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; }
      h1 { letter-spacing: -0.03em; }
      .risk-badge { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; }
      .sub { color: #9aabc6; font-size: .92rem; }
      .sdg { display: inline-block; padding: 3px 10px; border: 1px solid #2b3b57;
             border-radius: 99px; margin-right: 6px; font-size: .74rem; color: #b9c8e3; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model…")
def load_artifacts():
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    df = clean_dataframe(pd.read_csv(DATASET_PATH)) if DATASET_PATH.exists() else pd.DataFrame()
    return model, metadata, df


model, metadata, df = load_artifacts()

if model is None or df.empty:
    st.error("Model or dataset artifacts are missing. Run `python scripts/generate_data.py` and `python scripts/train_model.py`.")
    st.stop()

summary = describe_dataset(df)
comparison = metadata.get("model_comparison", {})
best_name = metadata.get("best_model", "N/A")
best_metrics = comparison.get(best_name, {})


# ---------------------------------------------------------------------------
# Sidebar — inputs & scenarios
# ---------------------------------------------------------------------------
SCENARIOS = {
    "☀ Clear day": {
        "hour": 11, "visibility_km": 9.5, "weather": "Clear", "road_condition": "Dry",
        "road_type": "Urban", "lighting": "Daylight", "traffic_density": 35,
        "avg_speed_kmph": 48, "speed_limit_kmph": 60, "accident_history_5yr": 2,
        "vehicle_count": 300,
    },
    "☂ Monsoon night": {
        "hour": 21, "visibility_km": 3.2, "weather": "Rain", "road_condition": "Wet",
        "road_type": "Highway", "lighting": "Dark", "traffic_density": 72,
        "avg_speed_kmph": 82, "speed_limit_kmph": 90, "accident_history_5yr": 14,
        "vehicle_count": 800,
    },
    "≋ Foggy highway": {
        "hour": 5, "visibility_km": 0.8, "weather": "Fog", "road_condition": "Wet",
        "road_type": "Highway", "lighting": "Dark", "traffic_density": 45,
        "avg_speed_kmph": 78, "speed_limit_kmph": 100, "accident_history_5yr": 19,
        "vehicle_count": 600,
    },
}

defaults = {
    "hour": 12, "visibility_km": 9.0, "weather": "Clear", "road_condition": "Dry",
    "road_type": "Urban", "lighting": "Daylight", "traffic_density": 40,
    "avg_speed_kmph": 55, "speed_limit_kmph": 60, "accident_history_5yr": 3,
    "vehicle_count": 300,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.sidebar.title("⚙️ Observation details")
st.sidebar.markdown('<p class="sub">Pick a quick scenario or fine-tune the conditions.</p>', unsafe_allow_html=True)

scenario_cols = st.sidebar.columns(3)
for col, (label, values) in zip(scenario_cols, SCENARIOS.items()):
    if col.button(label, width="stretch"):
        for key, value in values.items():
            st.session_state[key] = value
        st.rerun()

st.sidebar.divider()

locations = sorted(df["location"].dropna().unique().tolist()) if "location" in df else []
location = st.sidebar.selectbox("Road segment", locations) if locations else None

hour = st.sidebar.slider("Hour of day", 0, 23, key="hour")
visibility = st.sidebar.slider("Visibility (km)", 0.2, 10.0, step=0.1, key="visibility_km")
weather = st.sidebar.selectbox("Weather", CATEGORY_OPTIONS["weather"], key="weather")
road_condition = st.sidebar.selectbox("Road condition", CATEGORY_OPTIONS["road_condition"], key="road_condition")
road_type = st.sidebar.selectbox("Road type", CATEGORY_OPTIONS["road_type"], key="road_type")
lighting = st.sidebar.selectbox("Lighting", CATEGORY_OPTIONS["lighting"], key="lighting")
traffic = st.sidebar.slider("Traffic density (%)", 0, 100, key="traffic_density")
speed = st.sidebar.slider("Average speed (km/h)", 10, 130, key="avg_speed_kmph")
limit = st.sidebar.slider("Speed limit (km/h)", 30, 100, step=10, key="speed_limit_kmph")
history = st.sidebar.slider("Accidents in last 5 years", 0, 40, key="accident_history_5yr")
vehicles = st.sidebar.number_input("Vehicle count", 0, 5000, key="vehicle_count")

st.sidebar.divider()
st.sidebar.markdown(
    f'<p class="sub">Model: <b>{best_name}</b><br>Accuracy: <b>{best_metrics.get("accuracy", 0):.1%}</b> · '
    f'ROC-AUC: <b>{best_metrics.get("roc_auc", 0):.3f}</b></p>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
payload = {
    "hour": hour, "visibility_km": visibility, "traffic_density": traffic,
    "avg_speed_kmph": speed, "speed_limit_kmph": limit,
    "accident_history_5yr": history, "vehicle_count": vehicles,
    "weather": weather, "road_condition": road_condition,
    "road_type": road_type, "lighting": lighting, "location": location,
}

record = prepare_single_record(payload)
encoded = int(model.predict(record)[0])
risk = RISK_CLASSES[encoded]
probs = {name: float(model.predict_proba(record)[0][i]) for i, name in enumerate(RISK_CLASSES)}
risk_score = round(sum(probs[n] * i for i, n in enumerate(RISK_CLASSES)) / 2 * 100, 1)
factors = key_factors(record.iloc[0].to_dict())
recommendations = build_recommendations(record.iloc[0].to_dict(), risk, probs)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛡️ SafeDrive AI")
st.markdown(
    '<p class="sub">AI-powered road accident risk prediction and prevention · '
    'traffic · weather · road · historical accident data</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="sdg">SDG 3 · Good Health</span><span class="sdg">SDG 9 · Innovation</span>'
    '<span class="sdg">SDG 11 · Sustainable Cities</span><span class="sdg">SDG 4 · Quality Education</span>',
    unsafe_allow_html=True,
)
st.caption("Outputs are risk estimates, not certainties. Use them to support — not replace — human judgement.")

st.divider()

# Metrics row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Estimated risk", risk)
c2.metric("Risk score", f"{risk_score:.0f} / 100")
c3.metric(f"{risk} confidence", f"{probs[risk]:.1%}")
c4.metric("Records analysed", f"{summary['total_records']:,}")
st.markdown(
    f'<div class="risk-badge" style="color:{COLORS[risk]}">{risk.upper()} RISK</div>',
    unsafe_allow_html=True,
)
st.progress(min(1.0, max(0.0, probs[risk])), text=f"Model confidence: {probs[risk]:.1%}")

# Confidence breakdown
st.subheader("Prediction confidence")
for level in RISK_CLASSES:
    st.write(f"**{level}** — {probs[level]:.1%}")
    st.progress(probs[level])

# Factors + recommendations
left, right = st.columns(2)
with left:
    st.subheader("🔍 Key contributing factors")
    if factors:
        for f in factors:
            st.markdown(f"- **{f['factor']}** · {f['detail']}")
    else:
        st.info("No strong risk factors detected in this reading.")

with right:
    st.subheader("🚨 Safety recommendations")
    for item in recommendations:
        st.markdown(f"- {item}")

st.divider()

# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
st.header("📊 Dashboard analytics")

plt.rcParams.update({
    "figure.facecolor": "#111a2b", "axes.facecolor": "#111a2b",
    "text.color": "#e8eef9", "axes.labelcolor": "#b9c8e3",
    "xtick.color": "#8b9bb7", "ytick.color": "#8b9bb7",
    "axes.edgecolor": "#2b3b57", "font.size": 9,
})

a1, a2 = st.columns(2)
with a1:
    st.subheader("Risk distribution")
    dist = summary.get("risk_distribution", {})
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(RISK_CLASSES, [dist.get(k, 0) for k in RISK_CLASSES],
           color=[COLORS[k] for k in RISK_CLASSES], width=0.55)
    ax.set_ylabel("Observations")
    for label in ax.get_xticklabels():
        label.set_color(COLORS[label.get_text()])
    st.pyplot(fig, width="stretch")

with a2:
    st.subheader("Model comparison")
    names = list(comparison.keys())
    f1s = [comparison[n]["f1"] for n in names]
    accs = [comparison[n]["accuracy"] for n in names]
    fig, ax = plt.subplots(figsize=(5, 3))
    x = range(len(names))
    ax.bar([i - 0.18 for i in x], f1s, width=0.36, label="Macro F1", color="#38bdf8")
    ax.bar([i + 0.18 for i in x], accs, width=0.36, label="Accuracy", color="#818cf8")
    ax.set_xticks(list(x))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names])
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    st.pyplot(fig, width="stretch")

a3, a4 = st.columns(2)
with a3:
    st.subheader("Feature importance")
    importance = metadata.get("feature_importance", [])[:8][::-1]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.barh([i["feature"].replace("_", " ") for i in importance],
            [i["importance"] for i in importance], color="#22c55e")
    ax.set_xlabel("F1 drop (permutation)")
    st.pyplot(fig, width="stretch")

with a4:
    st.subheader("Observations by weather")
    wdist = summary.get("weather_distribution", {})
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(list(wdist.keys()), list(wdist.values()), color=["#22c55e", "#94a3b8", "#38bdf8", "#818cf8", "#f472b6"][:len(wdist)])
    st.pyplot(fig, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Risk map + table
# ---------------------------------------------------------------------------
st.header("🗺️ High-risk road segments")

if {"latitude", "longitude"}.issubset(df.columns):
    mapping = {"Low": 0, "Medium": 1, "High": 2}
    tmp = df.copy()
    tmp["_score"] = tmp["risk_level"].map(mapping).fillna(0)
    hotspots = (
        tmp.groupby("location")
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"),
             observations=("record_id", "count"), risk_index=("_score", "mean"),
             high_count=("risk_level", lambda s: int((s == "High").sum())))
        .reset_index()
    )
    hotspots["risk_pct"] = (hotspots["risk_index"] / 2 * 100).round(1)
    hotspots = hotspots.sort_values("risk_pct", ascending=False).drop(columns=["risk_index"])

    m1, m2 = st.columns([1.4, 1])
    with m1:
        st.map(hotspots[["latitude", "longitude"]], zoom=8, width="stretch")
    with m2:
        st.dataframe(
            hotspots.rename(columns={
                "location": "Road segment", "risk_pct": "Risk %",
                "observations": "Observations", "high_count": "High-risk readings",
            }),
            hide_index=True, width="stretch", height=420,
        )

st.divider()

with st.expander("ℹ️ About this project"):
    st.write("""
    **SafeDrive AI** is an academic prototype that analyses traffic, weather, road and historical
    accident data to estimate road accident risk. Four Machine Learning algorithms were trained and
    compared; the best model is served here with its preprocessing pipeline. The dashboard also
    provides explainability (permutation feature importance), safety recommendations and a map of
    accident-prone road segments around Satara district.

    **Tech stack:** Python · scikit-learn · pandas · Streamlit · Matplotlib
    **Source:** https://github.com/bhshan-git/SafeDrive-AI
    """)
