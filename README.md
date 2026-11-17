# 🛡️ SafeDrive AI

**AI-Powered Road Accident Risk Prediction and Prevention System**

An end-to-end Machine Learning web application that analyses road, traffic, weather and historical accident data to estimate accident risk, classify it as **Low / Medium / High**, and present actionable safety recommendations on an interactive dashboard with charts and a risk map.

**Live Streamlit demo:** https://safedrive-ai.streamlit.app

Two interfaces are provided from the same codebase:

- **Streamlit dashboard** (`streamlit_app.py`) — deployed on Streamlit Cloud.
- **Flask dashboard + REST API** (`run.py` + `src/api.py`) — for local development and API integrations.

> Academic prototype · Department of AI & DS Engineering · YSPM's Yashoda Technical Campus, Satara · 2026–27

---

## 📑 Table of Contents

- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [REST API Reference](#-rest-api-reference)
- [Machine Learning Methodology](#-machine-learning-methodology)
- [Model Results](#-model-results)
- [Sustainable Development Goals](#-sustainable-development-goals)
- [Running Tests](#-running-tests)
- [Limitations](#-limitations)
- [Tech Stack](#-tech-stack)
- [References](#-references)
- [Authors](#-authors)

---

## 🎯 Problem Statement

Road accidents occur when drivers and authorities lack timely information about hazardous road and traffic conditions. Existing approaches often rely only on historical accident data or manual monitoring and do not provide an integrated estimate of risk from multiple changing factors. SafeDrive AI addresses this by combining road, traffic, weather and historical accident information into a single risk estimate and highlighting potentially dangerous locations and conditions.

## ✅ Objectives

- Study the major factors associated with road accident risk.
- Collect and prepare relevant traffic, weather, road and historical accident data.
- Apply Machine Learning techniques to identify risk patterns.
- Estimate and classify accident risk for selected locations / time periods.
- Display risk information using charts, maps and a dashboard.
- Provide early warnings and safety recommendations based on estimated risk.
- Evaluate the model using suitable performance measures.

## ✨ Features

| Feature | Description |
|---|---|
| **Live risk predictor** | Enter conditions through a form (or use quick scenarios) and receive an instant Low / Medium / High prediction with confidence scores. |
| **Explainability** | Key contributing factors (overspeeding, low visibility, adverse weather, etc.) are shown alongside the prediction. |
| **Safety recommendations** | Rule-based, human-readable advice personalised to the observed conditions. |
| **Analytics dashboard** | Risk distribution, model comparison, permutation feature importance and weather observations using Chart.js. |
| **Interactive risk map** | Leaflet map of 12 real road segments around Satara district, colour-coded by historical risk. |
| **Restful API** | JSON endpoints for prediction (single and batch), model info, data summary, high-risk locations and prediction history. |
| **Reproducible pipeline** | Scripts to regenerate the dataset and retrain / re-evaluate all four candidate models. |
| **Tested** | 18 automated tests covering preprocessing, the model artifact and all API endpoints. |

## 🏗 System Architecture

The implementation follows the block diagram in the project synopsis:

```
┌──────────────────────────────────────────────────────────────┐
│  DATA INPUT                                                  │
│  Accident, traffic, weather and road data (CSV, 6,000 rows)   │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  DATA PREPROCESSING                                          │
│  Type coercion · median/mode imputation · one-hot encoding    │
│  · standardisation                                           │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  AI / MACHINE LEARNING MODEL                                 │
│  Decision Tree · Random Forest · Logistic Regression ·       │
│  Gradient Boosting  →  best model selected by macro F1       │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  RISK CLASSIFICATION                                         │
│  Low / Medium / High  +  class probabilities                 │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  RESULT DISPLAY                                              │
│  Flask dashboard · charts · Leaflet risk map · recommendations│
└──────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
SafeDrive-AI/
├── data/
│   └── processed/
│       └── safedrive_accident_data.csv   # 6,000 generated observations
├── models/
│   ├── accident_risk_pipeline.joblib     # trained preprocessing + classifier
│   └── model_metadata.json               # metrics, importance, dataset stats
├── scripts/
│   ├── generate_data.py                  # dataset generator
│   └── train_model.py                    # training, comparison, evaluation
├── src/
│   ├── __init__.py
│   ├── config.py                         # paths, schema, constants, locations
│   ├── data_preprocessing.py             # cleaning + sklearn preprocessor
│   ├── recommendation.py                 # safety advice + key factors
│   └── api.py                            # Flask app + REST endpoints
├── streamlit_app.py                      # Streamlit Cloud dashboard
├── .streamlit/config.toml                # Streamlit theme
├── runtime.txt                           # Python 3.12 for Streamlit Cloud
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html                        # dashboard UI
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── run.py                                # python run.py -> http://127.0.0.1:5000
```

## 🚀 Quick Start

### 1. Prerequisites

- Python **3.10+** (verified on 3.12)
- pip

### 2. Clone and install

```bash
git clone https://github.com/<your-username>/SafeDrive-AI.git
cd SafeDrive-AI
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the app

The generated dataset and trained model are already committed, so you can start immediately.

**Streamlit dashboard (same UI as the live demo):**

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501**.

**Flask dashboard + REST API:**

```bash
python run.py
```

Open **http://127.0.0.1:5000**.

### 4. (Optional) Rebuild data and model from scratch

```bash
python scripts/generate_data.py   # regenerate data/processed/safedrive_accident_data.csv
python scripts/train_model.py     # retrain all models and overwrite models/ artifacts
python -m pytest -q               # verify everything still works
```

## 🖥 Usage

1. Open the dashboard and scroll to **Risk Predictor**.
2. Pick a road segment and adjust the sliders / dropdowns — or click a quick scenario:
   - ☀ *Clear day* — low-risk baseline
   - ☂ *Monsoon night* — rainy, dark, congested highway
   - ≋ *Foggy highway* — near-zero visibility, overspeeding
3. Press **Predict Accident Risk**.
4. Review the estimated risk level, confidence bars, contributing factors and recommendations.
5. Scroll to **Analytics** for model metrics and to **Risk Map** for black-spot locations.

## 🔌 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service status and loaded model info |
| `POST` | `/api/predict` | Predict risk for one observation |
| `POST` | `/api/predict/batch` | Predict risk for `{"records": [...]}` |
| `GET` | `/api/model/info` | Model comparison, metrics, feature importance |
| `GET` | `/api/data/summary` | Dataset statistics and distributions |
| `GET` | `/api/locations/high-risk` | Road segments with aggregated risk (for map) |
| `GET` | `/api/history` | Last 20 predictions made in this session |

### Example request

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 2, "visibility_km": 1.0, "traffic_density": 88,
    "avg_speed_kmph": 100, "speed_limit_kmph": 60,
    "accident_history_5yr": 20, "vehicle_count": 900,
    "weather": "Fog", "road_condition": "Wet",
    "road_type": "Highway", "lighting": "Dark",
    "location": "Mahabaleshwar Ghat Section"
  }'
```

### Example response (abridged)

```json
{
  "risk_level": "High",
  "risk_score": 98.0,
  "probabilities": {"Low": 0.0, "Medium": 0.01, "High": 0.99},
  "key_factors": [
    {"factor": "Overspeeding", "detail": "100 km/h in a 60 km/h zone"},
    {"factor": "Low visibility", "detail": "1.0 km"}
  ],
  "recommendations": [
    "High risk detected. Consider delaying travel ...",
    "Use low-beam headlights and fog lamps ..."
  ]
}
```

## 🧠 Machine Learning Methodology

- **Dataset:** 6,000 synthetic but realistic hourly observations across 12 road segments in Satara district. The target label is generated from a transparent weighted risk-score function (speed excess, visibility, weather, road condition, traffic, accident history, lighting, road type) with Gaussian noise, giving a genuine learnable signal.
- **Train/test split:** stratified 80/20 (4,800 train / 1,200 test), `random_state=42`.
- **Preprocessing:** numeric features → median imputation + `StandardScaler`; categorical features → most-frequent imputation + `OneHotEncoder` with a fixed category list (unknown-safe).
- **Candidates:** Decision Tree, Random Forest, Logistic Regression, Gradient Boosting.
- **Selection:** highest macro F1-score on the held-out test set.
- **Explainability:** permutation importance (F1 drop) computed on the original feature columns.
- **Inference:** the entire preprocessing + classifier pipeline is persisted as one `joblib` artifact, so training and serving use identical transformations.

## 📊 Model Results

| Model | Accuracy | Precision | Recall | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.719 | 0.682 | 0.662 | 0.670 | 0.850 |
| Random Forest | 0.748 | 0.712 | 0.720 | 0.716 | 0.890 |
| **Logistic Regression (best)** | **0.753** | **0.723** | **0.740** | **0.730** | **0.898** |
| Gradient Boosting | 0.753 | 0.720 | 0.700 | 0.708 | 0.897 |

> Exact figures are stored in `models/model_metadata.json` and rendered live in the dashboard.

Top factors by permutation importance: **visibility**, **weather**, **road condition**, **lighting**, **accident history**.

## 🌍 Sustainable Development Goals

| SDG | Contribution |
|---|---|
| **3 – Good Health & Well-Being** | Timely warnings help reduce exposure to unsafe road conditions. |
| **9 – Industry, Innovation & Infrastructure** | AI/ML and analytics applied to intelligent transport infrastructure. |
| **11 – Sustainable Cities & Communities** | Identifies high-risk segments to support traffic planning and road improvements. |
| **4 – Quality Education** | Hands-on learning in AI, ML, preprocessing, visualisation and problem solving. |

## 🧪 Running Tests

```bash
python -m pytest -q
```

Expected: `18 passed`.

## ⚠️ Limitations

- Prediction quality depends on the accuracy, completeness and coverage of input data.
- The model may not perform equally well in every region or road condition.
- Real-time prediction requires timely and reliable traffic and weather data.
- Unexpected driver behaviour or sudden events may not be represented in historical data.
- The system provides **risk estimates, not certainty** that an accident will occur.
- Deployment on live roads requires extensive validation and approval from relevant authorities.

## 🛠 Tech Stack

**Backend / ML:** Python, Flask, Streamlit, scikit-learn, pandas, NumPy, joblib, Matplotlib
**Frontend:** HTML5, CSS3, JavaScript, Chart.js, Leaflet + OpenStreetMap (Flask UI); Streamlit widgets (Streamlit UI)
**Tooling:** pytest, Git/GitHub

## 📚 References

1. World Health Organization, *Global Status Report on Road Safety 2023*, Geneva: WHO, 2023.
2. World Health Organization, "Road traffic injuries," Fact Sheet. https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries
3. scikit-learn developers, *scikit-learn: Machine Learning in Python*. https://scikit-learn.org/stable/
4. The pandas development team, *pandas documentation*. https://pandas.pydata.org/docs/
5. NumPy Developers, *NumPy documentation*. https://numpy.org/doc/
6. United Nations, *Transforming Our World: The 2030 Agenda for Sustainable Development*, 2015.
7. National Highway Traffic Safety Administration, *Traffic Safety Facts and Data*. https://www.nhtsa.gov/
8. Python Software Foundation, *Python Documentation*. https://docs.python.org/3/

## 👥 Authors

- **Bhushan Mahavir Niprul** (24063181263040)
- **Shaniraj Shivaji Pilane** (24063181263044)

*Under the guidance of **Prof. Payal S. More**, Department of AI & DS Engineering.*

---

<p align="center">Built with Python &amp; a commitment to safer roads. 🛡️</p>
