"""Flask REST API and dashboard server for SafeDrive AI.

Endpoints
---------
GET  /                         Dashboard UI
GET  /api/health               Service + model status
POST /api/predict              Predict risk for one observation
POST /api/predict/batch        Predict risk for a list of observations
GET  /api/model/info           Model comparison, metrics and feature importance
GET  /api/data/summary         Dataset overview statistics
GET  /api/locations/high-risk  Aggregated high-risk road segments for the map
GET  /api/history              Last 20 predictions made in this session
"""

from __future__ import annotations

import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# Allow `python run.py` and `flask --app src.api:create_app` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .config import (  # noqa: E402
    BASE_DIR,
    DATASET_PATH,
    METADATA_PATH,
    MODEL_PATH,
    RISK_CLASSES,
)
from .data_preprocessing import (  # noqa: E402
    clean_dataframe,
    describe_dataset,
    prepare_single_record,
)
from .recommendation import build_recommendations, key_factors  # noqa: E402

_history: deque[dict[str, Any]] = deque(maxlen=20)


def _load_artifacts() -> tuple[Any | None, dict, dict]:
    """Load the trained model, metadata and dataset summary (if available)."""
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    summary = describe_dataset(clean_dataframe(pd.read_csv(DATASET_PATH))) if DATASET_PATH.exists() else {}
    return model, metadata, summary


def create_app() -> Flask:
    """Application factory — also makes testing straightforward."""
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    model, metadata, data_summary = _load_artifacts()

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok" if model is not None else "degraded",
                "model_loaded": model is not None,
                "best_model": metadata.get("best_model"),
                "model_trained_at": metadata.get("trained_at"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _predict_one(payload: dict) -> dict:
        if model is None:
            raise RuntimeError("Model artifacts are missing. Run scripts/train_model.py first.")

        record_df = prepare_single_record(payload)
        encoded = int(model.predict(record_df)[0])
        risk = RISK_CLASSES[encoded] if 0 <= encoded < len(RISK_CLASSES) else "Medium"

        probabilities: dict[str, float] = {}
        if hasattr(model, "predict_proba"):
            proba_row = model.predict_proba(record_df)[0]
            for idx, class_name in enumerate(RISK_CLASSES):
                probabilities[class_name] = round(float(proba_row[idx]), 4)
        else:
            probabilities = {name: (1.0 if name == risk else 0.0) for name in RISK_CLASSES}

        clean_record = record_df.iloc[0].to_dict()
        result = {
            "risk_level": risk,
            "risk_score": round(sum(probabilities.get(n, 0) * i for i, n in enumerate(RISK_CLASSES)) / 2 * 100, 1),
            "probabilities": probabilities,
            "key_factors": key_factors(clean_record),
            "recommendations": build_recommendations(clean_record, risk, probabilities),
            "location": payload.get("location"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _history.append(result)
        return result

    @app.route("/api/predict", methods=["POST", "OPTIONS"])
    def predict():
        if request.method == "OPTIONS":
            return jsonify({}), 204
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400
        try:
            return jsonify(_predict_one(payload))
        except Exception as exc:  # noqa: BLE001 - API must return a clean error
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/predict/batch", methods=["POST", "OPTIONS"])
    def predict_batch():
        if request.method == "OPTIONS":
            return jsonify({}), 204
        payload = request.get_json(silent=True)
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list) or not records:
            return jsonify({"error": "Provide a non-empty 'records' list."}), 400
        try:
            return jsonify({"predictions": [_predict_one(r if isinstance(r, dict) else {}) for r in records]})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/model/info")
    def model_info():
        return jsonify(
            {
                "best_model": metadata.get("best_model"),
                "trained_at": metadata.get("trained_at"),
                "model_comparison": metadata.get("model_comparison", {}),
                "feature_importance": metadata.get("feature_importance", []),
                "features": metadata.get("features", []),
                "classes": RISK_CLASSES,
                "train_size": metadata.get("train_size"),
                "test_size": metadata.get("test_size"),
            }
        )

    @app.route("/api/data/summary")
    def data_summary_route():
        return jsonify(data_summary)

    @app.route("/api/locations/high-risk")
    def high_risk_locations():
        """Aggregate historic risk by road segment for the Leaflet map."""
        if not DATASET_PATH.exists():
            return jsonify([])
        df = clean_dataframe(pd.read_csv(DATASET_PATH))
        if "location" not in df.columns or "risk_level" not in df.columns:
            return jsonify([])

        mapping = {"Low": 0, "Medium": 1, "High": 2}
        df["_score"] = df["risk_level"].map(mapping).fillna(0)
        grouped = (
            df.groupby("location")
            .agg(
                latitude=("latitude", "mean"),
                longitude=("longitude", "mean"),
                observations=("record_id", "count"),
                avg_risk=("_score", "mean"),
                high_count=("risk_level", lambda s: int((s == "High").sum())),
            )
            .reset_index()
        )
        grouped["risk_pct"] = (grouped["avg_risk"] / 2 * 100).round(1)
        grouped = grouped.sort_values("risk_pct", ascending=False)
        return jsonify(
            [
                {
                    "location": row.location,
                    "latitude": round(float(row.latitude), 4),
                    "longitude": round(float(row.longitude), 4),
                    "observations": int(row.observations),
                    "risk_pct": float(row.risk_pct),
                    "high_count": int(row.high_count),
                }
                for row in grouped.itertuples()
            ]
        )

    @app.route("/api/history")
    def history():
        return jsonify(list(reversed(_history)))

    return app


if __name__ == "__main__":  # pragma: no cover
    create_app().run(host="0.0.0.0", port=5000, debug=True)
