"""Train, compare and persist the SafeDrive AI risk-classification models.

Four algorithms mentioned in the synopsis are trained and compared:
Decision Tree, Random Forest, Logistic Regression and Gradient Boosting.
The best model (by macro F1-score) is saved together with its preprocessing
pipeline and evaluation metadata.

Run from the project root:
    python scripts/train_model.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    CLASS_MAPPING,
    DATASET_PATH,
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    RISK_CLASSES,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_preprocessing import (  # noqa: E402
    build_preprocessor,
    clean_dataframe,
    describe_dataset,
    load_dataset,
    split_features_target,
)


def candidate_models() -> dict:
    """Return the candidate classifiers referenced in the literature review."""
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=12, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=3, random_state=RANDOM_STATE
        ),
    }


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """Compute classification metrics for a fitted pipeline."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba, average="macro", multi_class="ovr")), 4),
    }


def main() -> None:
    print("Loading dataset ...")
    df = clean_dataframe(load_dataset(DATASET_PATH))
    X, y = split_features_target(df)
    y_encoded = y.map(CLASS_MAPPING).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, stratify=y_encoded, random_state=RANDOM_STATE
    )
    print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")

    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}

    for name, estimator in candidate_models().items():
        print(f"\nTraining {name} ...")
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("classifier", estimator)]
        )
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        results[name] = metrics
        fitted[name] = pipeline
        print("  " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name} (macro F1 = {results[best_name]['f1']:.4f})")

    # Permutation importance on the ORIGINAL feature columns (model agnostic).
    perm = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=8,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
        n_jobs=-1,
    )
    importance = [
        {"feature": feature, "importance": round(float(value), 5)}
        for feature, value in zip(FEATURE_COLUMNS, perm.importances_mean)
    ]
    importance.sort(key=lambda item: item["importance"], reverse=True)

    metadata = {
        "project": "SafeDrive AI",
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET_PATH.relative_to(DATASET_PATH.parents[2])),
        "dataset": describe_dataset(df),
        "best_model": best_name,
        "model_comparison": results,
        "feature_importance": importance,
        "features": FEATURE_COLUMNS,
        "classes": RISK_CLASSES,
        "class_mapping": CLASS_MAPPING,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_state": RANDOM_STATE,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved model  -> {MODEL_PATH}")
    print(f"Saved metadata -> {METADATA_PATH}")
    print("\nFeature importance (permutation, macro F1 drop):")
    for item in importance:
        print(f"  {item['feature']:<22} {item['importance']:.5f}")


if __name__ == "__main__":
    main()
