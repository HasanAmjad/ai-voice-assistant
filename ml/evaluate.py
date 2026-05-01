"""Authored by: Lina."""

import os
import sys
import json
import logging
import joblib
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sqlalchemy import text
from ml.train import train_model
from storage.db import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = Path(__file__).parent / "saved_models"
DOCS_DIR = Path(__file__).parent.parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)


def evaluate_model():
    """Score the latest model on the test set and write metrics, plots, and drift stats."""
    logger.info("Starting evaluation...")
    train_result = train_model()
    version_tag = train_result["version_tag"]

    clf = joblib.load(SAVED_MODELS_DIR / "latest_model.pkl")
    vectorizer = joblib.load(SAVED_MODELS_DIR / "tfidf_vectorizer.pkl")
    le = joblib.load(SAVED_MODELS_DIR / "label_encoder.pkl")
    data = train_result["data"]

    if data.get("skip"):
        hash_path = SAVED_MODELS_DIR / "last_data_hash.txt"
        cached = hash_path.read_text().strip() if hash_path.exists() else None
        if hash_path.exists():
            hash_path.unlink()
        from ml.prepare_data import prepare_features
        data = prepare_features()
        if cached:
            hash_path.write_text(cached)

    X_test = data["X_test"]
    y_test = data["y_test"]
    train_df = data["train_df"]

    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    logger.info(f"Accuracy: {test_acc:.4f} | Macro F1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    worst20_idx = np.argsort(per_class_acc)[:20]
    cm_sub = cm[np.ix_(worst20_idx, worst20_idx)]
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm_sub,
        xticklabels=le.classes_[worst20_idx],
        yticklabels=le.classes_[worst20_idx],
        annot=True, fmt="d", cmap="Blues", ax=ax,
    )
    ax.set_title("Confusion Matrix: 20 Most Confused Intents")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    logger.info("Saved confusion_matrix.png")

    feature_names = vectorizer.get_feature_names_out()
    fi = {}
    for i, intent in enumerate(le.classes_):
        coef = clf.coef_[i]
        top10 = coef.argsort()[-10:][::-1]
        fi[intent] = [{"token": feature_names[j], "weight": round(float(coef[j]), 4)} for j in top10]
    with open(DOCS_DIR / "feature_importance.json", "w") as f:
        json.dump(fi, f, indent=2)
    logger.info("Saved feature_importance.json")

    train_counts = train_df["intent_label"].value_counts(normalize=True).sort_index()
    pred_intents = le.inverse_transform(y_pred)
    live_counts = pd.Series(pred_intents).value_counts(normalize=True).reindex(train_counts.index, fill_value=0)
    drift_score = float(jensenshannon(train_counts.values, live_counts.values))
    drift_alert = drift_score > 0.15
    if drift_alert:
        logger.warning(f"DRIFT ALERT: {drift_score:.4f} exceeds 0.15")
    else:
        logger.info(f"Drift score: {drift_score:.4f} - OK")

    metrics = {
        "model_version": version_tag,
        "data_hash": data["data_hash"],
        "training_samples": data["training_samples"],
        "accuracy": round(test_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "drift_score": round(drift_score, 4),
        "drift_alert": drift_alert,
    }
    with open(DOCS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics.json")

    try:
        with get_session() as session:
            session.execute(
                text("UPDATE model_runs SET accuracy=:a, macro_f1=:f, drift_score=:d WHERE model_version=:v"),
                {"a": round(test_acc, 4), "f": round(macro_f1, 4), "d": round(drift_score, 4), "v": version_tag},
            )
            session.commit()
        logger.info(f"Updated model_runs for {version_tag}")
    except Exception as e:
        logger.warning(f"Could not update model_runs: {e}")

    return metrics


if __name__ == "__main__":
    m = evaluate_model()
    print("\n=== Metrics ===")
    for k, v in m.items():
        print(f"  {k}: {v}")
