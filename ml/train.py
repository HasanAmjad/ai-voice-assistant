"""Authored by: Lina."""

import os
import sys
import logging
import joblib
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from ml.prepare_data import prepare_features
from storage.db import get_session
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = Path(__file__).parent / "saved_models"
SAVED_MODELS_DIR.mkdir(exist_ok=True)


def pretrained_artifacts_exist():
    """Return True if every artifact needed to serve predictions is on disk."""
    required = ["latest_model.pkl", "latest_model.txt", "tfidf_vectorizer.pkl", "label_encoder.pkl"]
    return all((SAVED_MODELS_DIR / name).exists() for name in required)


def train_model():
    """Fit a LogisticRegression intent classifier and persist a versioned model file."""
    logger.info("Starting model training...")

    force_retrain = os.getenv("FORCE_RETRAIN", "").lower() in ("1", "true", "yes")
    if pretrained_artifacts_exist() and not force_retrain:
        version_tag = (SAVED_MODELS_DIR / "latest_model.txt").read_text().strip()
        logger.info(
            f"Pretrained model {version_tag} found in saved_models/ -- skipping training. "
            f"Set FORCE_RETRAIN=1 to retrain from scratch."
        )
        data = prepare_features()
        try:
            with get_session() as session:
                session.execute(
                    text(
                        "INSERT INTO model_runs (model_version, data_hash, training_samples) "
                        "SELECT :v, :h, :s "
                        "WHERE NOT EXISTS (SELECT 1 FROM model_runs WHERE model_version = :v)"
                    ),
                    {"v": version_tag, "h": data.get("data_hash"), "s": data.get("training_samples")},
                )
                session.commit()
        except Exception as e:
            logger.warning(f"Could not seed model_runs for pretrained model: {e}")
        return {"skip": True, "version_tag": version_tag, "data": data}

    data = prepare_features()

    if data.get("skip"):
        latest_txt = SAVED_MODELS_DIR / "latest_model.txt"
        version_tag = latest_txt.read_text().strip() if latest_txt.exists() else "unknown"
        logger.info(f"Data unchanged -- using existing model: {version_tag}")
        return {"skip": True, "version_tag": version_tag, "data": data}

    X_train = data["X_train"]
    X_val = data["X_val"]
    y_train = data["y_train"]
    y_val = data["y_val"]

    version_tag = "model_v" + datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Training {version_tag} on {X_train.shape[0]:,} samples...")

    clf = LogisticRegression(
        max_iter=1000,
        C=5,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    val_acc = accuracy_score(y_val, clf.predict(X_val))
    logger.info(f"Validation accuracy: {val_acc:.4f}")

    joblib.dump(clf, SAVED_MODELS_DIR / f"{version_tag}.pkl")
    joblib.dump(clf, SAVED_MODELS_DIR / "latest_model.pkl")
    (SAVED_MODELS_DIR / "latest_model.txt").write_text(version_tag)
    (SAVED_MODELS_DIR / "last_data_hash.txt").write_text(data["data_hash"])
    logger.info(f"Saved {version_tag}.pkl")

    try:
        with get_session() as session:
            session.execute(
                text("INSERT INTO model_runs (model_version, data_hash, training_samples) VALUES (:v, :h, :s) ON CONFLICT DO NOTHING"),
                {"v": version_tag, "h": data["data_hash"], "s": data["training_samples"]},
            )
            session.commit()
        logger.info(f"Inserted model_runs row for {version_tag}")
    except Exception as e:
        logger.warning(f"Could not insert model_runs: {e}")

    return {
        "skip": False,
        "version_tag": version_tag,
        "val_accuracy": round(val_acc, 4),
        "clf": clf,
        "data": data,
    }


if __name__ == "__main__":
    r = train_model()
    if r.get("skip"):
        print(f"Skipped, using {r['version_tag']}")
    else:
        print(f"Done: {r['version_tag']} | Val accuracy: {r['val_accuracy']:.4f}")
