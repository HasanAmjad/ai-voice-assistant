"""Authored by: Lina."""

import os
import sys
import hashlib
import logging
import joblib
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from storage.db import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = Path(__file__).parent / "saved_models"
SAVED_MODELS_DIR.mkdir(exist_ok=True)


def prepare_features(force_compute=False, save_artifacts=True):
    """Load cleaned transcripts, fit a TF-IDF vectorizer, and return train/val/test splits.

    Parameters
    ----------
    force_compute : bool
        Skip the data-hash short-circuit and always compute the full feature matrix.
        Used by `evaluate.py` when it needs the test split even though training was skipped.
    save_artifacts : bool
        Whether to write `tfidf_vectorizer.pkl` and `label_encoder.pkl` to disk.
        `evaluate.py` sets this to False when it forces a re-compute, so the existing
        on-disk vectorizer (which the pretrained classifier was fit against) is preserved.
    """
    logger.info("Loading cleaned transcripts from database...")
    with get_session() as session:
        result = session.execute(text("""
            SELECT t.cleaned_transcript, c.intent_label, c.split
            FROM transcripts t
            JOIN calls c ON t.call_id = c.id
            WHERE t.cleaned_transcript IS NOT NULL
              AND t.cleaned_transcript != '[INAUDIBLE]'
        """))
        rows = result.fetchall()

    df = pd.DataFrame(rows, columns=["cleaned_transcript", "intent_label", "split"])
    logger.info(f"Loaded {len(df):,} rows | {df['intent_label'].nunique()} intents")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    sorted_transcripts = train_df["cleaned_transcript"].sort_values().reset_index(drop=True)
    data_hash = hashlib.sha256(sorted_transcripts.to_csv().encode()).hexdigest()
    logger.info(f"Data hash: {data_hash}")

    hash_path = SAVED_MODELS_DIR / "last_data_hash.txt"
    if not force_compute and hash_path.exists() and hash_path.read_text().strip() == data_hash:
        logger.info("Data unchanged since last run -- skipping retraining")
        return {"skip": True, "data_hash": data_hash}

    logger.info("Fitting TF-IDF vectorizer on training data...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=2,
    )
    X_train_raw = vectorizer.fit_transform(train_df["cleaned_transcript"])
    X_test_raw = vectorizer.transform(test_df["cleaned_transcript"])
    logger.info(f"Vocab: {len(vectorizer.vocabulary_):,} | Train matrix: {X_train_raw.shape}")

    le = LabelEncoder()
    y_train_all = le.fit_transform(train_df["intent_label"])
    y_test = le.transform(test_df["intent_label"])
    logger.info(f"Classes: {len(le.classes_)}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(sss.split(X_train_raw, y_train_all))

    X_train = X_train_raw[train_idx]
    X_val = X_train_raw[val_idx]
    y_train = y_train_all[train_idx]
    y_val = y_train_all[val_idx]

    logger.info(f"Split -- Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test_raw.shape[0]:,}")

    if save_artifacts:
        joblib.dump(vectorizer, SAVED_MODELS_DIR / "tfidf_vectorizer.pkl")
        joblib.dump(le, SAVED_MODELS_DIR / "label_encoder.pkl")
        (SAVED_MODELS_DIR / "last_data_hash.txt").write_text(data_hash)
        logger.info("Saved tfidf_vectorizer.pkl, label_encoder.pkl, and last_data_hash.txt")
    else:
        logger.info("save_artifacts=False -- existing tfidf_vectorizer.pkl / label_encoder.pkl / last_data_hash.txt left untouched")

    return {
        "skip": False,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test_raw,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "vectorizer": vectorizer,
        "label_encoder": le,
        "data_hash": data_hash,
        "training_samples": int(X_train.shape[0]),
        "train_df": train_df,
    }


if __name__ == "__main__":
    r = prepare_features()
    if r.get("skip"):
        print("No changes detected -- skipping.")
    else:
        print(f"Features ready. Training samples: {r['training_samples']:,}")
