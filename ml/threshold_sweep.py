"""Authored by: Ibrahim Noor."""

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
from sqlalchemy import text

from storage.db import get_session
from config.settings import CONFIDENCE_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = Path(__file__).parent / "saved_models"
DOCS_DIR = Path(__file__).parent.parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)


def load_test_set():
    """Pull the held-out test split's cleaned transcripts and ground-truth labels."""
    with get_session() as session:
        rows = session.execute(text("""
            SELECT t.cleaned_transcript, c.intent_label
            FROM transcripts t
            JOIN calls c ON t.call_id = c.id
            WHERE c.split = 'test'
              AND t.cleaned_transcript IS NOT NULL
              AND t.cleaned_transcript != '[INAUDIBLE]'
        """)).fetchall()
    return pd.DataFrame(rows, columns=["text", "true"])


def sweep_thresholds():
    """Score every test sample once, then evaluate the policy at a range of thresholds."""
    df = load_test_set()
    logger.info(f"Loaded {len(df):,} test samples")

    clf = joblib.load(SAVED_MODELS_DIR / "latest_model.pkl")
    vectorizer = joblib.load(SAVED_MODELS_DIR / "tfidf_vectorizer.pkl")
    le = joblib.load(SAVED_MODELS_DIR / "label_encoder.pkl")

    X = vectorizer.transform(df["text"])
    proba = clf.predict_proba(X)
    pred_idx = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    pred_label = le.inverse_transform(pred_idx)
    correct = pred_label == df["true"].values
    n = len(df)

    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    rows = []
    for t in thresholds:
        kept = confidence >= t
        n_kept = int(kept.sum())
        n_escalated = n - n_kept
        precision_kept = float(correct[kept].sum() / n_kept) if n_kept > 0 else float("nan")
        wrong_routed = int(((~correct) & kept).sum())
        rows.append({
            "threshold": float(t),
            "escalation_rate": round(n_escalated / n, 4),
            "precision_confident": round(precision_kept, 4) if not np.isnan(precision_kept) else None,
            "wrong_routed_rate": round(wrong_routed / n, 4),
            "coverage_correct_rate": round(int(correct[kept].sum()) / n, 4),
            "n_confident": n_kept,
            "n_escalated": n_escalated,
        })
    return pd.DataFrame(rows), n


def render_plot(sweep_df: pd.DataFrame, selected: float, out_path: Path):
    """Render the dual-axis trade-off plot with the picked threshold marked."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(sweep_df["threshold"], sweep_df["escalation_rate"], "o-",
             color="#e67e22", label="Escalation rate")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Escalation rate", color="#e67e22")
    ax1.tick_params(axis="y", labelcolor="#e67e22")
    ax1.set_ylim(0, 1)
    ax1.axvline(x=selected, linestyle="--", color="red", alpha=0.6,
                label=f"Selected = {selected}")

    ax2 = ax1.twinx()
    ax2.plot(sweep_df["threshold"], sweep_df["precision_confident"], "s-",
             color="#2980b9", label="Precision (when not escalated)")
    ax2.plot(sweep_df["threshold"], sweep_df["coverage_correct_rate"], "^-",
             color="#27ae60", label="Coverage × accuracy")
    ax2.plot(sweep_df["threshold"], sweep_df["wrong_routed_rate"], "v-",
             color="#c0392b", label="Wrong-routed rate")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(0, 1)

    plt.title("Confidence threshold sweep — BANKING77 test set")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2,
               loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def commentary(sweep_df: pd.DataFrame, selected: float, n_total: int) -> str:
    """Produce a one-paragraph English summary of the trade-off at the selected threshold."""
    cur = sweep_df[sweep_df["threshold"] == selected]
    if cur.empty:
        return f"Selected threshold {selected} not in sweep grid."
    cur = cur.iloc[0]
    higher = sweep_df[sweep_df["threshold"] == round(selected + 0.10, 2)]
    lower  = sweep_df[sweep_df["threshold"] == round(selected - 0.10, 2)]
    parts = [
        f"At threshold {selected}: escalates {cur.escalation_rate:.1%} of {n_total} test calls "
        f"({cur.n_escalated} of {n_total}). On the {cur.n_confident} kept predictions the model is "
        f"{cur.precision_confident:.1%} accurate; only {cur.wrong_routed_rate:.1%} of *all* calls "
        f"receive a confidently-wrong canned reply, while {cur.coverage_correct_rate:.1%} are "
        f"auto-handled correctly."
    ]
    if not higher.empty:
        h = higher.iloc[0]
        parts.append(
            f"Raising to {h.threshold} would push escalation to {h.escalation_rate:.1%} and "
            f"reduce wrong-routing to {h.wrong_routed_rate:.1%}."
        )
    if not lower.empty:
        l = lower.iloc[0]
        parts.append(
            f"Lowering to {l.threshold} would drop escalation to {l.escalation_rate:.1%} but "
            f"raise wrong-routing to {l.wrong_routed_rate:.1%}."
        )
    return " ".join(parts)


def main():
    logger.info("Starting threshold sweep")
    sweep_df, n_total = sweep_thresholds()

    csv_path = DOCS_DIR / "threshold_sweep.csv"
    png_path = DOCS_DIR / "threshold_sweep.png"
    sweep_df.to_csv(csv_path, index=False)
    render_plot(sweep_df, CONFIDENCE_THRESHOLD, png_path)

    note = commentary(sweep_df, CONFIDENCE_THRESHOLD, n_total)
    summary = {
        "n_test_samples": n_total,
        "selected_threshold": CONFIDENCE_THRESHOLD,
        "commentary": note,
        "rows": sweep_df.to_dict(orient="records"),
    }
    json_path = DOCS_DIR / "threshold_sweep.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(sweep_df.to_string(index=False))
    print()
    print(note)
    print()
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
