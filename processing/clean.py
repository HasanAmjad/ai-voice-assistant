"""Authored by: Rohan."""

import re
import logging
import sys
import os

from sqlalchemy import select

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.db import get_session
from storage.models import Transcript

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def clean_text(raw: str) -> str:
    """Lowercase, strip, and remove punctuation; return [INAUDIBLE] for empty input."""
    if raw is None:
        return "[INAUDIBLE]"

    text = raw.lower()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ']", "", text)
    text = text.strip()

    if not text:
        return "[INAUDIBLE]"

    return text


def run_cleaning():
    """Apply clean_text to every transcript whose cleaned_transcript is still NULL."""
    with get_session() as session:
        rows_to_clean = session.execute(
            select(Transcript).where(Transcript.cleaned_transcript.is_(None))
        ).scalars().all()

        total_rows = len(rows_to_clean)
        logger.info(f"Found {total_rows} transcripts to clean")

        cleaned_count = 0

        for index, transcript_row in enumerate(rows_to_clean):
            transcript_row.cleaned_transcript = clean_text(transcript_row.raw_transcript)
            cleaned_count += 1

            if (index + 1) % 500 == 0:
                session.commit()
                logger.info(f"Committed cleaning batch: {index + 1}/{total_rows}")

        session.commit()
        logger.info(f"Cleaning complete! {cleaned_count} transcripts cleaned")


clean_all_transcripts = run_cleaning


if __name__ == "__main__":
    run_cleaning()
