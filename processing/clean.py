# AI620 VoiceIntent: Transcript Cleaning

import re
import logging

from sqlalchemy import select

from storage.db import get_session
from storage.models import Transcript

# Configuring the logger for this module

logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Normalising a single raw transcript string into a clean lowercase string

def clean_text(raw: str) -> str:

    text = raw.lower() # Step A: convert to lowercase

    text = text.strip() # Step B: remove leading and trailing whitespace

    text = re.sub(r"\s+", " ", text) # Step C: collapse all internal whitespace into a single space

    text = re.sub(r"[^a-z0-9 ']", "", text) # Step D: remove everything except letters, digits, spaces, and apostrophes

    text = text.strip() # Stripping again in case punctuation was at the edges

    if not text: # Step E: return the inaudible marker if nothing remains
        return "[INAUDIBLE]"

    return text

# Applying clean_text to all rows where cleaned_transcript has not been filled yet

def run_cleaning():

    with get_session() as session:

        rows_to_clean = session.execute(select(Transcript).where(Transcript.cleaned_transcript.is_(None)) # WHERE clause makes this idempotent
                                        ).scalars().all()

        total_rows = len(rows_to_clean)
        logger.info(f"Found {total_rows} transcripts to clean")

        cleaned_count = 0

        for index, transcript_row in enumerate(rows_to_clean):

            transcript_row.cleaned_transcript = clean_text(transcript_row.raw_transcript)
            cleaned_count = cleaned_count + 1

            if (index + 1) % 500 == 0: # Committing every 500 rows to keep transactions short
                session.commit()
                logger.info(f"Committed cleaning batch: {index + 1}/{total_rows} done so far")

        session.commit() # Final commit for the remaining rows

        logger.info(f"Cleaning complete! {cleaned_count} transcripts cleaned")


if __name__ == "__main__":
    run_cleaning()
