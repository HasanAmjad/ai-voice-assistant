"""Authored by: Rohan."""

import logging
import os
import sys

import torch
import whisper
from sqlalchemy import select, outerjoin

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.db import get_session
from storage.models import Call, Transcript

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_whisper_model():
    """Return a Whisper model on GPU if CUDA is available, otherwise on CPU."""
    if torch.cuda.is_available():
        device = "cuda"
        model_name = "small"
        logger.info("GPU detected: loading Whisper 'small' on CUDA")
    else:
        device = "cpu"
        model_name = "tiny"
        logger.info("No GPU detected: loading Whisper 'tiny' on CPU")

    return whisper.load_model(model_name, device=device)


def get_pending_calls(session):
    """Return rows from `calls` that don't yet have a matching `transcripts` row."""
    query = (
        select(Call.id, Call.audio_file_path)
        .select_from(outerjoin(Call, Transcript, Call.id == Transcript.call_id))
        .where(Transcript.id.is_(None))
    )
    return session.execute(query).fetchall()


def run_transcription():
    """Run Whisper on every pending call and insert the resulting transcripts."""
    model = load_whisper_model()

    with get_session() as session:
        pending_rows = get_pending_calls(session)
        total_pending = len(pending_rows)
        logger.info(f"Found {total_pending} audio files pending transcription")

        success_count = 0
        error_count = 0
        inaudible_count = 0

        for index, (call_id, audio_file_path) in enumerate(pending_rows):
            try:
                audio_path = os.path.abspath(audio_file_path)

                if not os.path.exists(audio_path):
                    logger.error(f"File not found on disk: {audio_path}")
                    error_count += 1
                    continue

                result = model.transcribe(audio_path)
                raw_text = result["text"]

                if not raw_text or raw_text.strip() == "":
                    logger.warning(f"Empty transcript for call_id={call_id}, inserting [INAUDIBLE]")
                    raw_text = "[INAUDIBLE]"
                    inaudible_count += 1

                session.add(Transcript(
                    call_id=call_id,
                    raw_transcript=raw_text,
                    cleaned_transcript=None,
                ))
                success_count += 1

            except Exception as transcription_error:
                logger.error(
                    f"Whisper failed on call_id={call_id}, path={audio_file_path}: {transcription_error}"
                )
                error_count += 1
                continue

            if (index + 1) % 100 == 0:
                session.commit()
                logger.info(
                    f"Committed batch: {index + 1}/{total_pending} | "
                    f"success: {success_count} | errors: {error_count}"
                )

        session.commit()

        logger.info(
            f"Transcription complete! success: {success_count}, "
            f"inaudible: {inaudible_count}, errors skipped: {error_count}"
        )


transcribe_all = run_transcription


if __name__ == "__main__":
    run_transcription()
