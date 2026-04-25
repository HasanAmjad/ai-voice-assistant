# AI620 VoiceIntent: Transcription

import logging
import torch
import whisper
from sqlalchemy import select, outerjoin
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.db import get_session
from storage.models import Call, Transcript

# Configuring the logger for this module
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Loading the Whisper model (GPU if available, CPU otherwise)
def load_whisper_model():

    if torch.cuda.is_available():
        device = "cuda"
        model_name = "small"
        logger.info("GPU detected: loading Whisper 'small' on CUDA")
    else:
        device = "cpu"
        model_name = "tiny"
        logger.info("No GPU detected: loading Whisper 'base' on CPU")

    model = whisper.load_model(model_name, device=device)
    return model

# Fetching all the calls that do not yet have a transcript row
def get_pending_calls(session):

    query = (select(Call.id, Call.audio_file_path)
             .select_from(outerjoin(Call, Transcript, Call.id == Transcript.call_id))
             .where(Transcript.id.is_(None)))  # LEFT JOIN + IS NULL = only untranscribed calls

    pending_rows = session.execute(query).fetchall()
    return pending_rows

# Running transcription over all pending audio files and inserting results
def run_transcription():

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
                # Resolve absolute path and check existence before calling Whisper
                audio_path = os.path.abspath(audio_file_path)

                if not os.path.exists(audio_path):
                    logger.error(f"File not found on disk: {audio_path}")
                    error_count += 1
                    continue

                result = model.transcribe(audio_path)
                raw_text = result["text"]

                if not raw_text or raw_text.strip() == "":
                    logger.warning(f"Empty transcript for call_id = {call_id}, inserting [INAUDIBLE]")
                    raw_text = "[INAUDIBLE]"
                    inaudible_count += 1

                new_transcript = Transcript(
                    call_id=call_id,
                    raw_transcript=raw_text,
                    cleaned_transcript=None  # filled by clean.py
                )

                session.add(new_transcript)
                success_count += 1

            except Exception as transcription_error:
                logger.error(f"Whisper failed on call_id = {call_id}, path = {audio_file_path}: {transcription_error}")
                error_count += 1
                continue

            if (index + 1) % 100 == 0:
                session.commit()
                logger.info(f"Committed batch: {index + 1}/{total_pending} processed | success: {success_count} | errors: {error_count}")

        session.commit()  # Final commit for remaining rows

        logger.info(f"Transcription complete! success: {success_count}, inaudible: {inaudible_count}, errors skipped: {error_count}")


if __name__ == "__main__":
    run_transcription()