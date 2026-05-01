"""Authored by: Ibrahim Noor."""

from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from logging_monitoring.logger import get_logger

logger = get_logger(__name__)


@task(
    retries=3,
    retry_delay_seconds=10,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=24),
)
def ingest_task():
    """Download the BANKING77 dataset from HuggingFace, cached for 24 hours."""
    logger.info("Starting data ingestion task")
    try:
        from ingestion.download_dataset import download
        download()
        logger.info("Data ingestion completed successfully")
    except Exception as e:
        logger.error(f"Ingestion task failed: {str(e)}", exc_info=True)
        raise


@task(retries=2, retry_delay_seconds=30)
def synthesize_task():
    """Synthesize an .mp3 for every BANKING77 row using gTTS."""
    logger.info("Starting audio synthesis task")
    try:
        from ingestion.synthesize_audio import synthesize
        synthesize()
        logger.info("Audio synthesis completed successfully")
    except Exception as e:
        logger.error(f"Synthesis task failed: {str(e)}", exc_info=True)
        raise


@task(retries=2, retry_delay_seconds=10)
def store_metadata_task():
    """Insert one row per audio file into the calls table; idempotent."""
    logger.info("Starting metadata storage task")
    try:
        from ingestion.store_metadata import store
        store()
        logger.info("Metadata storage completed successfully")
    except Exception as e:
        logger.error(f"Metadata storage task failed: {str(e)}", exc_info=True)
        raise


@task(retries=2, retry_delay_seconds=60)
def transcribe_task():
    """Transcribe every pending call's audio file with Whisper."""
    logger.info("Starting transcription task")
    try:
        from processing.transcribe import transcribe_all
        transcribe_all()
        logger.info("Transcription completed successfully")
    except Exception as e:
        logger.error(f"Transcription task failed: {str(e)}", exc_info=True)
        raise


@task(retries=1, retry_delay_seconds=10)
def clean_task():
    """Normalize raw Whisper transcripts in place; idempotent."""
    logger.info("Starting transcript cleaning task")
    try:
        from processing.clean import clean_all_transcripts
        clean_all_transcripts()
        logger.info("Transcript cleaning completed successfully")
    except Exception as e:
        logger.error(f"Cleaning task failed: {str(e)}", exc_info=True)
        raise


@task(retries=1)
def validate_task():
    """Run the Great Expectations suite; raises on failure to block training."""
    logger.info("Starting validation task")
    try:
        from processing.validate import run_validation
        run_validation()
        logger.info("Validation completed successfully")
    except Exception as e:
        logger.error(f"Validation task failed: {str(e)}", exc_info=True)
        raise


@task(retries=1)
def prepare_data_task():
    """Fit the TF-IDF vectorizer and produce train/val/test splits."""
    logger.info("Starting data preparation task")
    try:
        from ml.prepare_data import prepare_features
        prepare_features()
        logger.info("Data preparation completed successfully")
    except Exception as e:
        logger.error(f"Data preparation task failed: {str(e)}", exc_info=True)
        raise


@task(retries=1)
def train_task():
    """Train the intent classifier and save a versioned model file."""
    logger.info("Starting model training task")
    try:
        from ml.train import train_model
        train_model()
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Training task failed: {str(e)}", exc_info=True)
        raise


@task
def evaluate_task():
    """Score the trained model on the test set and write metrics to disk."""
    logger.info("Starting model evaluation task")
    try:
        from ml.evaluate import evaluate_model
        evaluate_model()
        logger.info("Model evaluation completed successfully")
    except Exception as e:
        logger.error(f"Evaluation task failed: {str(e)}", exc_info=True)
        raise


@flow(name="VoiceIntent Pipeline", log_prints=True)
def voiceintent_pipeline():
    """Run every VoiceIntent stage end-to-end in dependency order."""
    logger.info("=" * 60)
    logger.info("VoiceIntent Pipeline Started")
    logger.info("=" * 60)

    ingest_task()
    synthesize_task()
    store_metadata_task()
    transcribe_task()
    clean_task()
    validate_task()
    prepare_data_task()
    train_task()
    evaluate_task()

    logger.info("=" * 60)
    logger.info("VoiceIntent Pipeline Completed Successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VoiceIntent Prefect orchestration")
    parser.add_argument(
        "--mode",
        choices=["run", "serve"],
        default="run",
        help="'run' executes the pipeline once. 'serve' starts a long-running worker that triggers the flow on a schedule.",
    )
    parser.add_argument(
        "--cron",
        default="0 2 * * *",
        help="Cron expression for scheduled runs in --mode serve. Default: daily at 02:00.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Optional interval in seconds for scheduled runs (overrides --cron when provided).",
    )
    args = parser.parse_args()

    if args.mode == "serve":
        if args.interval:
            logger.info(f"Serving VoiceIntent pipeline on a {args.interval}-second interval schedule")
            voiceintent_pipeline.serve(
                name="voiceintent-scheduled",
                tags=["voiceintent", "scheduled"],
                interval=args.interval,
            )
        else:
            logger.info(f"Serving VoiceIntent pipeline on cron schedule '{args.cron}'")
            voiceintent_pipeline.serve(
                name="voiceintent-scheduled",
                tags=["voiceintent", "scheduled"],
                cron=args.cron,
            )
    else:
        voiceintent_pipeline()
