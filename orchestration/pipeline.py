"""
VoiceIntent Prefect Pipeline
Member 4 - Ibrahim Noor

Orchestrates the complete voice-to-intent pipeline with retries and caching.
"""

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
    cache_expiration=timedelta(hours=24)
)
def ingest_task():
    """
    Task 1: Download BANKING77 dataset from HuggingFace.
    Cached for 24 hours to avoid redundant downloads.
    """
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
    """
    Task 2: Synthesize audio from text using gTTS.
    May take 1-2 hours for full dataset.
    """
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
    """
    Task 3: Store audio file metadata in PostgreSQL.
    Idempotent via ON CONFLICT DO NOTHING.
    """
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
    """
    Task 4: Transcribe audio files using Whisper.
    CPU-heavy, may take several hours.
    """
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
    """
    Task 5: Clean and normalize transcripts.
    Fast, idempotent operation.
    """
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
    """
    Task 6: Validate data quality with Great Expectations.
    Raises exception on validation failure, blocking downstream tasks.
    """
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
    """
    Task 7: Prepare features for ML training.
    Checks data hash for idempotency.
    """
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
    """
    Task 8: Train intent classifier with versioning.
    """
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
    """
    Task 9: Evaluate model and detect drift.
    No retries - if evaluation fails, we want to know immediately.
    """
    logger.info("Starting model evaluation task")
    try:
        from ml.evaluate import evaluate_model
        evaluate_model()
        logger.info("Model evaluation completed successfully")
    except Exception as e:
        logger.error(f"Evaluation task failed: {str(e)}", exc_info=True)
        raise


@flow(name='VoiceIntent Pipeline', log_prints=True)
def voiceintent_pipeline():
    """
    Complete VoiceIntent pipeline orchestration.
    
    Runs all tasks in sequence with proper dependency management.
    Tasks are idempotent and can be safely re-run.
    """
    logger.info("="*60)
    logger.info("VoiceIntent Pipeline Started")
    logger.info("="*60)
    
    # Task 1: Data Ingestion
    ingest_task()
    
    # Task 2: Audio Synthesis
    synthesize_task()
    
    # Task 3: Metadata Storage
    store_metadata_task()
    
    # Task 4: Audio Transcription
    transcribe_task()
    
    # Task 5: Transcript Cleaning
    clean_task()
    
    # Task 6: Data Validation (blocks training if fails)
    validate_task()
    
    # Task 7: Feature Preparation
    prepare_data_task()
    
    # Task 8: Model Training
    train_task()
    
    # Task 9: Model Evaluation
    evaluate_task()
    
    logger.info("="*60)
    logger.info("VoiceIntent Pipeline Completed Successfully")
    logger.info("="*60)


if __name__ == "__main__":
    # For local testing
    voiceintent_pipeline()
