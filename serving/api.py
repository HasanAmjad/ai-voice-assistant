"""
VoiceIntent FastAPI Backend
Member 4 - Ibrahim Noor

Serves predictions and metrics via REST API.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import joblib
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime
import tempfile
import os

sys.path.append(str(Path(__file__).parent.parent))

from storage.db import get_session
from storage.models import Prediction, ModelRun, Call
from processing.clean import clean_text
from logging_monitoring.logger import get_logger
from sqlalchemy import func, desc

logger = get_logger(__name__)

app = FastAPI(
    title="VoiceIntent API",
    description="Automated Voice-to-Intent Intelligence Pipeline",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WHISPER_MODEL = None
CLASSIFIER_MODEL = None
VECTORIZER = None
LABEL_ENCODER = None
CURRENT_MODEL_VERSION = None


def load_models():
    """Load Whisper and ML models at startup"""
    global WHISPER_MODEL, CLASSIFIER_MODEL, VECTORIZER, LABEL_ENCODER, CURRENT_MODEL_VERSION
    
    try:
        logger.info("Loading Whisper model...")
        WHISPER_MODEL = whisper.load_model("base")
        
        latest_model_path = Path("ml/saved_models/latest_model.txt")
        if latest_model_path.exists():
            with open(latest_model_path, 'r') as f:
                CURRENT_MODEL_VERSION = f.read().strip()
            
            model_path = Path(f"ml/saved_models/{CURRENT_MODEL_VERSION}.pkl")
            CLASSIFIER_MODEL = joblib.load(model_path)
            
            VECTORIZER = joblib.load("ml/saved_models/vectorizer.pkl")
            LABEL_ENCODER = joblib.load("ml/saved_models/label_encoder.pkl")
            
            logger.info(f"Loaded model version: {CURRENT_MODEL_VERSION}")
        else:
            logger.warning("No trained model found. /predict will not work until model is trained.")
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}", exc_info=True)


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("FastAPI application starting up...")
    load_models()
    logger.info("FastAPI application ready")


@app.middleware("http")
async def log_requests(request, call_next):
    """Log all HTTP requests"""
    start_time = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start_time) * 1000
    
    logger.info(
        "HTTP Request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2)
        }
    )
    
    return response


@app.get("/")
def read_root():
    """API information"""
    return {
        "name": "VoiceIntent API",
        "version": "1.0.0",
        "description": "Automated Voice-to-Intent Intelligence Pipeline",
        "endpoints": {
            "/predict": "POST - Upload audio file for intent prediction",
            "/metrics": "GET - System metrics and statistics",
            "/health": "GET - Health check",
            "/pipeline/status": "GET - Pipeline execution status"
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict intent from uploaded audio file.
    
    Args:
        file: Audio file (.mp3 or .wav)
    
    Returns:
        JSON with intent, confidence, transcript, and model_version
    """
    if not CLASSIFIER_MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")
    
    logger.info(f"Prediction request received: {file.filename}")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            logger.info("Transcribing audio...")
            result = WHISPER_MODEL.transcribe(tmp_path)
            raw_transcript = result['text']
            
            cleaned_transcript = clean_text(raw_transcript)
            
            if cleaned_transcript == '[INAUDIBLE]':
                raise HTTPException(status_code=400, detail="Audio is inaudible or empty")
            
            features = VECTORIZER.transform([cleaned_transcript])
            
            prediction = CLASSIFIER_MODEL.predict(features)[0]
            probabilities = CLASSIFIER_MODEL.predict_proba(features)[0]
            
            predicted_intent = LABEL_ENCODER.inverse_transform([prediction])[0]
            confidence_score = float(np.max(probabilities))
            
            with get_session() as session:
                pred = Prediction(
                    predicted_intent=predicted_intent,
                    confidence_score=confidence_score,
                    model_version=CURRENT_MODEL_VERSION
                )
                session.add(pred)
                session.commit()
            
            logger.info(f"Prediction successful: {predicted_intent} ({confidence_score:.3f})")
            
            return {
                "intent": predicted_intent,
                "confidence": round(confidence_score, 4),
                "transcript": cleaned_transcript,
                "raw_transcript": raw_transcript,
                "model_version": CURRENT_MODEL_VERSION,
                "top_5_intents": [
                    {
                        "intent": LABEL_ENCODER.inverse_transform([i])[0],
                        "confidence": round(float(probabilities[i]), 4)
                    }
                    for i in np.argsort(probabilities)[-5:][::-1]
                ]
            }
        
        finally:
            os.unlink(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics")
def get_metrics():
    """
    Return system metrics and statistics.
    
    Returns:
        - Intent distribution from predictions
        - Average confidence score
        - Pipeline run count
        - Last run status
        - Drift score
    """
    try:
        with get_session() as session:
            intent_dist = session.query(
                Prediction.predicted_intent,
                func.count(Prediction.id).label('count')
            ).group_by(Prediction.predicted_intent).all()
            
            intent_distribution = {intent: count for intent, count in intent_dist}
            
            avg_confidence = session.query(
                func.avg(Prediction.confidence_score)
            ).scalar() or 0.0
            
            total_predictions = session.query(func.count(Prediction.id)).scalar() or 0
            
            latest_run = session.query(ModelRun).order_by(desc(ModelRun.trained_at)).first()
            
            return {
                "intent_distribution": intent_distribution,
                "average_confidence": round(float(avg_confidence), 4),
                "total_predictions": total_predictions,
                "current_model": {
                    "version": latest_run.model_version if latest_run else None,
                    "accuracy": round(float(latest_run.accuracy), 4) if latest_run and latest_run.accuracy else None,
                    "f1_score": round(float(latest_run.macro_f1), 4) if latest_run and latest_run.macro_f1 else None,
                    "drift_score": round(float(latest_run.drift_score), 4) if latest_run and latest_run.drift_score else None,
                    "trained_at": latest_run.trained_at.isoformat() if latest_run else None
                },
                "drift_alert": latest_run.drift_score > 0.15 if latest_run and latest_run.drift_score else False
            }
    
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    
    Returns:
        - API status
        - Database connection status
        - Model loaded status
    """
    health_status = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "db_connected": False,
        "whisper_loaded": WHISPER_MODEL is not None,
        "classifier_loaded": CLASSIFIER_MODEL is not None
    }
    
    try:
        with get_session() as session:
            session.query(Call).limit(1).all()
        health_status["db_connected"] = True
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        health_status["status"] = "degraded"
    
    if not WHISPER_MODEL or not CLASSIFIER_MODEL:
        health_status["status"] = "degraded"
    
    return health_status


@app.get("/pipeline/status")
def pipeline_status():
    """
    Get most recent pipeline execution status.
    
    In production, this would query Prefect API.
    For now, returns latest model run status.
    """
    try:
        with get_session() as session:
            latest_run = session.query(ModelRun).order_by(desc(ModelRun.trained_at)).first()
            
            if not latest_run:
                return {
                    "status": "NOT_RUN",
                    "message": "No pipeline runs found"
                }
            
            return {
                "status": "COMPLETED",
                "last_run": latest_run.trained_at.isoformat(),
                "model_version": latest_run.model_version,
                "accuracy": round(float(latest_run.accuracy), 4) if latest_run.accuracy else None,
                "f1_score": round(float(latest_run.macro_f1), 4) if latest_run.macro_f1 else None,
                "training_samples": latest_run.training_samples
            }
    
    except Exception as e:
        logger.error(f"Pipeline status error: {str(e)}", exc_info=True)
        return {
            "status": "ERROR",
            "message": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
