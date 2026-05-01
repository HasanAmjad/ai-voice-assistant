"""Authored by: Ibrahim Noor."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import whisper
import joblib
import numpy as np
from scipy.spatial.distance import jensenshannon
from pathlib import Path
import sys
import time
import json
import io
import threading
from datetime import datetime
import tempfile
import os
from gtts import gTTS

sys.path.append(str(Path(__file__).parent.parent))

from storage.db import get_session, apply_pending_migrations
from storage.models import Prediction, ModelRun, Call
from processing.clean import clean_text
from logging_monitoring.logger import get_logger
from config.settings import CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_RESPONSE
from sqlalchemy import func, desc

logger = get_logger(__name__)

app = FastAPI(
    title="VoiceIntent API",
    description="Automated Voice-to-Intent Intelligence Pipeline",
    version="1.0.0",
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

INTENT_RESPONSES = {}
RESPONSE_AUDIO_CACHE = {}
RESPONSE_AUDIO_LOCK = threading.Lock()

RESPONSE_AUDIO_DIR = Path("ml/saved_models/response_audio")
ESCALATION_CACHE_KEY = "__escalation__"


def _load_intent_responses():
    """Read the canned per-intent reply sentences from config/intent_responses.json."""
    path = Path("config/intent_responses.json")
    if not path.exists():
        logger.warning("config/intent_responses.json not found; spoken responses will be empty.")
        return {}
    with open(path) as f:
        return json.load(f)


def _audio_cache_path(key: str) -> Path:
    """File path for the on-disk MP3 cache entry corresponding to `key`."""
    return RESPONSE_AUDIO_DIR / f"{key}.mp3"


def _hydrate_audio_cache_from_disk():
    """Pre-populate the in-memory audio cache from any MP3 files saved by previous runs."""
    RESPONSE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    loaded = 0
    for mp3 in RESPONSE_AUDIO_DIR.glob("*.mp3"):
        try:
            RESPONSE_AUDIO_CACHE[mp3.stem] = mp3.read_bytes()
            loaded += 1
        except Exception as e:
            logger.warning(f"Could not load cached response audio {mp3.name}: {e}")
    if loaded:
        logger.info(f"Hydrated response audio cache from disk: {loaded} entries")


def _format_response(intent: str) -> str:
    """Build the spoken sentence: 'It appears that you are asking about <intent>. <canned reply>'."""
    canned = INTENT_RESPONSES.get(intent, "")
    if not canned:
        return ""
    spoken_intent = intent.replace("_", " ")
    return f"It appears that you are asking about {spoken_intent}. {canned}"


def _store_audio(key: str, audio_bytes: bytes):
    """Write an MP3 to both the in-memory and on-disk cache."""
    RESPONSE_AUDIO_CACHE[key] = audio_bytes
    try:
        RESPONSE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        _audio_cache_path(key).write_bytes(audio_bytes)
    except Exception as e:
        logger.warning(f"Could not persist response audio for {key!r}: {e}")


def _synthesize_response_audio(intent: str) -> bytes:
    """Return the gTTS-rendered MP3 for `intent`'s prefixed reply, cached in memory and on disk."""
    cached = RESPONSE_AUDIO_CACHE.get(intent)
    if cached is not None:
        return cached
    text = _format_response(intent)
    if not text:
        raise HTTPException(status_code=404, detail=f"No canned response defined for intent '{intent}'")
    with RESPONSE_AUDIO_LOCK:
        cached = RESPONSE_AUDIO_CACHE.get(intent)
        if cached is not None:
            return cached
        buf = io.BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buf)
        audio_bytes = buf.getvalue()
        _store_audio(intent, audio_bytes)
        return audio_bytes


def _synthesize_escalation_audio() -> bytes:
    """Return the gTTS-rendered MP3 for the low-confidence handoff message, cached in memory and on disk."""
    cached = RESPONSE_AUDIO_CACHE.get(ESCALATION_CACHE_KEY)
    if cached is not None:
        return cached
    with RESPONSE_AUDIO_LOCK:
        cached = RESPONSE_AUDIO_CACHE.get(ESCALATION_CACHE_KEY)
        if cached is not None:
            return cached
        buf = io.BytesIO()
        gTTS(text=LOW_CONFIDENCE_RESPONSE, lang="en").write_to_fp(buf)
        audio_bytes = buf.getvalue()
        _store_audio(ESCALATION_CACHE_KEY, audio_bytes)
        return audio_bytes


def _prewarm_response_audio_cache():
    """Synthesize every canned response in the background so live requests hit a warm cache."""
    try:
        _synthesize_escalation_audio()
        intents = list(INTENT_RESPONSES.keys())
        logger.info(f"Pre-warming response audio cache for {len(intents)} intents...")
        for i, intent in enumerate(intents, 1):
            if intent in RESPONSE_AUDIO_CACHE:
                continue
            try:
                _synthesize_response_audio(intent)
            except Exception as e:
                logger.warning(f"Pre-warm failed for '{intent}': {e}")
            time.sleep(0.5)
            if i % 20 == 0:
                logger.info(f"Pre-warm progress: {i}/{len(intents)} cached")
        logger.info("Response audio cache fully warmed.")
    except Exception as e:
        logger.error(f"Pre-warm worker crashed: {e}", exc_info=True)


def load_models():
    """Load the Whisper transcriber, the trained intent classifier, and the canned response sentences."""
    global WHISPER_MODEL, CLASSIFIER_MODEL, VECTORIZER, LABEL_ENCODER, CURRENT_MODEL_VERSION
    global INTENT_RESPONSES

    INTENT_RESPONSES = _load_intent_responses()
    logger.info(f"Loaded {len(INTENT_RESPONSES)} canned intent responses")
    _hydrate_audio_cache_from_disk()

    try:
        logger.info("Loading Whisper model...")
        WHISPER_MODEL = whisper.load_model("base")

        latest_model_path = Path("ml/saved_models/latest_model.txt")
        if latest_model_path.exists():
            with open(latest_model_path, "r") as f:
                CURRENT_MODEL_VERSION = f.read().strip()

            model_path = Path(f"ml/saved_models/{CURRENT_MODEL_VERSION}.pkl")
            CLASSIFIER_MODEL = joblib.load(model_path)

            VECTORIZER = joblib.load("ml/saved_models/tfidf_vectorizer.pkl")
            LABEL_ENCODER = joblib.load("ml/saved_models/label_encoder.pkl")

            logger.info(f"Loaded model version: {CURRENT_MODEL_VERSION}")
        else:
            logger.warning("No trained model found. /predict will not work until model is trained.")

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}", exc_info=True)


@app.on_event("startup")
async def startup_event():
    """Load all models when the FastAPI process boots and start warming the response audio cache."""
    logger.info("FastAPI application starting up...")
    apply_pending_migrations()
    load_models()
    threading.Thread(target=_prewarm_response_audio_cache, daemon=True).start()
    logger.info("FastAPI application ready")


@app.middleware("http")
async def log_requests(request, call_next):
    """Log every HTTP request with method, path, status, and latency."""
    start_time = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start_time) * 1000

    logger.info(
        "HTTP Request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
        },
    )

    return response


@app.get("/")
def read_root():
    """Return basic API metadata and the list of available endpoints."""
    return {
        "name": "VoiceIntent API",
        "version": "1.0.0",
        "description": "Automated Voice-to-Intent Intelligence Pipeline",
        "endpoints": {
            "POST /predict": "Upload an audio file; returns predicted intent, confidence, transcript, response.",
            "GET /intent_response/{intent}": "Stream the gTTS-rendered MP3 reply for a given intent.",
            "GET /escalation": "Stream the gTTS-rendered low-confidence handoff message.",
            "GET /metrics": "Prediction-distribution stats and current model summary.",
            "GET /drift": "Live drift score over rolling windows of recent predictions.",
            "GET /health": "API, database, and model-loading status.",
            "GET /pipeline/status": "Most recent training-pipeline run.",
        },
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file and return the predicted banking intent."""
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
            try:
                result = WHISPER_MODEL.transcribe(tmp_path)
            except RuntimeError as audio_err:
                logger.warning(f"Whisper could not decode audio: {audio_err}")
                raise HTTPException(
                    status_code=400,
                    detail="Could not decode audio. Make sure the file is a valid mp3, wav, or webm recording with actual audio content.",
                )
            raw_transcript = result["text"]

            cleaned_transcript = clean_text(raw_transcript)

            if cleaned_transcript == "[INAUDIBLE]":
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
                    cleaned_transcript=cleaned_transcript,
                    model_version=CURRENT_MODEL_VERSION,
                )
                session.add(pred)
                session.commit()

            logger.info(f"Prediction successful: {predicted_intent} ({confidence_score:.3f})")

            escalated = confidence_score < CONFIDENCE_THRESHOLD
            if escalated:
                response_text = LOW_CONFIDENCE_RESPONSE
                response_audio_url = "/escalation"
            else:
                response_text = _format_response(predicted_intent)
                response_audio_url = f"/intent_response/{predicted_intent}"

            return {
                "intent": predicted_intent,
                "confidence": round(confidence_score, 4),
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "escalated": escalated,
                "transcript": cleaned_transcript,
                "raw_transcript": raw_transcript,
                "model_version": CURRENT_MODEL_VERSION,
                "response_text": response_text,
                "response_audio_url": response_audio_url,
                "top_5_intents": [
                    {
                        "intent": LABEL_ENCODER.inverse_transform([i])[0],
                        "confidence": round(float(probabilities[i]), 4),
                    }
                    for i in np.argsort(probabilities)[-5:][::-1]
                ],
            }

        finally:
            os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/intent_response/{intent}")
def get_intent_response(intent: str):
    """Return the gTTS-synthesized MP3 of the canned reply for `intent`."""
    if intent not in INTENT_RESPONSES:
        raise HTTPException(status_code=404, detail=f"Unknown intent '{intent}'")
    audio_bytes = _synthesize_response_audio(intent)
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Content-Disposition": f'inline; filename="{intent}.mp3"',
        },
    )


@app.get("/escalation")
def get_escalation_audio():
    """Return the gTTS-synthesized MP3 of the low-confidence agent-handoff message."""
    audio_bytes = _synthesize_escalation_audio()
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Content-Disposition": 'inline; filename="escalation.mp3"',
        },
    )


@app.get("/metrics")
def get_metrics():
    """Return aggregate prediction stats and the current model's training metrics."""
    try:
        with get_session() as session:
            intent_dist = session.query(
                Prediction.predicted_intent,
                func.count(Prediction.id).label("count"),
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
                    "trained_at": latest_run.trained_at.isoformat() if latest_run else None,
                },
                "drift_alert": latest_run.drift_score > 0.15 if latest_run and latest_run.drift_score else False,
            }

    except Exception as e:
        logger.error(f"Metrics error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")


@app.get("/health")
def health_check():
    """Report API, database, and model loading status."""
    health_status = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "db_connected": False,
        "whisper_loaded": WHISPER_MODEL is not None,
        "classifier_loaded": CLASSIFIER_MODEL is not None,
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
    """Return the most recent training run as a proxy for pipeline status."""
    try:
        with get_session() as session:
            latest_run = session.query(ModelRun).order_by(desc(ModelRun.trained_at)).first()

            if not latest_run:
                return {
                    "status": "NOT_RUN",
                    "message": "No pipeline runs found",
                }

            return {
                "status": "COMPLETED",
                "last_run": latest_run.trained_at.isoformat(),
                "model_version": latest_run.model_version,
                "accuracy": round(float(latest_run.accuracy), 4) if latest_run.accuracy else None,
                "f1_score": round(float(latest_run.macro_f1), 4) if latest_run.macro_f1 else None,
                "training_samples": latest_run.training_samples,
            }

    except Exception as e:
        logger.error(f"Pipeline status error: {str(e)}", exc_info=True)
        return {
            "status": "ERROR",
            "message": str(e),
        }


@app.get("/drift")
def get_live_drift(window: int = 100):
    """Compare recent prediction distribution against the prior window via JS divergence.

    Returns NaN-style placeholders when there are not yet 2*window predictions in total.
    """
    if window < 10:
        raise HTTPException(status_code=400, detail="window must be >= 10")
    try:
        with get_session() as session:
            recent_q = (
                session.query(Prediction.predicted_intent)
                .order_by(desc(Prediction.predicted_at))
                .limit(window * 2)
                .all()
            )
        intents = [r[0] for r in recent_q]
        total = len(intents)
        if total < window * 2:
            return {
                "ready": False,
                "message": f"Need at least {window * 2} predictions to compute live drift; have {total}.",
                "window": window,
                "total_predictions": total,
            }

        recent = intents[:window]
        prior = intents[window:window * 2]
        all_intents = sorted(set(recent) | set(prior))
        recent_counts = np.array([recent.count(i) / window for i in all_intents])
        prior_counts = np.array([prior.count(i) / window for i in all_intents])
        drift = float(jensenshannon(recent_counts, prior_counts))
        alert = drift > 0.15

        top_movers = sorted(
            [
                {
                    "intent": i,
                    "delta": round(float(r - p), 4),
                    "recent_share": round(float(r), 4),
                    "prior_share": round(float(p), 4),
                }
                for i, r, p in zip(all_intents, recent_counts, prior_counts)
            ],
            key=lambda d: abs(d["delta"]),
            reverse=True,
        )[:5]

        return {
            "ready": True,
            "window": window,
            "drift_score": round(drift, 4),
            "drift_alert": alert,
            "threshold": 0.15,
            "recent_window_size": window,
            "prior_window_size": window,
            "top_movers": top_movers,
        }
    except Exception as e:
        logger.error(f"Drift error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to compute drift: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
