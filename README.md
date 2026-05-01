# VoiceIntent

### An Intelligent Voice Assistant for Customer Support

---

## Overview

VoiceIntent is an end-to-end voice-to-intent pipeline. It ingests BANKING77 customer queries, synthesises speech via gTTS, transcribes audio back with OpenAI Whisper, validates transcripts with Great Expectations, trains an intent classifier on the cleaned data, and serves live predictions through a FastAPI backend with a Streamlit dashboard. The whole pipeline is orchestrated with Prefect and packaged with Docker.

**Dataset:** BANKING77 — 13,069 real customer queries across 77 fine-grained banking intents.

**Headline numbers (current model):**

| Metric | Value |
| ------ | ----- |
| Test accuracy | 0.8791 |
| Macro F1 | 0.8792 |
| Drift score (test vs train distribution) | 0.0907 |
| Threshold-policy precision when not escalated | 95.9% |
| Threshold-policy escalation rate | 23.3% |

---

## Architecture

```mermaid
flowchart TB
    subgraph INGEST["📥 Ingestion"]
        HF["HuggingFace<br/>BANKING77"] --> DL["download_dataset.py"]
        DL --> CSV["Raw CSVs"]
        CSV --> SYN["synthesize_audio.py<br/>(gTTS)"]
        SYN --> MP3["13k .mp3 files"]
        MP3 --> META["store_metadata.py"]
    end

    subgraph PROCESS["🎙️ Processing"]
        TR["transcribe.py<br/>(Whisper)"]
        CL["clean.py"]
        VAL["validate.py<br/>(Great Expectations)"]
        TR --> CL --> VAL
    end

    subgraph DB[("🗄️ PostgreSQL")]
        CALLS[(calls)]
        TRANS[(transcripts)]
        PRED[(predictions)]
        MR[(model_runs)]
    end

    subgraph ML["🤖 ML Training"]
        PD["prepare_data.py<br/>(TF-IDF)"]
        TRAIN["train.py<br/>(LogisticRegression)"]
        EV["evaluate.py<br/>(metrics + drift)"]
        SWEEP["threshold_sweep.py"]
        PD --> TRAIN --> EV
        TRAIN --> SWEEP
    end

    subgraph SERVE["🛎️ Serving"]
        USER([👤 User audio])
        API["FastAPI<br/>/predict"]
        WHISP2["Whisper<br/>transcribe"]
        CLF["Classifier"]
        GATE{"Confidence<br/>≥ threshold?"}
        TTS["gTTS<br/>(cached on disk)"]
        ESC["Escalation<br/>handoff"]
        DASH["Streamlit<br/>Dashboard"]

        USER --> API --> WHISP2 --> CLF --> GATE
        GATE -- yes --> TTS
        GATE -- no --> ESC
        TTS --> DASH
        ESC --> DASH
    end

    META --> CALLS
    CALLS --> TR
    TR --> TRANS
    CL --> TRANS
    TRANS --> PD
    EV --> MR
    CLF --> PRED
    MR --> API
    PRED --> DASH

    PREFECT["⚙️ Prefect<br/>orchestration/pipeline.py"]:::orchestration
    PREFECT -.-> INGEST
    PREFECT -.-> PROCESS
    PREFECT -.-> ML

    classDef orchestration fill:#ffe4b5,stroke:#cc8800,color:#333
```

The Prefect flow runs every stage in dependency order; each task is idempotent so partial reruns are cheap. At inference time, the FastAPI service holds the trained model + Whisper in memory and answers `/predict` with both the predicted intent and a synthesized voice reply (cached on disk after the first synthesis per intent).

---

## Prerequisites

- Python 3.11+
- PostgreSQL 16
- ffmpeg (Whisper requires it for audio decoding)
- Docker Desktop (only for the containerised path)

```bash
# macOS
brew install ffmpeg postgresql@16
```

---

## Quick start (local)

```bash
# 1. Clone, enter, create venv
git clone <repo-url> && cd voiceintent
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD"

# 2. Copy env template and fill in DB password + HF token
cp .env.example .env
# DATABASE_URL=postgresql://postgres:PASSWORD@localhost:5432/voiceintent
# HF_TOKEN=...

# 3. Restore the database from the shared dump (fastest path)
psql -U postgres -h localhost -p 5432 -d postgres -c "DROP DATABASE IF EXISTS voiceintent;"
psql -U postgres -h localhost -p 5432 -d postgres -c "CREATE DATABASE voiceintent;"
psql -U postgres -h localhost -p 5432 -d voiceintent < voiceintent_dump.sql

# 4. Run the orchestrated pipeline (skips training if a model is already present)
python orchestration/pipeline.py

# 5. Start API + dashboard in two terminals
python serving/api.py                          # http://localhost:8000
streamlit run serving/dashboard.py             # http://localhost:8501
```

> The dump's `audio_file_path` values are absolute paths from the original machine. If the pipeline complains about pending transcriptions after restore, run:
> ```sql
> UPDATE calls SET audio_file_path =
>   '<your-project-root>/data/audio/' ||
>   REPLACE(SUBSTRING(audio_file_path FROM POSITION('data\audio\' IN audio_file_path) + 11), '\', '/');
> ```

---

## Running from scratch (without the dump)

If you want to regenerate everything yourself instead of using the shared dump:

```bash
psql -U postgres -h localhost -p 5432 -c "CREATE DATABASE voiceintent;"
python -c "from storage.db import init_db; init_db()"

python ingestion/download_dataset.py
python ingestion/synthesize_audio.py    # ~2 hr (rate-limited by gTTS)
python ingestion/store_metadata.py
python processing/transcribe.py         # 4-8 hr CPU, ~30 min GPU
python processing/clean.py
python processing/validate.py
python ml/prepare_data.py
python ml/train.py
python ml/evaluate.py
python ml/threshold_sweep.py
```

Or run all of them through the Prefect flow:

```bash
python orchestration/pipeline.py
```

---

## API endpoints

Base URL: `http://localhost:8000` · Swagger UI at `/docs`.

| Method | Path | Description |
| ------ | ---- | ----------- |
| `POST` | `/predict` | Upload audio (mp3/wav/webm). Returns predicted intent, confidence, transcript, canned response text, and an audio URL. Persists the prediction to the DB. |
| `GET`  | `/intent_response/{intent}` | Streams the gTTS-rendered MP3 reply for a given intent. Cached in memory and on disk. |
| `GET`  | `/escalation` | Streams the gTTS-rendered "redirecting to an agent" message used when confidence is below threshold. |
| `GET`  | `/metrics` | Prediction-distribution stats and the current model's training metrics. |
| `GET`  | `/drift?window=N` | Live drift score: JS divergence between the most recent N predictions and the prior N. Returns `ready: false` if there are fewer than `2N` predictions. |
| `GET`  | `/pipeline/status` | Most recent training-pipeline run summary. |
| `GET`  | `/health` | API, DB, Whisper, classifier readiness. |

### `/predict` response shape

```json
{
  "intent": "card_blocked",
  "confidence": 0.5852,
  "confidence_threshold": 0.4,
  "escalated": false,
  "transcript": "i need my card now",
  "raw_transcript": " I need my card now.",
  "model_version": "model_v20260501_191559",
  "response_text": "It appears that you are asking about card blocked. ...",
  "response_audio_url": "/intent_response/card_blocked",
  "top_5_intents": [ {"intent": "...", "confidence": ...}, ... ]
}
```

When `confidence < confidence_threshold`, `escalated` is `true`, `response_text` is replaced with the agent-handoff message, and `response_audio_url` becomes `/escalation`.

---

## Dashboard

Streamlit UI on `http://localhost:8501` with three tabs:

- **🎯 Predict** — Record from microphone *or* upload a file. Shows the predicted intent, confidence vs. threshold, the transcript, a synthesized voice reply, and a top-5 candidates chart with the predicted intent visually highlighted and the threshold drawn as a dashed line.
- **📊 Analytics** — Total predictions, average confidence, model accuracy, drift status, a 📞 Recent Calls table (last 10 with transcript + routing badge), a 🌀 Live Distribution Drift card with top movers, and the intent distribution chart.
- **⚙️ Pipeline Status** — Latest Prefect run, model training history, accuracy trend, and a system health panel.

### Try the dashboard without recording

Five sample MP3s ship with the repo under `data/audio/samples/` so you can hit the **Predict** tab and upload one without setting up a microphone:

| File | Expected outcome |
| ---- | ---------------- |
| `card_blocked.mp3` | Confident `card_blocked` reply (≈ 0.59) |
| `cancel_transfer.mp3` | Confident `cancel_transfer` reply |
| `lost_or_stolen_card.mp3` | Confident `lost_or_stolen_card` reply |
| `card_swallowed.mp3` | Confident `card_swallowed` reply |
| `automatic_top_up.mp3` | Low confidence (≈ 0.26) — **demonstrates the agent-handoff escalation path** |

---

## Configuration knobs

All in `config/settings.py`:

| Setting | Default | Effect |
| ------- | ------- | ------ |
| `CONFIDENCE_THRESHOLD` | `0.4` | Predictions below this confidence are routed to the escalation path. Backed by the threshold sweep in `docs/threshold_sweep.png`. |
| `LOW_CONFIDENCE_RESPONSE` | text | Sentence spoken to the caller when escalating. |
| `INTENT_NAMES` | 77 entries | Folder-index → intent-name mapping for `store_metadata.py`. Source of truth for intent labels. |

Environment variables (`.env`):

```env
HF_TOKEN=hf_...
DATABASE_URL=postgresql://postgres:PASSWORD@localhost:5432/voiceintent
POSTGRES_USER=postgres
POSTGRES_PASSWORD=PASSWORD
POSTGRES_DB=voiceintent
FORCE_RETRAIN=1   # optional: forces train.py to retrain even when artifacts exist
```

---

## Database schema

| Table | Purpose | Key columns |
| ----- | ------- | ----------- |
| `calls` | One row per audio file | `audio_file_path` (unique), `intent_label`, `split` |
| `transcripts` | Whisper output per call | `call_id` (FK, unique), `raw_transcript`, `cleaned_transcript` |
| `predictions` | API prediction log | `predicted_intent`, `confidence_score`, `cleaned_transcript`, `model_version` |
| `model_runs` | Training-run metrics | `model_version`, `accuracy`, `macro_f1`, `drift_score`, `data_hash`, `training_samples` |

The full ORM definition is in `storage/models.py`.

---

## Folder structure

```
voiceintent/
├── config/
│   ├── settings.py            # paths, DB URL, INTENT_NAMES, CONFIDENCE_THRESHOLD
│   └── intent_responses.json  # canned reply sentence per intent
├── ingestion/                 # download · synthesize · store metadata
├── processing/                # transcribe · clean · validate (GX)
├── storage/                   # SQLAlchemy ORM + session factory
├── ml/                        # prepare_data · train · evaluate · threshold_sweep
│   └── saved_models/          # versioned classifier + vectorizer + label encoder + cached MP3s
├── orchestration/             # Prefect flow
├── serving/                   # FastAPI + Streamlit
├── logging_monitoring/        # JSON-structured logger
├── data/                      # audio + raw CSVs (not committed)
├── gx/                        # Great Expectations project root
├── docs/                      # generated metrics, plots, sweep results
├── voiceintent_dump.sql       # shared full-DB snapshot
├── docker-compose.yml         # multi-container deployment
├── Dockerfile
└── requirements.txt
```

---

## Docker

```bash
docker-compose up --build
```

Stands up four services: `postgres`, `pipeline`, `api` (port 8000), `dashboard` (port 8501). First build pulls torch + Whisper, takes ~50 min on a fresh machine; subsequent builds are cached.

---

## Team

| Member | Scope |
| ------ | ----- |
| Hasan | Ingestion + storage schema |
| Rohan | Transcription + data quality |
| Lina | ML / intent classification |
| Ibrahim Noor | Orchestration, deployment, serving |

---

## AI Usage Declaration

- **Tool:** Claude (Anthropic)
- **Used for:** Debugging environment issues.
- **Extent:** Potential buggy code was given to claude to debug and find issues.

---
