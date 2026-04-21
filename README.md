# VoiceIntent

### An Intelligent Voice Assistant for Customer Support

**AI 620: Fundamentals of Data Engineering · LUMS**

---

## Project Overview

VoiceIntent simulates a production-grade call-centre intelligence system. The pipeline ingests BANKING77 text queries, synthesises speech audio via gTTS, transcribes audio back with OpenAI Whisper, validates transcript quality with Great Expectations, trains an intent classifier, and serves predictions through a FastAPI endpoint — all orchestrated end-to-end with Prefect inside Docker.

**Dataset:** BANKING77 — 13,069 real customer queries across 77 fine-grained banking intents (e.g. `transfer_not_received`, `card_swallowed`, `exchange_rate`).

---

## Quick Start (Local Development)

### Prerequisites

- Python 3.11+
- PostgreSQL 16 (local) or Docker Desktop
- Node.js (for orchestration dependencies)

### Setup

**1. Clone the repo and navigate into it:**

```bash
git clone <repo-url>
cd voiceintent
```

**2. Create and activate virtual environment:**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Set up environment variables:**

```bash
cp .env.example .env
# Edit .env and fill in your values
```

**5. Set PYTHONPATH (Windows CMD):**

```bash
set PYTHONPATH=<absolute-path-to-voiceintent-folder>
```

**6. Create the database and initialize tables:**

```bash
psql -U postgres -h localhost -p 5432 -c "CREATE DATABASE voiceintent;"
python -c "from storage.db import init_db; init_db(); print('Tables created.')"
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```env
HF_TOKEN=your_huggingface_token_here
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/voiceintent
POSTGRES_USER=postgres
POSTGRES_PASSWORD=YOUR_PASSWORD
POSTGRES_DB=voiceintent
```

Get your HuggingFace token at: https://huggingface.co/settings/tokens

---

## Folder Structure

```
voiceintent/
├── .env                        # DB credentials, HF_TOKEN — never commit
├── .env.example                # Template for teammates
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── README.md
│
├── config/
│   └── settings.py             # Centralised config: paths, DB URL, INTENT_NAMES
│
├── ingestion/                  # MEMBER 1
│   ├── download_dataset.py     # Pull BANKING77 from HuggingFace
│   ├── synthesize_audio.py     # gTTS: text → .mp3 per sample
│   └── store_metadata.py       # Insert audio metadata into PostgreSQL
│
├── processing/                 # MEMBER 2
│   ├── transcribe.py           # Whisper: audio → raw transcript
│   ├── clean.py                # Normalise transcripts
│   └── validate.py             # Great Expectations suite + checkpoint
│
├── storage/                    # MEMBER 1 (shared schema)
│   ├── db.py                   # SQLAlchemy engine + session factory
│   └── models.py               # ORM table definitions — source of truth
│
├── ml/                         # MEMBER 3
│   ├── prepare_data.py         # Feature extraction + stratified splits
│   ├── train.py                # Model training + versioned save
│   ├── evaluate.py             # Accuracy, F1, confusion matrix
│   └── saved_models/           # model_v{YYYYMMDD_HHMMSS}.pkl + latest_model.pkl
│
├── orchestration/              # MEMBER 4
│   └── pipeline.py             # Prefect flow + tasks + scheduling
│
├── serving/                    # MEMBER 4
│   ├── api.py                  # FastAPI: /predict, /metrics, /health
│   └── dashboard.py            # Streamlit dashboard
│
├── logging_monitoring/         # MEMBER 4
│   └── logger.py               # JSON-structured logger
│
├── data/
│   ├── audio/                  # Generated .mp3 files
│   │   ├── train/              # Subfolders by intent label (numbered)
│   │   └── test/
│   └── raw/                    # Downloaded CSV snapshots
│
├── gx/                         # Great Expectations project root
│   ├── expectations/
│   └── checkpoints/
│
└── docs/
    ├── architecture.png
    └── schema.png
```

---

## Database Schema

Four tables — defined in `storage/models.py`. **Do not modify this file without team consensus.**

| Table         | Purpose                    | Written by                           |
| ------------- | -------------------------- | ------------------------------------ |
| `calls`       | One row per audio file     | Member 1 (`store_metadata.py`)       |
| `transcripts` | Whisper output per call    | Member 2 (`transcribe.py`)           |
| `predictions` | Model predictions per call | Member 4 (`api.py`)                  |
| `model_runs`  | Training metrics per run   | Member 3 (`train.py`, `evaluate.py`) |

---

## Pipeline Steps

| #   | Script                            | What It Does                                    |
| --- | --------------------------------- | ----------------------------------------------- |
| 1   | `ingestion/download_dataset.py`   | HuggingFace API → CSVs saved to `data/raw/`     |
| 2   | `ingestion/synthesize_audio.py`   | CSVs → 13,069 `.mp3` files in `data/audio/`     |
| 3   | `ingestion/store_metadata.py`     | Audio paths + labels → `calls` table            |
| 4   | `processing/transcribe.py`        | Audio → `raw_transcript` in `transcripts` table |
| 5   | `processing/clean.py`             | `raw_transcript` → `cleaned_transcript`         |
| 6   | `processing/validate.py`          | Great Expectations checkpoint on transcripts    |
| 7   | `ml/prepare_data.py`              | Transcripts → TF-IDF matrix                     |
| 8   | `ml/train.py`                     | TF-IDF → versioned `.pkl` model                 |
| 9   | `ml/evaluate.py`                  | Accuracy, F1, confusion matrix, drift score     |
| 10  | `serving/api.py` + `dashboard.py` | FastAPI + Streamlit serving                     |

All scripts are **idempotent** — safe to re-run. Already-processed rows/files are skipped automatically.

---

## Running Ingestion (Member 1)

```bash
# Step 1: Download dataset
python ingestion\download_dataset.py

# Step 2: Synthesise audio (takes 1-2 hours for 13,069 files)
python ingestion\synthesize_audio.py

# Step 3: Store metadata in database
python ingestion\store_metadata.py
```

If synthesis is interrupted, just re-run — it skips existing files automatically.

---

## API Endpoints (Member 4)

| Endpoint           | Method | Description                                                               |
| ------------------ | ------ | ------------------------------------------------------------------------- |
| `/predict`         | POST   | Upload `.mp3` → returns `{intent, confidence, transcript, model_version}` |
| `/metrics`         | GET    | Intent distribution, avg confidence, drift score                          |
| `/health`          | GET    | `{status: ok, db: connected, model: loaded}`                              |
| `/pipeline/status` | GET    | Most recent Prefect flow run status                                       |

Swagger UI available at: `http://localhost:8000/docs`

---

## Requirements

```
datasets>=2.18
huggingface_hub
gTTS==2.5.1
sqlalchemy>=2.0
psycopg2-binary
python-dotenv
alembic
```

---

## Member Responsibilities

| Member             | Scope                               | Key Files                                                                               |
| ------------------ | ----------------------------------- | --------------------------------------------------------------------------------------- |
| Member 1 - Hasan   | Data Ingestion & Storage Schema     | `ingestion/`, `storage/models.py`, `storage/db.py`                                      |
| Member 2 - Rohan   | Transcription & Data Quality        | `processing/`                                                                           |
| Member 3 - Lina    | ML / Intent Classification          | `ml/`                                                                                   |
| Member 4 - Ibrahim | Orchestration, Deployment & Serving | `orchestration/`, `serving/`, `logging_monitoring/`, `Dockerfile`, `docker-compose.yml` |

---
