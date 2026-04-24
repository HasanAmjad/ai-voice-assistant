# VoiceIntent

### An Intelligent Voice Assistant for Customer Support

**AI 620: Fundamentals of Data Engineering · LUMS**

---

## Project Overview

VoiceIntent simulates a production-grade call-centre intelligence system. The pipeline ingests BANKING77 text queries, synthesises speech audio via gTTS, transcribes audio back with OpenAI Whisper, validates transcript quality with Great Expectations, trains an intent classifier, and serves predictions through a FastAPI endpoint — all orchestrated end-to-end with Prefect inside Docker.

**Dataset:** BANKING77 — 13,069 real customer queries across 77 fine-grained banking intents (e.g. `transfer_not_received`, `card_swallowed`, `exchange_rate`).

**Problem Statement:** Despite the rise of text-based chatbots, voice calls remain the dominant channel for customer support — yet most call centres still rely on manual agents to interpret caller intent in real time. This creates bottlenecks, inconsistent service quality, and high operational costs. VoiceIntent addresses this gap by building a production-grade, end-to-end pipeline that automatically converts raw customer audio into classified intents — from speech synthesis and transcription to ML classification and live API serving.

---

## Prerequisites

- Python 3.11+
- PostgreSQL 16
- ffmpeg — **required by Whisper for audio loading**
- Docker Desktop (for final integration only)
- Node.js (for orchestration dependencies)

### Installing ffmpeg (Windows) — Critical

Whisper uses ffmpeg internally to load `.mp3` files. Without it, transcription fails entirely.

1. Download from https://www.gyan.dev/ffmpeg/builds/ → `ffmpeg-release-essentials.zip`
2. Extract to `C:\ffmpeg`
3. Add to PATH permanently via Windows → Start → Search "Environment Variables" → System Variables → Path → New → `C:\ffmpeg\bin`
4. Verify: `ffmpeg -version`

---

## Quick Start (Local Development)

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

**5. Set PYTHONPATH (Windows CMD) — required every new session:**

```cmd
set PYTHONPATH=D:\path\to\voiceintent
```

To set permanently so you don't repeat this every time, add it via Windows Environment Variables (same steps as ffmpeg above).

**6. Add PostgreSQL to PATH (Windows CMD):**

```cmd
set PATH=%PATH%;C:\Program Files\PostgreSQL\16\bin
```

**7. Create the database and initialize tables:**

```cmd
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
│   └── settings.py             # Centralised config: paths, DB URL, INTENT_NAMES mapping
│
├── ingestion/
│   ├── download_dataset.py     # Pull BANKING77 from HuggingFace
│   ├── synthesize_audio.py     # gTTS: text → .mp3 per sample
│   └── store_metadata.py       # Insert audio metadata into PostgreSQL
│
├── processing/
│   ├── transcribe.py           # Whisper: audio → raw transcript
│   ├── clean.py                # Normalise transcripts
│   └── validate.py             # Great Expectations suite + checkpoint
│
├── storage/
│   ├── db.py                   # SQLAlchemy engine + session factory
│   └── models.py               # ORM table definitions — source of truth
│
├── ml/
│   ├── prepare_data.py         # Feature extraction + stratified splits
│   ├── train.py                # Model training + versioned save
│   ├── evaluate.py             # Accuracy, F1, confusion matrix
│   └── saved_models/           # model_v{YYYYMMDD_HHMMSS}.pkl + latest_model.pkl
│
├── orchestration/
│   └── pipeline.py             # Prefect flow + tasks + scheduling
│
├── serving/
│   ├── api.py                  # FastAPI: /predict, /metrics, /health
│   └── dashboard.py            # Streamlit dashboard
│
├── logging_monitoring/
│   └── logger.py               # JSON-structured logger
│
├── data/
│   ├── audio/                  # Generated .mp3 files (not committed to Git)
│   │   ├── train/              # Subfolders by intent label number (0-76)
│   │   └── test/
│   └── raw/                    # Downloaded CSV snapshots (not committed to Git)
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

### Important Notes on Schema

- `calls.intent_label` stores the **string intent name** (e.g. `card_arrival`), not the integer folder number
- The integer → intent name mapping lives in `config/settings.py` as `INTENT_NAMES` list
- `calls.audio_file_path` stores the **absolute path** to the `.mp3` file on disk
- All scripts import `INTENT_NAMES` from `config/settings.py` — never hardcode intent names elsewhere

---

## Running the Pipeline

### 1 — Data Ingestion (Done ✅)

```cmd
python ingestion\download_dataset.py
python ingestion\synthesize_audio.py
python ingestion\store_metadata.py
```

### 2 — Transcription & Cleaning

```cmd
python processing\transcribe.py   # takes 4-8 hrs on CPU, 30-45 min on GPU
python processing\clean.py
python processing\validate.py
```

### 3 — ML Training

```cmd
python ml\prepare_data.py
python ml\train.py
python ml\evaluate.py
```

### 4 — Serving

```cmd
python orchestration\pipeline.py
python serving\api.py
python serving\dashboard.py
```

---

## API Endpoints

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
torch
openai-whisper
```

---

## .gitignore

Make sure your `.gitignore` contains:

```
.env
data/
logs/
venv/
ml/saved_models/
__pycache__/
*.pyc
```

The `data/` folder is excluded because audio files are too large for Git. Each member regenerates them locally by running the ingestion scripts.

---

## Known Issues & Solutions

| Issue                                           | Cause                             | Fix                                                                                |
| ----------------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'config'` | PYTHONPATH not set                | `set PYTHONPATH=D:\path\to\voiceintent`                                            |
| `psql not recognized`                           | PostgreSQL not in PATH            | `set PATH=%PATH%;C:\Program Files\PostgreSQL\16\bin`                               |
| `password authentication failed`                | Wrong password in DATABASE_URL    | Check `.env` — must be `postgresql://postgres:PASSWORD@localhost:5432/voiceintent` |
| `gTTS 429 Too Many Requests`                    | Rate limited by Google TTS API    | Switch network (hotspot), delete zero-byte files, re-run                           |
| `Whisper FileNotFoundError WinError 2`          | ffmpeg not installed              | Install ffmpeg and add `C:\ffmpeg\bin` to PATH                                     |
| `SHA256 checksum does not match`                | Corrupted Whisper model download  | `rmdir /s /q C:\Users\<you>\.cache\whisper` then re-run                            |
| `Dataset scripts no longer supported`           | Legacy HuggingFace dataset format | Use `mteb/banking77` instead of `PolyAI/banking77`                                 |
| `clean.py found 0 transcripts`                  | `transcripts` table empty         | Run `transcribe.py` first, then `clean.py`                                         |

---

## Member Responsibilities

| Member             | Scope                               | Key Files                                                                               |
| ------------------ | ----------------------------------- | --------------------------------------------------------------------------------------- |
| Member 1 — Hasan   | Data Ingestion & Storage Schema     | `ingestion/`, `storage/models.py`, `storage/db.py`, `config/settings.py`                |
| Member 2 — Rohan   | Transcription & Data Quality        | `processing/transcribe.py`, `processing/clean.py`, `processing/validate.py`             |
| Member 3 — Lina    | ML / Intent Classification          | `ml/`                                                                                   |
| Member 4 — Ibrahim | Orchestration, Deployment & Serving | `orchestration/`, `serving/`, `logging_monitoring/`, `Dockerfile`, `docker-compose.yml` |

---

## AI Usage Declaration

- **Tool:** Claude (Anthropic)
- **Used for:** Project architecture planning, step-by-step implementation guidance, database schema design, debugging environment issues (PATH, PYTHONPATH, ffmpeg), and documentation.
- **Extent:** Guidance and planning only. All code was written, tested, and debugged by team members.

---

_VoiceIntent · AI 620 · LUMS_
