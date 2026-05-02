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

## How to Run

There are **two ways** to run VoiceIntent. Pick the one that matches your environment:

| Path | When to use | First-run cost |
| ---- | ----------- | -------------- |
| **A. Dockerised** | Cleanest, works identically on macOS / Linux / Windows. No need to install Python, Postgres, ffmpeg yourself. | ~9 min build, then `up -d` is seconds |
| **B. Conventional** | Faster iteration during development, lower memory footprint. You install Python, Postgres, and ffmpeg yourself. | ~5 min if Python is already installed |

Both paths land at the same place: API on `http://localhost:8000`, dashboard on `http://localhost:8501`.

---

## Path A — Dockerised Setup

Works the same on macOS, Ubuntu, and Windows. Only prerequisite is **Docker Desktop** (https://www.docker.com/products/docker-desktop/).

```bash
git clone https://github.com/HasanAmjad/ai-voice-assistant.git
cd ai-voice-assistant
cp .env.example .env       # fill in HF_TOKEN if you want to re-run ingestion
docker-compose up --build
```

That's it. On the first run the build takes ~9 min (pulls the CPU-only torch wheel, Whisper, etc.). Subsequent `docker-compose up -d` starts in seconds. The postgres container auto-loads `voiceintent_dump.sql` on first start so you don't need to restore the database manually.

When the API logs `Application startup complete`, open:
- API Swagger: http://localhost:8000/docs
- Dashboard: http://localhost:8501

To stop everything: `docker-compose down`.

---

## Path B — Conventional Setup

Three ordered steps. The **first step (B.1 / B.2 / B.3) is OS-specific** — pick the one that matches your machine. Everything after that (B.4 and B.5) is the same regardless of OS.

```
B.1 / B.2 / B.3   →   B.4: seed the database   →   B.5: start the project
   (your OS)            (fast OR slow)               (same for everyone)
```

---

### B.1 — macOS prerequisites

```bash
# Install system tools
brew install python@3.11 ffmpeg postgresql@14
brew services start postgresql@14

# Clone the repo
git clone https://github.com/HasanAmjad/ai-voice-assistant.git
cd ai-voice-assistant

# Python venv + dependencies
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Open .env and set POSTGRES_PASSWORD to whatever your local postgres uses
# (Brew installs often have no password — leave it blank in that case)
```

Now go to **B.4** below.

---

### B.2 — Ubuntu / Debian prerequisites

```bash
# Install system tools
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip ffmpeg postgresql postgresql-contrib
sudo systemctl start postgresql

# Set the postgres user's password (skip if you already configured one)
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'YOUR_PASSWORD';"

# Clone the repo
git clone https://github.com/HasanAmjad/ai-voice-assistant.git
cd ai-voice-assistant

# Python venv + dependencies
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Open .env and set POSTGRES_PASSWORD and DATABASE_URL to use YOUR_PASSWORD
```

Now go to **B.4** below.

---

### B.3 — Windows (native PowerShell) prerequisites

> If you're comfortable with WSL2, install Ubuntu via WSL and follow the **B.2 Ubuntu** instructions instead — it's smoother. The native PowerShell instructions below work too.

**Install system tools** (pick one method per row):

| Tool | Installer (manual) | Package manager |
| ---- | ----------------- | --------------- |
| Python 3.11 | https://www.python.org/downloads/ | `winget install Python.Python.3.11` |
| PostgreSQL | https://www.postgresql.org/download/windows/ | `winget install PostgreSQL.PostgreSQL` |
| ffmpeg | https://www.gyan.dev/ffmpeg/builds/ → unzip to `C:\ffmpeg`, add `C:\ffmpeg\bin` to PATH | `winget install Gyan.FFmpeg` or `choco install ffmpeg` |
| Git | https://git-scm.com/download/win | `winget install Git.Git` |

Open a **fresh PowerShell window** (so the new PATH takes effect) and verify each tool:

```powershell
python --version          # should show 3.11.x
ffmpeg -version           # should show ffmpeg 6+
psql --version            # should show 16.x
```

Then:

```powershell
# Clone the repo
git clone https://github.com/HasanAmjad/ai-voice-assistant.git
cd ai-voice-assistant

# Python venv + dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Make sure PostgreSQL service is running
Start-Service postgresql-x64-16   # service name may differ; check with: Get-Service postgresql*

# Configure environment
copy .env.example .env
# Open .env and set POSTGRES_PASSWORD and DATABASE_URL to use the password you set during PostgreSQL install
$env:PYTHONPATH = $PWD
```

Now go to **B.4** below.

---

### B.4 — Seed the database (pick ONE of two options)

> | | **🚀 Option 1 — Fast path (recommended)** | **🐢 Option 2 — Slow path** |
> | --- | --- | --- |
> | What it does | Restores `voiceintent_dump.sql` — a snapshot of the fully-trained DB | Re-runs the full pipeline (audio synthesis → Whisper transcription → training) |
> | Time | ~30 seconds | ~6–10 hours |
> | Needs `HF_TOKEN`? | No | Yes |

**Pick one option below — do not run both.** Then continue to B.5.

---

#### B.4 — Option 1 — 🚀 Fast path (restore the dump)

Pick the variant for your OS:

**macOS / Ubuntu:**

```bash
psql -U postgres -h localhost -p 5432 -d postgres -c "DROP DATABASE IF EXISTS voiceintent;"
psql -U postgres -h localhost -p 5432 -d postgres -c "CREATE DATABASE voiceintent;"
psql -U postgres -h localhost -p 5432 -d voiceintent < voiceintent_dump.sql
```

(On Ubuntu, prefix each command with `PGPASSWORD=YOUR_PASSWORD ` if you set a password.)

**Windows (PowerShell):**

```powershell
psql -U postgres -h localhost -p 5432 -d postgres -c "DROP DATABASE IF EXISTS voiceintent;"
psql -U postgres -h localhost -p 5432 -d postgres -c "CREATE DATABASE voiceintent;"
psql -U postgres -h localhost -p 5432 -d voiceintent -f voiceintent_dump.sql
```

Verify it worked — should print `13068`:

```bash
psql -U postgres -h localhost -p 5432 -d voiceintent -c "SELECT COUNT(*) FROM calls;"
```

Now skip to **B.5**.

---

#### B.4 — Option 2 — 🐢 Slow path (regenerate from scratch)

Use this only if you don't have the dump file, or you want to verify the full pipeline end-to-end. Total runtime is ~6–10 hours, most of it Whisper transcribing 13,069 audio clips on CPU.

**Step 2.1 — Get a HuggingFace token** (the pipeline downloads BANKING77 from HuggingFace and needs a free token):

1. Sign up / log in at https://huggingface.co
2. Go to https://huggingface.co/settings/tokens → **New token** → name it anything → role **Read** → **Generate**.
3. Copy the token string (starts with `hf_…`).
4. Open `.env` in any editor and set:
   ```env
   HF_TOKEN=hf_your_token_here
   ```

**Step 2.2 — Create an empty database** (same DROP/CREATE as Option 1, but do **not** load the dump file):

macOS / Ubuntu:
```bash
psql -U postgres -h localhost -p 5432 -d postgres -c "DROP DATABASE IF EXISTS voiceintent;"
psql -U postgres -h localhost -p 5432 -d postgres -c "CREATE DATABASE voiceintent;"
```

Windows (PowerShell): exactly the same two commands as above.

**Step 2.3 — Create the four tables inside the empty database:**

```bash
python -c "from storage.db import init_db; init_db()"
```

(No output means success. If you see an error, `DATABASE_URL` in `.env` is the usual culprit.)

**Step 2.4 — Run the pipeline stages.** Two ways to do this:

*Way A — Stage by stage* (more visibility; recommended for the first run):

```bash
python ingestion/download_dataset.py    # ~30 sec — pulls BANKING77 CSVs from HuggingFace
python ingestion/synthesize_audio.py    # ~2 hr — gTTS makes 13k .mp3 files (rate-limited)
python ingestion/store_metadata.py      # ~1 min — registers each .mp3 in the calls table
python processing/transcribe.py         # 4-8 hr on CPU, ~30 min on GPU — Whisper transcribes everything
python processing/clean.py              # ~1 min — normalises transcripts
python processing/validate.py           # ~30 sec — Great Expectations data-quality checks
python ml/prepare_data.py               # ~10 sec — fits TF-IDF vectorizer
python ml/train.py                      # ~30 sec — trains the classifier
python ml/evaluate.py                   # ~20 sec — writes metrics + confusion matrix
python ml/threshold_sweep.py            # ~10 sec — writes threshold-sweep artifacts
```

*Way B — All at once via Prefect* (less verbose; retries on failure; logs to `logs/pipeline.log`):

```bash
python orchestration/pipeline.py
```

**Step 2.5 — Verify the database is populated:**

```bash
psql -U postgres -h localhost -p 5432 -d voiceintent -c "
  SELECT 'calls' AS t, COUNT(*) FROM calls
  UNION ALL SELECT 'transcripts', COUNT(*) FROM transcripts
  UNION ALL SELECT 'model_runs',  COUNT(*) FROM model_runs;"
```

Expected: `calls` ≈ **13,068**, `transcripts` ≈ **13,068**, `model_runs` ≥ **1**. If `transcripts` is much lower, `transcribe.py` was interrupted — re-run it (it's idempotent and picks up where it left off).

Now continue to **B.5**.

---

### B.5 — Start the API and dashboard (all OSes)

You're here whether you took the fast or the slow path — your database is now populated either way.

#### Option 1 — One-shot helper script (macOS / Linux only)

```bash
./run.sh
```

This stops any leftover api/dashboard, starts postgres if needed, boots FastAPI on `:8000` and Streamlit on `:8501`, waits for both to be ready, prints the URLs, and tails logs. Press `Ctrl+C` to stop both cleanly.

#### Option 2 — Manual (works on all OSes including Windows)

You'll need three terminals (or background each command).

```bash
# Terminal 1 — Prefect pipeline (skips training if a model already exists)
python orchestration/pipeline.py
```

```bash
# Terminal 2 — FastAPI
python serving/api.py
# → http://localhost:8000/docs
```

```bash
# Terminal 3 — Streamlit dashboard
streamlit run serving/dashboard.py
# → http://localhost:8501
```

> **Fast-path users only:** if `pipeline.py` reports thousands of "pending" transcriptions, the dump's `audio_file_path` values are absolute paths from the original machine. Translate them to your local layout:
> ```sql
> UPDATE calls SET audio_file_path =
>   '<your-project-root>/data/audio/' ||
>   REPLACE(SUBSTRING(audio_file_path FROM POSITION('data\audio\' IN audio_file_path) + 11), '\', '/');
> ```

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
├── data/                      # audio + raw CSVs (samples committed under data/audio/samples/)
├── gx/                        # Great Expectations project root
├── docs/                      # architecture/schema diagrams, metrics, sweep results, report
├── voiceintent_dump.sql       # shared full-DB snapshot
├── run.sh                     # one-shot dev launcher (macOS / Linux)
├── docker-compose.yml         # multi-container deployment
├── Dockerfile
└── requirements.txt
```

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
