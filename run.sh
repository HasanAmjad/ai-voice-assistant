#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

OS="$(uname -s)"

info() { echo "[info] $*"; }
warn() { echo "[warn] $*"; }
fatal() { echo "[fatal] $*" >&2; exit 1; }

start_postgres() {
    case "$OS" in
        Darwin)
            brew services start postgresql@14 >/dev/null 2>&1 \
            || brew services start postgresql >/dev/null 2>&1 \
            || true
            ;;
        Linux)
            sudo systemctl start postgresql >/dev/null 2>&1 \
            || sudo service postgresql start >/dev/null 2>&1 \
            || true
            ;;
        *)
            warn "unrecognised OS '$OS' — start postgres manually if it isn't running."
            ;;
    esac
}

open_url() {
    case "$OS" in
        Darwin) command -v open >/dev/null 2>&1 && open "$1" || true ;;
        Linux) command -v xdg-open >/dev/null 2>&1 && xdg-open "$1" || true ;;
        *) command -v start >/dev/null 2>&1 && start "$1" || true ;;
    esac
}

if [ ! -d ".venv" ]; then
    fatal ".venv not found. Run: python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
fi
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

if ! command -v ffmpeg >/dev/null 2>&1; then
    warn "ffmpeg not on PATH. /predict will fail on uploaded audio."
    warn "macOS: brew install ffmpeg"
    warn "Ubuntu: sudo apt install ffmpeg"
fi

if ! pg_isready -h localhost -p 5432 -q 2>/dev/null; then
    info "postgres not running on :5432, attempting to start it"
    start_postgres
    for _ in {1..15}; do
        pg_isready -h localhost -p 5432 -q 2>/dev/null && break
        sleep 1
    done
    pg_isready -h localhost -p 5432 -q 2>/dev/null \
        || fatal "could not bring postgres up. Start it manually and retry."
fi

info "stopping any previous api/dashboard..."
pkill -f "python -u serving/api.py" >/dev/null 2>&1 || true
pkill -f "streamlit run serving/dashboard.py" >/dev/null 2>&1 || true
sleep 2

mkdir -p logs
: > logs/api.log
: > logs/dashboard.log

cleanup() {
    echo ""
    info "stopping services..."
    pkill -f "python -u serving/api.py" >/dev/null 2>&1 || true
    pkill -f "streamlit run serving/dashboard.py" >/dev/null 2>&1 || true
    exit 0
}
trap cleanup INT TERM

info "starting FastAPI on :8000..."
python -u serving/api.py > logs/api.log 2>&1 &
API_PID=$!

info "starting Streamlit on :8501..."
streamlit run serving/dashboard.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    > logs/dashboard.log 2>&1 &
DASH_PID=$!

info "waiting for api (Whisper load takes ~60 sec on cold start)..."
for _ in $(seq 1 120); do
    grep -q "Application startup complete" logs/api.log 2>/dev/null && break
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo ""
        warn "api process died. last 20 lines of logs/api.log:"
        tail -20 logs/api.log
        cleanup
    fi
    sleep 1
done

info "waiting for dashboard..."
for _ in $(seq 1 60); do
    grep -qE "External URL|You can now view" logs/dashboard.log 2>/dev/null && break
    if ! kill -0 "$DASH_PID" 2>/dev/null; then
        echo ""
        warn "dashboard process died. last 20 lines of logs/dashboard.log:"
        tail -20 logs/dashboard.log
        cleanup
    fi
    sleep 1
done

cat <<EOF

VoiceIntent is up.

API (Swagger): http://localhost:8000/docs
Dashboard: http://localhost:8501

Sample audio for live demo (Predict tab -> upload):
data/audio/samples/card_blocked.mp3 (high-confidence reply)
data/audio/samples/cancel_transfer.mp3 (high-confidence reply)
data/audio/samples/automatic_top_up.mp3 (low-confidence -> agent handoff)

Logs streaming below. Ctrl+C to stop both services.

EOF

(sleep 1 && open_url http://localhost:8501) &

tail -F -q logs/api.log logs/dashboard.log &
TAIL_PID=$!
wait "$TAIL_PID"
