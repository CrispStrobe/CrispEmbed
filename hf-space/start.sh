#!/bin/bash
# Start the CrispEmbed C++ server in the background, then launch Gradio.

set -e

MODEL="${CRISPEMBED_MODEL:-all-MiniLM-L6-v2}"
OCR_MODEL="${CRISPEMBED_OCR_MODEL:-hmer-hw}"
SERVER_HOST="${CRISPEMBED_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${CRISPEMBED_SERVER_PORT:-8090}"
N_THREADS="${CRISPEMBED_THREADS:-4}"

echo "=== CrispEmbed Space ==="
echo "  Model:     $MODEL"
echo "  OCR model: $OCR_MODEL"
echo "  Server:    $SERVER_HOST:$SERVER_PORT"
echo "  Threads:   $N_THREADS"

# Auto-download and start the embedding server.
# -m accepts a model name (auto-resolves + downloads from HF registry).
crispembed-server \
  -m "$MODEL" \
  --ocr "$OCR_MODEL" \
  --host "$SERVER_HOST" \
  --port "$SERVER_PORT" \
  -t "$N_THREADS" &

SERVER_PID=$!

# Wait for the server to come up (up to 120s for first model download)
echo "Waiting for CrispEmbed server..."
for i in $(seq 1 120); do
  if curl -sf "http://$SERVER_HOST:$SERVER_PORT/health" >/dev/null 2>&1; then
    echo "Server ready!"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: Server process died"
    exit 1
  fi
  sleep 1
done

# Start Gradio
exec python3 /space/app.py
