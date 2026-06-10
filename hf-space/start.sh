#!/bin/bash
# Start the CrispEmbed C++ server in the background, then launch Gradio.

set -e

MODEL="${CRISPEMBED_MODEL:-all-MiniLM-L6-v2}"
OCR_MODEL="${CRISPEMBED_OCR_MODEL:-pix2tex-mfr}"
SERVER_HOST="${CRISPEMBED_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${CRISPEMBED_SERVER_PORT:-8090}"
N_THREADS="${CRISPEMBED_THREADS:-4}"

echo "=== CrispEmbed Space ==="
echo "  Model:     $MODEL"
echo "  OCR model: $OCR_MODEL"
echo "  Server:    $SERVER_HOST:$SERVER_PORT"
echo "  Threads:   $N_THREADS"

# Auto-download and start the embedding server
crispembed-server \
  --model "$MODEL" \
  --ocr "$OCR_MODEL" \
  --host "$SERVER_HOST" \
  --port "$SERVER_PORT" \
  --threads "$N_THREADS" \
  --auto-download &

SERVER_PID=$!

# Wait for the server to come up
echo "Waiting for CrispEmbed server..."
for i in $(seq 1 60); do
  if curl -sf "http://$SERVER_HOST:$SERVER_PORT/health" >/dev/null 2>&1; then
    echo "Server ready!"
    break
  fi
  sleep 1
done

# Start Gradio
exec python3 /space/app.py
