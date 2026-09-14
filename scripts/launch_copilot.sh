#!/usr/bin/env bash
# Launch interview_copilot in single-app mode (#607): one process = the dashboard + the managed
# recorder + the three on-screen switches (transcription / suggestions / local-vs-local+api).
# This is what the desktop shortcut runs. It opens the browser itself; Ctrl-C here stops
# everything and releases the mic. Override the session or interpreter via env if needed:
#   COPILOT_SESSION=<id>   pick a different context bundle (default: example_ai_engineer)
#   COPILOT_PYTHON=<path>  a different interpreter (default: python on PATH)
#   COPILOT_MIC / COPILOT_SOURCE  the capture devices (see config/settings.py — set your external mic!)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the project root
cd "$HERE"
PY="${COPILOT_PYTHON:-python}"
SESSION="${COPILOT_SESSION:-example_ai_engineer}"
# Keep the local model RESIDENT for the whole call (#session-18 latency fix). Ollama unloads a model
# after 5 idle minutes, and the OpenAI-compatible endpoint the client uses cannot pass keep_alive,
# so a quiet gap mid-interview would make the next question pay a ~10 s cold load + prefill
# (measured 2026-09-14: cold 10.0 s TTFT vs 0.16 s warm). This pinger refreshes the timer every
# 2 min via the native API (empty prompt = load/refresh only, no generation). Dies with this shell.
MODEL="${OLLAMA_MODEL:-interview-copilot:14b}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"; OLLAMA_URL="${OLLAMA_URL%/v1}"
( while true; do
    curl -s -m 20 "$OLLAMA_URL/api/generate" -d "{\"model\":\"$MODEL\",\"keep_alive\":\"3h\"}" >/dev/null 2>&1 || true
    sleep 120
  done ) &
KEEPALIVE_PID=$!
trap 'kill "$KEEPALIVE_PID" 2>/dev/null' EXIT INT TERM
"$PY" scripts/dashboard.py --app --session "$SESSION" "$@"
