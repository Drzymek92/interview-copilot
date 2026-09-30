#!/usr/bin/env bash
# Launch the interview copilot hub (D35): one process = the opening menu (Start a call · Train ·
# Review · Generate context · Settings) + every mode view. This is what the desktop shortcuts run.
# It opens the browser itself; Ctrl-C here stops everything and releases the mic.
#   COPILOT_SESSION=<id>   optional: preselect this context bundle on the Call screen (the shortcut
#                          for a specific interview); without it the browser opens on the menu
#   COPILOT_PYTHON=<path>  a different interpreter (default: python on PATH)
# Devices, models and per-mode behaviour are set in the app's Settings (config/user_settings.json,
# D36); an environment variable still wins over a saved value (COPILOT_MIC, COPILOT_SOURCE, ...).
# The pre-hub single-screen app is still `python scripts/dashboard.py --app --session <id>`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the project root
cd "$HERE"
PY="${COPILOT_PYTHON:-python}"
SESSION="${COPILOT_SESSION:-}"
# Keep the local model RESIDENT for the whole call (#session-18 latency fix). Ollama unloads a model
# after 5 idle minutes, and the OpenAI-compatible endpoint the client uses cannot pass keep_alive,
# so a quiet gap mid-interview would make the next question pay a ~10 s cold load + prefill
# (measured 2026-09-14: cold 10.0 s TTFT vs 0.16 s warm). This pinger refreshes the timer every
# 2 min via the native API (empty prompt = load/refresh only, no generation). Dies with this shell.
# The model resolved exactly as the app resolves it (env > saved Settings > default, D36).
MODEL="$("$PY" -c 'from config import settings; print(settings.LOCAL_MODEL)' 2>/dev/null || true)"
MODEL="${MODEL:-${OLLAMA_MODEL:-interview-copilot:14b}}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"; OLLAMA_URL="${OLLAMA_URL%/v1}"
( while true; do
    curl -s -m 20 "$OLLAMA_URL/api/generate" -d "{\"model\":\"$MODEL\",\"keep_alive\":\"3h\"}" >/dev/null 2>&1 || true
    sleep 120
  done ) &
KEEPALIVE_PID=$!
trap 'kill "$KEEPALIVE_PID" 2>/dev/null' EXIT INT TERM
OPEN_PATH="/"
if [ -n "$SESSION" ]; then OPEN_PATH="/call?session=$SESSION"; fi
"$PY" scripts/app_hub.py --open-path "$OPEN_PATH" "$@"
