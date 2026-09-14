#!/usr/bin/env bash
# Build the custom Ollama models the copilot needs (context window baked to 16384 — see the
# Modelfiles for why the stock tags will not do). Run once, after installing Ollama.
#
#   ./ollama/build_models.sh            # builds both the 8b (fast default) and 14b (higher quality)
#   ./ollama/build_models.sh 8b         # builds only interview-copilot:8b
#
# NOTE: `ollama create` reads the Modelfile from disk; if your Ollama is a snap/flatpak it may not
# be able to read paths outside $HOME — in that case copy the ollama/ dir under $HOME first.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHICH="${1:-both}"

pull_if_missing() { ollama list | grep -q "^$1" || { echo "pulling base model $1 ..."; ollama pull "$1"; }; }

if [[ "$WHICH" == "8b" || "$WHICH" == "both" ]]; then
  pull_if_missing "llama3.1:8b"
  echo "building interview-copilot:8b ..."
  ollama create interview-copilot:8b -f "$HERE/Modelfile.interview_copilot"
fi

if [[ "$WHICH" == "14b" || "$WHICH" == "both" ]]; then
  pull_if_missing "qwen3:14b"
  echo "building interview-copilot:14b ..."
  ollama create interview-copilot:14b -f "$HERE/Modelfile.ic_qwen"
fi

echo
echo "done. Verify:"
echo "  ollama list | grep interview-copilot"
