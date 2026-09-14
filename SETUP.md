# Setup

This is the full setup guide for running interview_copilot on a fresh machine. It has real external
dependencies (a GPU, Ollama, a Linux audio stack), so read the [Requirements](#requirements) first —
some of them are not `pip`-installable.

If you only want to run the **tests / code** (no audio, no model), jump to
[Code-only setup](#code-only-setup-no-gpu-no-audio).

---

## Requirements

| Need | Why | Notes |
|---|---|---|
| **Linux** with PipeWire or PulseAudio | audio is captured at the OS layer with `parec` | Built and tested on Ubuntu. The capture path is Linux-specific. |
| **NVIDIA GPU + CUDA** | local `faster-whisper` (STT) and the local LLM run on the GPU | ~16 GB VRAM comfortably runs Whisper + the 14B model together; 8 GB works with the 8b model. CPU is possible for STT but slow. |
| **Ollama** | the local, private LLM backend (the default) | https://ollama.com/download |
| **Python 3.11+** | the app | 3.13 is what CI runs. |
| **~15 GB disk** | model weights (Whisper + one or two Ollama models) | |
| A **headset** (headphones) | keeps the interviewer and your mic on separate channels | On speakers, the mic hears the interviewer and speaker attribution degrades. |

> **Disclosure & policy.** This is a **disclosed-use** tool (D11): only use it where the interview
> explicitly permits AI assistance, and tell the interviewer. It has no stealth/anti-detection
> features and never will (SI2). Confirm the policy for each interview.

---

## 1. System packages

```bash
# audio: parec (capture) + pactl (list sources), and the native PortAudio lib sounddevice needs
sudo apt update
sudo apt install -y pulseaudio-utils libportaudio2 portaudio19-dev
```

If you run Microsoft Teams, the **`teams-for-linux`** flatpak works because its sandbox shares the
audio server; a browser tab in the same PipeWire graph works too. What matters is that Teams' output
appears as a **monitor source** you can capture (step 5).

## 2. Ollama + the custom models

Install Ollama (link above), then build the two models the copilot uses. **This step is not
optional and not obvious:** the stock model tags ship a 4096-token context window, and a session
bundle is ~9–12k tokens. Ollama **silently truncates** an over-long prompt (no error) and its
OpenAI-compatible endpoint **ignores** a runtime `num_ctx`, so the window must be baked into the
model. The [`ollama/`](ollama/) Modelfiles do exactly that.

```bash
./ollama/build_models.sh          # builds interview-copilot:8b (fast) and :14b (higher quality)
# or just the fast one:
./ollama/build_models.sh 8b

ollama list | grep interview-copilot   # verify both/the model exist
```

- `interview-copilot:8b` — the fast default (~7 GB VRAM). Start here.
- `interview-copilot:14b` — tighter, more speakable answers (~11.5 GB VRAM); on a 16 GB card it
  leaves only ~2 GB of headroom alongside Whisper, so it is what breaks if something else claims
  VRAM mid-call. Fall back with `--model interview-copilot:8b`.

The GPU is shared between Ollama and Whisper — run `ollama ps` before a call.

## 3. Python environment

```bash
git clone https://github.com/Drzymek92/interview-copilot.git
cd interview-copilot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`faster-whisper` downloads its STT weights (`large-v3-turbo`) on first use.

## 4. Configure

```bash
cp config/.env.example config/.env
```

The defaults are already correct for a local Ollama run — you usually don't need to change anything.
`config/.env` is where secrets and machine-specific endpoints live; non-secret tunables (STT model,
segmentation, meter, salience, model choice) are in [`config/settings.py`](config/settings.py) and
overridable by env or CLI (precedence: CLI > env > config default). Key vars:

- `REASONING_BACKEND` — `local` (default) keeps everything on your machine. `cloud` is an **explicit
  egress** of the transcript + your bundle to a third party; it is opt-in, announced on every call,
  and refuses to run unless all three `CLOUD_*` vars are set (SI1). Leave it `local` unless you mean it.
- `OLLAMA_MODEL` — defaults to `interview-copilot:14b`; set to `interview-copilot:8b` for less VRAM.

Check the backend is alive:

```bash
python scripts/llm_client.py --smoke
```

## 5. Find your audio sources

```bash
python scripts/live_transcribe.py --list
```

You need two device names:
- the **monitor** of the sink Teams plays into (the interviewer's audio), and
- your **microphone**.

> **Gotcha:** `--mic auto` resolves the system *default* input, which is often a webcam mic rather
> than your good USB mic. Pass `--mic` explicitly (or set `COPILOT_MIC`).

Prove capture → STT works with no call, using a sink monitor as a fake mic:

```bash
python scripts/spike_capture.py --selftest-sink <your-sink-monitor> --seconds 8
```

## 6. A session bundle

The copilot needs a per-interview **context bundle** (JD, company brief, résumé, STAR bank, honesty
boundary, plan). A complete **synthetic example** ships at
[`examples/sessions/example_ai_engineer/`](examples/sessions/example_ai_engineer/) and the quickstart
runs against it directly.

For a real interview, build your own — copy the example and replace every field per
[`docs/SESSION_BUNDLE.md`](docs/SESSION_BUNDLE.md):

```bash
mkdir -p scripts/inputs/sessions
cp -r examples/sessions/example_ai_engineer scripts/inputs/sessions/my_interview
# then edit scripts/inputs/sessions/my_interview/*  (bundle.json + the .md files)
```

`scripts/inputs/` is gitignored — your real interview data never leaves your machine.

---

## Running it

### The dashboard (the surface you actually run an interview on)
Single-app mode manages the recorder for you and exposes three on-screen switches (transcription,
suggestions, local-vs-cloud):

```bash
python scripts/dashboard.py --app --session my_interview
# → opens http://127.0.0.1:8765  (loopback only)
```

Transcription starts **off** — flip it on when the call starts. Set your mic first:

```bash
COPILOT_MIC=<your-mic-source> python scripts/dashboard.py --app --session my_interview
```

### Dry-run without a live call
Drive the dashboard from a finished transcript at its true cadence:

```bash
python scripts/replay_transcript.py <a-transcript.txt> --speed 8
# then point the dashboard at the file it prints (pass the --ceiling-seconds it tells you)
```

### Reasoning only / one-shot
```bash
python scripts/reasoning.py --session example_ai_engineer --text "Tell me about a hard ML project"
python scripts/reasoning.py --session my_interview --watch          # follow the newest live transcript
```

### Desktop launcher (optional)
```bash
sed "s|__PROJECT_DIR__|$PWD|g" interview-copilot.desktop.template \
  > ~/.local/share/applications/interview-copilot.desktop
# then mark it trusted (file manager: right-click ▸ Allow Launching)
```
Set a default session with `COPILOT_SESSION` (see [`scripts/launch_copilot.sh`](scripts/launch_copilot.sh)).

---

## Code-only setup (no GPU, no audio)

To hack on the code and run the tests, you can skip Ollama and the audio stack:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/        # model/live tests self-skip when Ollama is unreachable
ruff check .
```

---

## Troubleshooting

- **Answers ignore most of the bundle / the model translates the question instead of answering.**
  Your Ollama model has the stock 4096 window. Rebuild with `./ollama/build_models.sh` and confirm
  with `ollama list`. `llm_client` logs `PROMPT TRUNCATED` when it detects this.
- **`ollama create` can't read the Modelfile.** If Ollama is a snap/flatpak it may not read paths
  outside `$HOME` — copy the `ollama/` directory under your home directory and build from there.
- **Transcript is garbage / every segment runs to the length cap.** `webrtcvad` alone labels steady
  mic hiss as speech; the app pairs it with an adaptive noise floor, but a very noisy mic or a
  near-silent window can still misbehave. Use a decent mic and check `--list` picked the right one.
- **Only one side of the call is transcribed.** A sink monitor does not contain your own voice — the
  app captures the mic as a second stream. Make sure `--mic` is set correctly.
- **Wrong mic.** `--mic auto` is the webcam mic on many machines; pass `--mic` / `COPILOT_MIC`.
- **VRAM error mid-call with the 14B.** It leaves little headroom next to Whisper. Run `ollama ps`
  before starting, unload stray models (`ollama stop <model>`), or use `--model interview-copilot:8b`.
- **A cloud call refuses to run.** By design: `cloud` needs all three `CLOUD_*` vars and prints an
  egress banner naming the host. That is the SI1 guard, not a bug.
- **Dashboard shows nothing / websocket won't connect.** It binds loopback only; open
  `http://127.0.0.1:8765` on the same machine, not a LAN address.
