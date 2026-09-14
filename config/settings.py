"""Non-secret runtime settings for interview_copilot (CFG).

Precedence: CLI > env > config-default. Secrets never live here (D14 — provider
keys come from the environment). This module holds only tunables that Day-2/3/4
will also read, so the STT model choice and audio params are settable in one place.
"""

from __future__ import annotations

import os

# --- STT (D16 — local GPU faster-whisper) ---
# The interview runs MOSTLY IN POLISH with English fragments (user, 2026-08-31),
# so the model MUST be multilingual.
#
# WARNING — do not go back to `distil-large-v3`. Distil-Whisper models are
# ENGLISH-ONLY (the Day-1 comment here claimed "multilingual"; that was wrong and
# it is why the first live Teams test returned garbage). Forced onto Polish speech
# it emitted plausible-looking English nonsense: real call audio came back as
# "Thank you, Mr. Michael. I am, very, on the conversation A.I. Enginer LLM".
#
# Chosen on measured A/B against the real Teams recording (2026-08-31, 50s Polish):
#   large-v3-turbo  forced pl : decode 0.35s (rtf 0.007), load 36s  <- CHOSEN
#   large-v3        forced pl : decode 0.93s (rtf 0.019), load 66s  (no better here)
# Both produced clean Polish; turbo is 2.6x faster and got "AI Engineer" right
# where large-v3 said "AI Engineering". large-v3 stays the fallback if turbo
# regresses on harder audio.
STT_MODEL: str = os.environ.get("STT_MODEL", "large-v3-turbo")

# Device / precision. RTX 5060 Ti (Blackwell sm_120) has CUDA; float16 is the
# fastest safe compute type. Override for CPU practice runs (STT_DEVICE=cpu,
# STT_COMPUTE_TYPE=int8).
STT_DEVICE: str = os.environ.get("STT_DEVICE", "cuda")
STT_COMPUTE_TYPE: str = os.environ.get("STT_COMPUTE_TYPE", "float16")

# faster-whisper decode knobs. beam_size=5 measured at rtf 0.007 on turbo — the
# latency headroom is ~100x, so greedy buys nothing worth having; take the accuracy.
STT_BEAM_SIZE: int = int(os.environ.get("STT_BEAM_SIZE", "5"))

# Forced `pl` beat auto-detect on the real recording: identical text, faster decode
# (0.35s vs 0.49s), and no risk of language detection flipping on a short ambient
# window. English fragments still transcribe as English with pl forced.
# KNOWN WEAKNESS (measured, unsolved): both models garble the POLISH->ENGLISH
# code-switch seam ("jak i a do tyłu bit of English"). Clean either side of it.
# This is the FORCED language when detection is off, and the FALLBACK when detection is
# uncertain. It is deliberately Polish — the interview is mostly Polish (see below).
STT_LANGUAGE: str = os.environ.get("STT_LANGUAGE", "pl")

# --- Per-segment language detection (user decision 2026-09-02) ---
# The interview is mostly Polish with occasional fully-English questions. To answer each
# question in its OWN language, STT detects the language per segment instead of forcing pl.
# It is BIASED to Polish: Whisper detects the segment's language, but we only accept the
# detection when it is one of STT_LANGUAGE_CANDIDATES *and* clears STT_LANGUAGE_MIN_PROB —
# otherwise we fall back to STT_LANGUAGE (pl) and decode in Polish. A Polish sentence with
# embedded English tech terms detects as pl (its dominant language), which is what we want.
# Turn this off (STT_DETECT_LANGUAGE=0) to restore the proven forced-pl path.
STT_DETECT_LANGUAGE: bool = os.environ.get("STT_DETECT_LANGUAGE", "1") not in ("0", "false", "False")
STT_LANGUAGE_CANDIDATES: tuple[str, ...] = tuple(
    c.strip() for c in os.environ.get("STT_LANGUAGE_CANDIDATES", "pl,en").split(",") if c.strip()
)
STT_LANGUAGE_MIN_PROB: float = float(os.environ.get("STT_LANGUAGE_MIN_PROB", "0.7"))

# --- Straddling-segment code-switch handling (D26, G8 d / #397, measured 2026-09-05) ---
# ONE `detect_language` call per segment returns ONE language token, and on the real HR call
# that token is wrong for a segment that STRADDLES a language switch. Measured on the two
# known failures: at [14:39-15:09] the segment is ~70% English and the whole-segment vote is
# still `pl` (p=0.58); at [17:53-18:23] it opens with 9 s of English and the vote is `pl`
# (p=0.99). So STT_LANGUAGE_MIN_PROB catches the FIRST and not the second — the whole-segment
# probability is not the uncertainty signal this needs. Sub-window detection over the SAME
# audio recovers the truth in both (windows vote pl,pl,en,en,... and en,en,pl,pl,...), which
# is what this knob set uses as the trigger.
#   off     — one language per segment, the pre-#397 path.
#   rescore — candidate (a): on a detected switch, decode the segment once per candidate
#             language and keep the higher duration-weighted avg_logprob.
#   split   — candidate (b): on a detected switch, cut the AUDIO at the detected boundary and
#             decode each part in its own language. This does NOT touch segmentation: the
#             transcript line keeps its VAD start/end, so D21 and #400's SEGMENT_MAX_SECONDS
#             lever are untouched.
# DEFAULT = split (D26, chosen on the measurement below; `rescore` is the rejected alternative). `off` restores the pre-#397 path
# exactly and costs exactly what it used to.
STT_CODESWITCH_MODE: str = os.environ.get("STT_CODESWITCH_MODE", "split")
# Sub-window geometry for the switch detector. 6 s / 3 s was chosen BEFORE any result was
# seen (it is two windows per SEGMENT_MIN_SECONDS) and not tuned afterwards.
STT_CODESWITCH_WINDOW_SECONDS: float = float(
    os.environ.get("STT_CODESWITCH_WINDOW_SECONDS", "6.0")
)
STT_CODESWITCH_HOP_SECONDS: float = float(os.environ.get("STT_CODESWITCH_HOP_SECONDS", "3.0"))
# A language must hold this many CONSECUTIVE confident windows to count as a side of a switch.
# 1 flags any disagreement; 2 ignores a single blip in one window. On the real call this is a
# COST knob, not a correctness one — at 1 it fires on 6/110 segments, at 2 on 2/110, and the
# four extra segments resolve to the same language either way.
STT_CODESWITCH_MIN_WINDOWS: int = int(os.environ.get("STT_CODESWITCH_MIN_WINDOWS", "2"))
# Segments shorter than this are never probed: below window+hop there are not two windows to
# disagree, and the probe would cost an encoder pass to learn nothing.
STT_CODESWITCH_MIN_SECONDS: float = float(
    os.environ.get("STT_CODESWITCH_MIN_SECONDS", "10.0")
)
# How much of the segment the detector looks at BEFORE it has any reason to suspect a switch.
#   ends — probe only the first and last window. A switch OF THE UTTERANCE'S LANGUAGE ends the
#          segment in a different language than it starts, so two encoder passes decide it;
#          the full sliding scan is then paid only by the segments that actually straddle one.
#   full — slide over the whole segment every time.
# This is a LATENCY knob, and the P5 meter is what makes it one. MEASURED on the real call
# (2026-09-05): the full scan costs +1.363 s on a 30 s segment, and 54% of that call's segments
# are 30 s ones — enough to push nearly every closing line past the 31.0 s ceiling
# (SEGMENT_MAX_SECONDS + METER_DECODE_ALLOWANCE_SECONDS). `ends` costs two passes, ~0.27 s, and
# keeps a max-length line inside the ceiling. KNOWN LIMIT: a switch that returns before the
# segment ends (pl -> en -> pl) leaves both ends agreeing and is not detected — such a segment
# is then served exactly as it is today, no worse.
STT_CODESWITCH_SCAN: str = os.environ.get("STT_CODESWITCH_SCAN", "ends")

# --- Suggestion language (user decision 2026-08-31, revised 2026-09-02) ---
# The on-screen suggestion language is a separate axis from the spoken language. Three modes:
#   "match" (default) — answer each question in the language it was ASKED in (uses the
#            per-segment detection above); a Polish question gets a Polish scaffold, an
#            English question an English one.
#   "en" / "pl" — force every suggestion into that language regardless of the question.
# Must be SWITCHABLE AT SESSION START (user requirement) — the dashboard exposes it too.
SUGGESTION_LANGUAGE: str = os.environ.get("SUGGESTION_LANGUAGE", "match")  # "match" | "en" | "pl"

# --- Audio capture (D17) ---
# Whisper expects 16 kHz mono float32; keep capture at that rate to avoid resampling.
SAMPLE_RATE: int = int(os.environ.get("SAMPLE_RATE", "16000"))
CHANNELS: int = int(os.environ.get("CHANNELS", "1"))

# --- Live segmentation (scripts/live_transcribe.py) ---
# MEASURED CONSTRAINT (2026-08-31, same audio): window length drives accuracy —
#   60s -> 3.8% WER / 10-10 technical terms · 15s -> 5.7% / 9-10 · 8s -> 7.6% / 8-10.
# Losses land on multi-word English phrases because a fixed-time cut bisects them.
# Decode latency is nowhere near binding (max 0.36 s per chunk, rtf ~0.01), so the
# loop buys accuracy with LONG segments that end at a natural pause. These knobs are
# that policy; the loop must not hardcode them (CFG).
#
# webrtcvad aggressiveness 0-3 (3 = most eager to call audio non-speech). 2 for call
# audio: 3 clips quiet sentence tails, 0-1 lets keyboard/fan noise hold a segment open.
VAD_AGGRESSIVENESS: int = int(os.environ.get("VAD_AGGRESSIVENESS", "2"))
# webrtcvad accepts ONLY 10/20/30 ms frames at 8/16/32/48 kHz.
VAD_FRAME_MS: int = int(os.environ.get("VAD_FRAME_MS", "20"))

# Adaptive noise floor on top of webrtcvad. MEASURED 2026-08-31: webrtcvad alone
# calls steady microphone hiss "speech" — on a live two-stream run the webcam mic's
# noise held one segment open for the full 30 s cap and the decode came back as
# garbage ("...to have worked on...", "KONIEC!"). So a frame counts as speech only
# if it is ALSO meaningfully louder than that channel's own recent noise floor
# (10th-percentile frame RMS over the trailing window). Per channel, because mic and
# monitor levels differ by a lot.
VAD_NOISE_WINDOW_SECONDS: float = float(os.environ.get("VAD_NOISE_WINDOW_SECONDS", "20.0"))
VAD_SPEECH_RMS_MULT: float = float(os.environ.get("VAD_SPEECH_RMS_MULT", "3.0"))
VAD_SPEECH_MIN_RMS: float = float(os.environ.get("VAD_SPEECH_MIN_RMS", "0.004"))

# A/B THAT SETTLED THE POLICY (2026-08-31, identical 37.7 s audio pushed through the
# LIVE loop with --from-wav; reference scripts/outputs/ab_source_en.wav):
#   VAD pause>=0.9s  -> WER  1.8%  terms  9/10   <- CHOSEN (2 segments)
#   VAD pause>=0.45s -> WER  1.8%  terms  9/10      (4 segments; same text, lower latency)
#   fixed  8 s       -> WER  5.4%  terms  8/10
#   fixed 15 s       -> WER 24.1%  terms  7/10      (short tail window HALLUCINATED
#                                                    fluent Polish: "Znaczymy, jak u
#                                                    mnie zbierzal?" — forced pl on a
#                                                    content-poor window)
#   fixed 8 s + 1.5 s overlap -> WER 44.6%  terms 7/10
# The one term VAD loses is "LoRA", which the TTS voice pronounces "low array" — an
# artifact of the synthetic source, not of the loop. Caveat: this is READ English TTS
# with clean sentence pauses, so it flatters VAD; it reproduces the MECHANISM (fixed
# cuts bisect phrases and starve short windows) rather than predicting live numbers.
#
# A pause this long ends a segment (a natural sentence/turn boundary).
SEGMENT_SILENCE_SECONDS: float = float(os.environ.get("SEGMENT_SILENCE_SECONDS", "0.9"))
# ...but only once the segment is this long, so short bursts glue into a usable window
# instead of each "mhm" becoming its own decode.
SEGMENT_MIN_SECONDS: float = float(os.environ.get("SEGMENT_MIN_SECONDS", "4.0"))
# Hard cap so an uninterrupted monologue still reaches the screen. Above the 15 s knee
# in the table, and inside Whisper's own 30 s receptive window.
# STAYS 30 s (D28, #400): 20 s was measured on the real call — it changes 10.0 %/20 s-window of
# tokens (sign unmeasured, D24), pushes force-cuts 52 %->65 %, and its -10.1 s latency win is
# largely already bought back by D25; D26 and the D27 meter both survive it. Lowering this is a
# per-session CFG choice, not the default. Re-evaluate after technical-round audio (#323) + a
# human WER pass (#387). See design/MEASUREMENT_segment_cap_400.md.
SEGMENT_MAX_SECONDS: float = float(os.environ.get("SEGMENT_MAX_SECONDS", "30.0"))
# Close a stalled short segment after this much silence even if it never reached
# SEGMENT_MIN_SECONDS (otherwise a lone short burst waits forever).
SEGMENT_MAX_SILENCE_SECONDS: float = float(
    os.environ.get("SEGMENT_MAX_SILENCE_SECONDS", "2.5")
)
# Audio kept from *before* VAD fires, so a segment never starts mid-word.
SEGMENT_PREROLL_SECONDS: float = float(os.environ.get("SEGMENT_PREROLL_SECONDS", "0.4"))
# Only a SEGMENT_MAX_SECONDS force-cut lands mid-speech; carry this much audio into the
# next segment so a bisected phrase survives whole in one of them. The duplicated words
# are then removed from the TEXT by the overlap de-duplicator.
SEGMENT_CARRYOVER_SECONDS: float = float(
    os.environ.get("SEGMENT_CARRYOVER_SECONDS", "1.5")
)
# Segments quieter than this are dropped WITHOUT a decode: Whisper hallucinates fluent
# filler ("Dziekuje za uwage", "Napisy stworzone przez...") on near-silence.
SEGMENT_MIN_PEAK: float = float(os.environ.get("SEGMENT_MIN_PEAK", "0.005"))
# Minimum voiced audio for a segment to be worth decoding at all.
SEGMENT_MIN_SPEECH_SECONDS: float = float(
    os.environ.get("SEGMENT_MIN_SPEECH_SECONDS", "0.5")
)

# --- D25 provisional lines (#399) — written by the RECORDER, decoded on its own model ---
# The screen is blank for a MEASURED median 28 s mid-answer (OPEN_DESIGN P5). D25's answer is a
# growing-prefix decode of the STILL-OPEN segment, written to a SEPARATE
# scripts/outputs/live_transcript_<stamp>.partial that consumers tail exactly as they tail the
# .txt; the closing .txt line always supersedes it and D19's contract for the .txt is untouched.
# The cost is VRAM, not compute, which is why this cannot live in a consumer: a second Whisper in
# the dashboard process is ~2.2 GB against the ~2.1 GB the 14B leaves, while the recorder's model
# is idle 97.4% of the call.
# Default ON: the .partial is a separate file that nothing reads yet, so an enabled default buys
# the artefact the #323 dry-run needs and cannot change one byte of the transcript, the scorer
# file or the WAV. Turn it off with PARTIAL_DECODE_ENABLED=0 or `--no-partials`.
PARTIAL_DECODE_ENABLED: bool = os.environ.get("PARTIAL_DECODE_ENABLED", "1") not in (
    "0", "false", "False"
)
# Cadence: re-decode the open segment this often. MEASURED on the real 42-min HR call (2026-09-04):
# decode costs 0.248 + 0.0144*audio_s and is 2.6% of GPU duty, so a 5 s cadence adds ~10% duty
# (487 decodes ~= 244 GPU-s). Token survival of a 5 s prefix into the final line is 0.86 (0.94 @10 s,
# 0.96 @15-20 s) over 86 prefixes of that call — which is CHURN, not accuracy (D24): no human
# reference exists for that audio. Shortening this buys freshness and loses survival; lengthening it
# does the reverse. It is NOT a latency knob for the .txt — SEGMENT_MAX_SECONDS owns that (#400).
PARTIAL_DECODE_SECONDS: float = float(os.environ.get("PARTIAL_DECODE_SECONDS", "5.0"))
# The CONSUMER half of D25 (#399): does the dashboard render the provisional line at all?
# Separate from PARTIAL_DECODE_ENABLED, which is the recorder's knob — the recorder can keep
# writing the artefact for the #323 dry-run and D11 scoring while a screen chooses not to show
# provisional text. Default ON: an empty slot is the state a run without a .partial already has,
# and the whole point of D25 is the MEASURED 28 s the screen is otherwise blank (OPEN_DESIGN P5).
DASHBOARD_SHOW_PARTIALS: bool = os.environ.get("DASHBOARD_SHOW_PARTIALS", "1") not in (
    "0", "false", "False"
)

# --- Reasoning backend (D14 — BYOK cloud default, local fallback; SI1) ---
# SI1 flips D14's *default* on this seat: the decision names cloud as the preferred
# backend for quality, but SI1 forbids a silent egress of transcript + bundle, so the
# shipped default is LOCAL and cloud is an explicit opt-in (REASONING_BACKEND=cloud +
# three CLOUD_* env vars, announced by llm_client.announce_backend on every build).
REASONING_BACKEND: str = os.environ.get("REASONING_BACKEND", "local")  # "local" | "cloud"

# Local (Ollama) model. RE-DECIDED 2026-09-02 after two client-side fixes made a bigger
# model affordable. All figures: a representative ~9.2k-prompt-token bundle, 3 reps, one model
# resident at a time, streaming, thinking disabled.
#
#   model                   TTFT     total   out   tok/s   VRAM    + Whisper (2.2 GB)
#   interview-copilot:8b    0.14s    3.33s   216    67.5   6.9 GB   9.2 / 15.9  OK
#   interview-copilot:14b   0.15s    5.86s   220    38.5  11.5 GB  13.8 / 15.9  OK  <- CHOSEN
#   (measured live E2E with Whisper actually resident: peak 11.7 GB of 15.9)
#
# WHY THE BIGGER MODEL IS NOW THE RIGHT CHOICE — the reasoning inverted twice, so read this
# before "optimising" it back:
#   * TTFT is ~0.15s for BOTH, at any prompt size. Prefill is free; the wait is generation.
#     Since the output streams, what the user experiences is the POINT line, and that lands
#     instantly either way. The 14B costs 2.5s more to FINISH, while the user is already talking.
#   * Before streaming + the raw-HTTP path, the 14B took ~18s end-to-end and was unusable.
#     Fixing the client, not the model, is what bought the quality.
#   * 14B output is tighter and more speakable, and it reaches for the prep doc's own framing
#     ("the hard part isn't the model - it's everything the model has to touch").
#     8B padded, and its DETAIL section restated the bullets.
# COST: headroom drops from 6.7 GB to ~2.1 GB. If anything else claims VRAM mid-call the 14B
# is what breaks. Fall back instantly with `--model interview-copilot:8b` (or OLLAMA_MODEL).
#
# Both are custom builds - stock tags ship num_ctx=4096 and silently truncate this bundle:
#   printf 'FROM llama3.1:8b\nPARAMETER num_ctx 16384\n' > ~/Modelfile.interview_copilot
#   printf 'FROM qwen3:14b\nPARAMETER num_ctx 16384\n'   > ~/Modelfile.ic_qwen
#   ollama create interview-copilot:8b  -f ~/Modelfile.interview_copilot
#   ollama create interview-copilot:14b -f ~/Modelfile.ic_qwen
# qwen3:14b is a 40K-context model, so 16384 is a VRAM choice, not a ceiling.
# gpt-oss:20b was rejected on arithmetic: 14 GB + 2.2 GB Whisper does not fit in 15.9 GB.
# Earlier note about llama3.2:3b still stands: fast, but answers with meta-instructions.
LOCAL_MODEL: str = os.environ.get("OLLAMA_MODEL", "interview-copilot:14b")
# VRAM measured with both models resident: 7.8 GB of 15.9 GB. 8b alone is 5.3 GB, leaving
# room for large-v3-turbo. Still run `ollama ps` before a call (ENVIRONMENT.md).

# Qwen3 thinks by default and Ollama hides the reasoning tokens from `content` while still
# charging you the wall-clock for them. See llm_client.get_llm for the measurement.
LOCAL_DISABLE_THINKING: bool = os.environ.get("LOCAL_DISABLE_THINKING", "1") not in ("0", "false", "False")

# Stream from Ollama over plain HTTP instead of through langchain. MEASURED 2026-09-02 on the
# identical prompt and model: 66.2 tok/s raw vs 21.3 tok/s through langchain_openai.stream() —
# a 3.1x penalty in the CLIENT, not the model, and the single biggest latency lever found.
# Local only; the cloud path stays on langchain. Set to 0 to fall back if this ever misbehaves.
LOCAL_FAST_STREAM: bool = os.environ.get("LOCAL_FAST_STREAM", "1") not in ("0", "false", "False")

# Cloud model: NO DEFAULT ON PURPOSE. BYOK means the user brings base_url + model + key;
# a baked-in default is one env var away from a silent egress (SI1).
CLOUD_MODEL: str = os.environ.get("CLOUD_MODEL", "")

# Prompt caching on the cloud backend. The system block (the whole context bundle) is
# identical on every call, so caching it turns ~14k full-price input tokens into ~14k
# cache-read tokens billed at ~0.1x. Write costs ~1.25x once. Local is unaffected — Ollama
# manages its own KV cache and reports no cache usage.
CLOUD_PROMPT_CACHE: bool = os.environ.get("CLOUD_PROMPT_CACHE", "1") not in ("0", "false", "False")

# Deterministic answers: this is a factual scaffold from a fixed bundle, not creative writing.
REASONING_TEMPERATURE: float = float(os.environ.get("REASONING_TEMPERATURE", "0.0"))
# Hard ceiling per call. A suggestion the user cannot skim mid-answer is worse than none,
# and the cap is also the main latency lever on a local model (tokens dominate wall-clock).
SUGGESTION_MAX_TOKENS: int = int(os.environ.get("SUGGESTION_MAX_TOKENS", "220"))
# Abandon a call that outlives its usefulness rather than printing it late.
REASONING_TIMEOUT_SECONDS: float = float(os.environ.get("REASONING_TIMEOUT_SECONDS", "25.0"))

# --- Ambient trigger policy (G6 — when the loop fires the LLM) ---
# Determinism First (CLAUDE.md): a rule-based question detector, NOT an LLM classifier per
# segment. It costs no tokens, adds no latency, and cannot hallucinate.
# MEASURED 2026-09-01 against tests/fixtures/question_fixtures.json (60 hand-labelled pl/en
# interview lines), reproduce with `python scripts/reasoning.py --selftest-heuristic`:
#   first cut (? + interrogative in first 3 words + prompt verb) : P 0.86 / R 1.00 / F1 0.92
#   after the idiom guards                                       : P 1.00 / R 1.00 / F1 1.00
# READ THE SECOND NUMBER WITH SUSPICION. The five false positives that motivated the guards
# came from this same fixture set, so 1.00 is measured on lines the rule was then fixed
# against — it is not an out-of-sample result. What the guards encode is real Polish idiom
# ("jak Pan widzi" = "as you can see", "co ciekawe" = "interestingly") and one genuine
# grammatical fact (English auxiliaries only ask by inversion, so they must lead), not a
# tuned threshold. The honest claim is "no known false positive class remains"; the real
# rate is unknown until it runs on a live call transcript (#324).
# An LLM classifier per segment would cost a whole second model call to beat this — the
# expensive option, not the default.
FIRE_ON_QUESTIONS_ONLY: bool = os.environ.get("FIRE_ON_QUESTIONS_ONLY", "1") not in ("0", "false", "False")
# Segments shorter than this are turn-taking noise ("mhm", "tak, jasne"), not a question
# worth a model call, even when they end in a question mark.
SUGGESTION_MIN_WORDS: int = int(os.environ.get("SUGGESTION_MIN_WORDS", "4"))
# Never fire twice inside this window: a long question often arrives as two segments and
# the second suggestion would land on top of the first, unread.
SUGGESTION_COOLDOWN_SECONDS: float = float(os.environ.get("SUGGESTION_COOLDOWN_SECONDS", "8.0"))
# How many previous transcript lines are shown to the model as conversation context. Each is
# labelled by speaker ("Interviewer:" / "You:", G9/#326) so the model knows whose turn it was.
SUGGESTION_HISTORY_LINES: int = int(os.environ.get("SUGGESTION_HISTORY_LINES", "4"))
# How many past suggestions the dashboard keeps for live scrollback ("return to a previous
# suggestion during the interview") and, on complete ones, appends to live_suggestions_<stamp>.jsonl.
# Bounds memory on a long call; 0 disables the history panel entirely.
SUGGESTION_HISTORY_MAX: int = int(os.environ.get("SUGGESTION_HISTORY_MAX", "50"))
# Whose turns the copilot answers (G9/P4, #326). The monitor channel carries ONLY the
# interviewer ("them") and the mic ONLY the candidate ("you") — on headphones there is no
# acoustic bleed, so live_transcribe tags each line by its dominant channel. "them" (the
# default) means the copilot never fires on the candidate's own questions; "any" restores
# the old channel-blind behaviour. An UNTAGGED line (old transcript, or a monitor-only
# run with no mic) is always treated as "them" so nothing is silently dropped.
ANSWER_SPEAKER: str = os.environ.get("ANSWER_SPEAKER", "them")  # "them" | "you" | "any"

# --- Context bundle (D13) ---
# Per-session bundle root: scripts/inputs/sessions/<session_id>/bundle.json
SESSIONS_DIRNAME: str = os.environ.get("SESSIONS_DIRNAME", "sessions")
# How often the reasoning layer re-reads the growing transcript file (D19 seam). The
# transcript is fsync'd per line, so this is pure added latency: 0.25 s against a 5-30 s
# segment cadence is 1-5% of the budget, which is what made file-tailing affordable.
TRANSCRIPT_POLL_SECONDS: float = float(os.environ.get("TRANSCRIPT_POLL_SECONDS", "0.25"))

# --- Dashboard (D18 — bind 127.0.0.1 only, SI1/SI2) ---
DASHBOARD_HOST: str = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT: int = int(os.environ.get("DASHBOARD_PORT", "8765"))

# --- Single-app mode (#607): the dashboard spawns/kills the recorder itself ---
# `--app` turns the dashboard into the one app the desktop shortcut launches: it manages
# live_transcribe.py as a child so the on-screen "transcription" switch really starts and stops
# audio capture (mic released, ~2.2 GB Whisper VRAM freed — a true privacy stop, not a paused
# display). These name the capture devices the child is spawned with. "auto" matches
# live_transcribe's own defaults, but NOTE the gotcha: `--mic auto` resolves the lab's WEBCAM
# mic, not your external mic — set COPILOT_MIC to your external mic source for a real call.
COPILOT_SOURCE: str = os.environ.get("COPILOT_SOURCE", "auto")   # monitor of the sink Teams plays to
COPILOT_MIC: str = os.environ.get("COPILOT_MIC", "auto")         # your microphone (set explicitly!)
# Open the browser at the dashboard URL on launch (the shortcut wants this; a headless run does not).
COPILOT_OPEN_BROWSER: bool = os.environ.get("COPILOT_OPEN_BROWSER", "1") not in ("0", "false", "False")
# Seconds the app waits for the freshly-spawned recorder to write its first transcript file
# before giving up and reporting the recorder failed to start.
COPILOT_RECORDER_WAIT_SECONDS: float = float(os.environ.get("COPILOT_RECORDER_WAIT_SECONDS", "30.0"))
# How many transcript lines the server keeps for a late-joining browser. The real 42-min HR
# call produced 110 lines, so 400 holds a long technical round whole; beyond it the oldest
# are dropped and the UI says so rather than silently showing a truncated call.
DASHBOARD_MAX_LINES: int = int(os.environ.get("DASHBOARD_MAX_LINES", "400"))

# --- P5 bounded-wait meter (OPEN_DESIGN P5, option A — the interim affordance before D25) ---
# The dashboard is blank for a median 28 s mid-answer. The meter renders "last line N s ago ·
# next due within <=M s" from the D19 tail plus a clock: no second Whisper, no second model,
# no new dependency, and NO SPEECH OF ITS OWN — it cannot be wrong about the interview.
METER_ENABLED: bool = os.environ.get("METER_ENABLED", "1") not in ("0", "false", "False")
# How often the server recomputes and pushes the meter. The math lives in Python
# (`dashboard.meter_state`, a pure tested function), NOT in the browser, so the one number the
# user reads under pressure has exactly one implementation. 0.25 s over a loopback websocket is
# ~4 messages/s and matches TRANSCRIPT_POLL_SECONDS.
METER_TICK_SECONDS: float = float(os.environ.get("METER_TICK_SECONDS", "0.25"))
# The ceiling the meter promises against. 0 = derive it as SEGMENT_MAX_SECONDS +
# METER_DECODE_ALLOWANCE_SECONDS. Set it explicitly only to keep a TIME-COMPRESSED replay
# truthful (`replay_transcript.py --speed N` wants ceiling/N).
METER_CEILING_SECONDS: float = float(os.environ.get("METER_CEILING_SECONDS", "0"))
# MEASURED 2026-09-04 on the real 42-min call, and the reason this knob exists at all: a line
# cannot reach the screen until its segment closes AND is decoded. Scoring the 109 arrival gaps
# against a bare SEGMENT_MAX_SECONDS=30 ceiling holds only 84/109 — because 54% of segments end
# exactly at the cap and then take D25's measured decode (0.248 + 0.0144*audio_s, max 0.96 s) on
# top, landing at 30.1-30.4 s. At 30.5 s the ceiling holds 108/109 (the one true overrun is 43.2 s,
# a silence between segments, which the ceiling does not bound and must not pretend to). 1.0 s
# clears the measured max decode with margin; it is an honesty allowance, not padding.
# DECIDED WITH A NUMBER 2026-09-05 (D27, #428): this stays **1.0 s**, and what it covers is the
# latency STEP between consecutive lines, not a decode time. The meter compares an arrival gap, and
#     gap = (end(N) - end(N-1)) + (latency(N) - latency(N-1))
# so a UNIFORM slowdown cancels; only the variation reaches the meter. Verified against the real
# call's own line-arrival wall clocks: residual to that model is mean +0.010 s, median -0.001 s.
# Measured false overdues (back-to-back gaps only — a gap across a silence is not this knob's to fix):
#   allowance   real call(108)   full-call replay(112)   forced collisions(51)   D26 off(51)
#     0.25 s          12                 10                      5                    4
#     0.50 s           3                  4                      1                    0
#     0.75 s           1*                 1                      0                    0
#     1.00 s           1*                 1                      0                    0   <- shipped
#     1.50 s           1*                 0                      0                    0
#     2.00 s           1*                 0                      0                    0
#   (*the real call's residual is a 2.0 s CAPTURE STALL, not a decode: line 29, gap 32.31 s,
#    spacing 30 s, decode 0.70 s vs 0.38 s. No allowance should hide it.)
# BOUNDS THAT BIND: do not go below 0.75 s (the knob starts biting), and do not go above 2.0 s (the
# ceiling then passes that stall). Do NOT widen it to cover the structural worst case — a D26-flagged
# 30 s segment (~2.34 s) behind a provisional (~0.77 s) after a fast short line (~0.30 s) is a ~2.8 s
# step, and covering that would blind the meter to the only real anomaly it caught.
# The residual at 1.0 s is 1 false overdue per ~112 back-to-back gaps, overdue by 0.193 s = 0.8 of one
# 0.25 s tick, and that one is a Whisper temperature-fallback outlier, not D26.
# Re-measure with: live_transcribe.py --latency-trace FILE  ->  scripts/meter_budget.py
METER_DECODE_ALLOWANCE_SECONDS: float = float(os.environ.get("METER_DECODE_ALLOWANCE_SECONDS", "1.0"))

# --- Dashboard honesty knobs (see the module docstring in scripts/dashboard.py) ---
# A finished suggestion answers the question it was fired on. Once this many NEW transcript
# lines have landed under it, the conversation has moved on and the panel says so instead of
# leaving a confident stale answer standing (worse than a blank panel).
SUGGESTION_STALE_LINES: int = int(os.environ.get("SUGGESTION_STALE_LINES", "2"))
# Read-only plan panel (P3/G7 is NOT settled — no auto-"covered" state machine is invented here).
# When on, a plan step is badged `mentioned` on a deterministic literal match of one of its
# `done_signals` against a transcript line. The badge means exactly that and the UI says so;
# it is never rendered as "covered".
PLAN_MENTION_TRACKING: bool = os.environ.get("PLAN_MENTION_TRACKING", "1") not in ("0", "false", "False")

# --- Salience gate (D23 — #386; runs AFTER the D20 question trigger, never instead of it) ---
# MEASURED against an in-sample fixture set (the 24 turns the D20
# trigger fired on during the real 42-min HR screen, 8 hand-labelled salient). Reproduce with
# `python scripts/reasoning.py --selftest-salience`:
#   backend                                        P     R    F1  fires  median  extra VRAM
#   embed  bge-m3, best threshold 0.436          0.38  1.00  0.55  21/24    120ms      664 MB
#   llm    llama3.2:3b, whole segment            0.67  0.50  0.57   6/24    134ms     2593 MB
#   llm    llama3.2:3b, question sentences       0.88  0.88  0.88   8/24    123ms     2593 MB
#   llm    llama3.1:8b, question sentences       0.62  1.00  0.76  13/24    142ms     5300 MB
#   llm    interview-copilot:14b, q-sentences    0.80  1.00  0.89  10/24    157ms         0  <- DEFAULT
# The cosine route the subtask row settled on was built and REJECTED on measurement, not on
# taste — see the module docstring in scripts/salience.py for the four variants and the two
# reasons (nomic-embed-text cannot read Polish; and topical relevance is the wrong question
# for an HR screen, where every turn is on-topic). `embed` stays available so that result is
# reproducible rather than merely asserted.
# WHY THE BIG MODEL JUDGES ITSELF: it is already resident for the suggestion, so the gate
# costs 0 MiB. Measured, a 3B judge beside the 14B leaves 2273 MiB for a Whisper that needs
# ~2200 — not a margin to take into a live interview.
SALIENCE_GATE_ENABLED: bool = os.environ.get("SALIENCE_GATE_ENABLED", "1") not in ("0", "false", "False")
# "llm" (one-word YES/NO, shipped) | "embed" (cosine vs bundle topics, measured + rejected) | "off"
SALIENCE_BACKEND: str = os.environ.get("SALIENCE_BACKEND", "llm")
# Empty means "whatever the local suggestion backend is already using" — that is the whole
# point of the default: no second model, no second VRAM claim, no second download.
SALIENCE_MODEL: str = os.environ.get("SALIENCE_MODEL", "")
# Embed backend only, and it is a MONUMENT, not a tuning knob. 0.436 is the best-F1 cut found by
# sweeping the whole fixture distribution, and "best" there means firing 21 of 24 — no gate at all.
# The number that settles it: salient turns average 0.510, non-salient 0.518. The cosine signal is
# not weak, it is ABSENT and very slightly inverted, so no threshold exists to be chosen.
SALIENCE_THRESHOLD: float = float(os.environ.get("SALIENCE_THRESHOLD", "0.436"))
# bge-m3 is multilingual (Polish 5/7 top-1 on clean probes); nomic-embed-text, the model the
# subtask row named, scored 1/7 on the same probes and 4/4 on their English translations.
SALIENCE_EMBED_MODEL: str = os.environ.get("SALIENCE_EMBED_MODEL", "bge-m3")
# The gate must be cheap relative to the ~5.9 s suggestion it protects. Measured median 157 ms
# / p90 194 ms, so 3 s is a hang-detector, not a working budget.
SALIENCE_TIMEOUT_SECONDS: float = float(os.environ.get("SALIENCE_TIMEOUT_SECONDS", "3.0"))
# A wasted call costs tokens; a suggestion the candidate needed and did not get costs the
# interview. So a broken gate fires everything (today's behaviour) rather than going silent.
SALIENCE_FAIL_OPEN: bool = os.environ.get("SALIENCE_FAIL_OPEN", "1") not in ("0", "false", "False")
