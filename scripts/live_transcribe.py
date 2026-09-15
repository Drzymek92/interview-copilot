"""Continuous LIVE transcript loop — capture a call, print it as it is spoken (subtask #320).

The one thing that must work tomorrow morning: start this before the call, watch
transcript lines appear, hit Ctrl-C afterwards, and be left with a recording and a
transcript that `score_transcript.py` can read without any reconstruction.

--------------------------------------------------------------------------------
QUICKSTART (09:55, nervous, one screen)
--------------------------------------------------------------------------------
    cd ~/Desktop/Claude_Projects/projects/interview_copilot
    python scripts/live_transcribe.py            # sources auto-picked; ~40 s model load

 1. Wait for the line `READY — listening. Speak or start the call. Ctrl-C to stop.`
    Nothing before that line is recorded, so do not start the call until you see it.
 2. Say a sentence into your mic. Within a few seconds a line appears:
    `[00:03-00:08] Dzień dobry, testuję nagrywanie.`  <- THAT is "it's working".
 3. If nothing appears: `python scripts/live_transcribe.py --list`, then pass the
    monitor of the sink Teams actually plays to: `--source <name>.monitor`.
 4. Tell HR you are recording (D11 — disclosed, no stealth). One line is enough:
    "Nagrywam rozmowę na własne potrzeby, żeby zrobić notatki — czy to w porządku?"
 5. Ctrl-C ONCE to stop. It finishes the pending decodes, then prints the paths of
    the WAV + transcript. Delete the raw WAV once you have scored it.

--------------------------------------------------------------------------------
WHAT IT DOES
--------------------------------------------------------------------------------
`parec` streams the sink monitor (the other person's voice, D17) and — unless
`--no-mic` — your microphone too, since the monitor does NOT contain your own
voice. Both are recorded to one stereo WAV (L = remote, R = mic) so speakers stay
separable later, and mixed to mono for Whisper.

Segmentation is **VAD-driven, never a fixed clock**: `webrtcvad` marks each 20 ms
frame speech/silence and a segment is closed by a PAUSE (see config/settings.py for
why — short fixed windows measurably cost accuracy on multi-word English terms).
The only fixed cut is the SEGMENT_MAX_SECONDS safety valve for a long monologue,
and that one carries audio over into the next segment, with the duplicated words
removed from the text again by `drop_overlap`.

`--segmentation fixed --fixed-seconds N` exists ONLY to reproduce the fixed-window
baseline for A/B measurement. It is not a supported live mode.

While a segment is still open the recorder ALSO decodes it every
`PARTIAL_DECODE_SECONDS` and appends the result to a SEPARATE
`scripts/outputs/live_transcript_<stamp>.partial` (D25 — the screen is otherwise blank
for a measured median 28 s mid-answer). Same line format, same tail, so a consumer reads
it with the same parser; the closing `.txt` line always supersedes it, and NOTHING
provisional is ever written to the `.txt` or to the plain scorer file. It is the
recorder's job because the cost is VRAM: a second Whisper in a consumer does not fit
beside the reasoning model. `--no-partials` turns it off.

Offline / no-human modes (these are how the loop is proven, not inspected):
    --from-wav FILE          push a WAV through the whole loop as fast as it decodes
    --selftest-sink SINK     play a WAV into SINK, capture SINK.monitor, auto-stop

SI1: capture, VAD and transcription are entirely local; nothing leaves the box.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings  # noqa: E402
from scripts.logger import get_logger  # noqa: E402
from scripts import audio_backend  # noqa: E402

logger = get_logger("live_transcribe")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "outputs"
RUNS_CSV = PROJECT_ROOT / "logs" / "runs.csv"


# --------------------------------------------------------------------------
# Capture backend — parec on Linux, sounddevice (PortAudio) on Windows/macOS.
# Everything below consumes fixed int16 mono 16 kHz frames; the backend produces them.
# --------------------------------------------------------------------------
BACKEND = audio_backend.select_backend(settings.AUDIO_BACKEND)


def source_names() -> list[str]:
    return BACKEND.source_names()


def default_monitor() -> str | None:
    return BACKEND.default_monitor()


def default_mic() -> str | None:
    return BACKEND.default_mic()


def list_sources() -> None:
    BACKEND.print_sources()


def mix(remote: bytes, mic: bytes | None) -> bytes:
    """Sum two mono int16 frames with clipping (Whisper gets one mono stream)."""
    if mic is None:
        return remote
    a = np.frombuffer(remote, dtype=np.int16).astype(np.int32)
    b = np.frombuffer(mic, dtype=np.int16).astype(np.int32)
    return np.clip(a + b, -32768, 32767).astype(np.int16).tobytes()


# --------------------------------------------------------------------------
# Speech detection
# --------------------------------------------------------------------------
class ChannelGate:
    """webrtcvad plus an adaptive noise floor, for ONE channel.

    webrtcvad on its own labels steady microphone hiss as speech. Measured on a
    live two-stream run (2026-08-31): the webcam mic's noise never let the VAD see
    silence, so every segment ran to the 30 s safety cap and decoded to garbage
    ("...to have worked on...", "KONIEC!", ""). A frame therefore also has to be
    meaningfully louder than this channel's OWN recent noise floor. Per channel,
    because a monitor and a mic sit at very different levels.
    """

    def __init__(self, vad: object, label: str) -> None:
        self.vad = vad
        self.label = label
        frames_per_window = int(settings.VAD_NOISE_WINDOW_SECONDS / (settings.VAD_FRAME_MS / 1000.0))
        self.history: deque = deque(maxlen=max(50, frames_per_window))
        self.floor = 0.0

    def is_speech(self, frame: bytes) -> bool:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        self.history.append(rms)
        # Below a full window the floor is not trustworthy yet — fall back to the
        # absolute minimum so the loop still works in its first seconds.
        self.floor = float(np.percentile(self.history, 10)) if len(self.history) >= 50 else 0.0
        threshold = max(settings.VAD_SPEECH_MIN_RMS, self.floor * settings.VAD_SPEECH_RMS_MULT)
        if rms < threshold:
            return False
        return bool(self.vad.is_speech(frame, settings.SAMPLE_RATE))  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
@dataclass
class Segment:
    index: int
    start: float
    end: float
    pcm: bytes
    speech_seconds: float
    continued: bool = False  # produced by a max-length force-cut -> text may overlap
    speaker: str = "them"  # "them" (monitor/interviewer) | "you" (mic/candidate), G9/#326
    provisional: bool = False  # a snapshot of a STILL-OPEN segment (D25) -> .partial, never .txt
    # Wall (monotonic) time the capture loop handed this segment to the worker. Set on FINAL
    # segments only; it is what makes "how long did a final wait behind a provisional" a
    # measurement rather than an estimate (D25's cost, OPEN_DESIGN P5).
    queued_at: float = 0.0


@dataclass
class VadSegmenter:
    """Close a segment on a PAUSE, not on a clock (the measured constraint)."""

    frame_seconds: float
    frames: list[bytes] = field(default_factory=list)
    preroll: deque = field(default_factory=deque)
    in_speech: bool = False
    speech_frames: int = 0
    silence_run: int = 0
    start_time: float = 0.0
    index: int = 0
    carry: list[bytes] = field(default_factory=list)
    continued: bool = False
    # Per-channel speech-frame tally for the open segment. The monitor carries only the
    # interviewer and the mic only the candidate, so whichever channel spoke more decides
    # the speaker tag (G9/#326). Reset on every _emit.
    remote_votes: int = 0
    mic_votes: int = 0

    def __post_init__(self) -> None:
        self.preroll = deque(maxlen=max(1, int(settings.SEGMENT_PREROLL_SECONDS / self.frame_seconds)))

    def push(self, frame: bytes, remote_speech: bool, mic_speech: bool, now: float) -> Segment | None:
        is_speech = remote_speech or mic_speech
        if not self.in_speech:
            self.preroll.append(frame)
            if not is_speech:
                return None
            # Speech onset: open a segment with the pre-roll (and any carried-over
            # audio from a force-cut) so it never starts mid-word.
            self.in_speech = True
            self.frames = [*self.carry, *self.preroll]
            self.start_time = now - len(self.frames) * self.frame_seconds
            self.preroll.clear()
            self.carry = []
            self.speech_frames = 1
            self.silence_run = 0
            self.remote_votes = 1 if remote_speech else 0
            self.mic_votes = 1 if mic_speech else 0
            return None

        self.frames.append(frame)
        if is_speech:
            self.speech_frames += 1
            self.silence_run = 0
            self.remote_votes += 1 if remote_speech else 0
            self.mic_votes += 1 if mic_speech else 0
        else:
            self.silence_run += 1

        duration = len(self.frames) * self.frame_seconds
        silence = self.silence_run * self.frame_seconds

        if silence >= settings.SEGMENT_SILENCE_SECONDS and duration >= settings.SEGMENT_MIN_SECONDS:
            return self._emit(now)
        if silence >= settings.SEGMENT_MAX_SILENCE_SECONDS:
            return self._emit(now)
        if duration >= settings.SEGMENT_MAX_SECONDS:
            return self._emit(now, force_cut=True)
        return None

    def _emit(self, now: float, force_cut: bool = False) -> Segment | None:
        frames, continued = self.frames, self.continued
        speech_seconds = self.speech_frames * self.frame_seconds
        # Dominant channel decides the speaker. Tie (or a monitor-only run where mic_votes
        # never moves) falls to "them", the side that must never be missed.
        speaker = "you" if self.mic_votes > self.remote_votes else "them"
        self.frames, self.in_speech, self.speech_frames, self.silence_run = [], False, 0, 0
        self.remote_votes = self.mic_votes = 0
        self.preroll.clear()

        if force_cut:
            # Mid-speech cut: carry the tail into the next segment so a bisected
            # phrase survives whole somewhere, and stay "in speech" via the carry.
            n_carry = int(settings.SEGMENT_CARRYOVER_SECONDS / self.frame_seconds)
            self.carry = frames[-n_carry:] if n_carry else []
            self.continued = True
        else:
            self.carry, self.continued = [], False

        if speech_seconds < settings.SEGMENT_MIN_SPEECH_SECONDS:
            return None
        self.index += 1
        return Segment(
            index=self.index,
            start=self.start_time,
            end=now,
            pcm=b"".join(frames),
            speech_seconds=speech_seconds,
            continued=continued,
            speaker=speaker,
        )

    def flush(self, now: float) -> Segment | None:
        return self._emit(now) if self.in_speech and self.frames else None

    # -- D25: read-only views of the segment that is still open ------------
    def open_start(self) -> float | None:
        """Start time of the segment currently open, or None when nobody is speaking."""
        return self.start_time if (self.in_speech and self.frames) else None

    def snapshot(self, now: float) -> Segment | None:
        """A PROVISIONAL copy of the still-open segment (D25) — never mutates state.

        This is a growing prefix anchored at a true speech onset, not a fixed-clock
        window: it truncates only its right edge, which is why D21's fixed-chunking
        catastrophe (which cuts BOTH edges, so a window opens mid-phrase) does not
        transfer to it. The segmenter is left exactly as it was found, so the closing
        `_emit` produces the identical final segment whether or not anyone snapshotted.
        """
        if not self.in_speech or not self.frames:
            return None
        speech_seconds = self.speech_frames * self.frame_seconds
        if speech_seconds < settings.SEGMENT_MIN_SPEECH_SECONDS:
            return None
        speaker = "you" if self.mic_votes > self.remote_votes else "them"
        return Segment(
            # The index `_emit` WILL hand this segment — advisory only. Consumers join a
            # provisional to its final line on the START stamp, which cannot drift.
            index=self.index + 1,
            start=self.start_time,
            end=now,
            pcm=b"".join(self.frames),
            speech_seconds=speech_seconds,
            continued=self.continued,
            speaker=speaker,
            provisional=True,
        )


@dataclass
class FixedSegmenter:
    """Fixed-clock windows — A/B COMPARISON ONLY, not a live mode (see module docstring)."""

    frame_seconds: float
    window_seconds: float
    overlap_seconds: float = 0.0
    frames: list[bytes] = field(default_factory=list)
    start_time: float = 0.0
    index: int = 0
    started: bool = False

    def push(self, frame: bytes, remote_speech: bool, mic_speech: bool, now: float) -> Segment | None:
        del remote_speech, mic_speech  # fixed windows ignore voice activity — that is the point
        if not self.started:
            self.start_time, self.started = now - self.frame_seconds, True
        self.frames.append(frame)
        if len(self.frames) * self.frame_seconds < self.window_seconds:
            return None
        return self._emit(now, keep_overlap=True)

    def _emit(self, now: float, keep_overlap: bool) -> Segment | None:
        if not self.frames:
            return None
        frames = self.frames
        continued = self.index > 0 and self.overlap_seconds > 0
        n_overlap = int(self.overlap_seconds / self.frame_seconds) if keep_overlap else 0
        self.frames = frames[-n_overlap:] if n_overlap else []
        self.start_time = now - len(self.frames) * self.frame_seconds
        self.index += 1
        return Segment(
            index=self.index,
            start=now - len(frames) * self.frame_seconds,
            end=now,
            pcm=b"".join(frames),
            speech_seconds=len(frames) * self.frame_seconds,
            continued=continued,
        )

    def flush(self, now: float) -> Segment | None:
        return self._emit(now, keep_overlap=False)

    def open_start(self) -> float | None:
        return None  # D25 provisional lines are a live-mode affordance, not part of the A/B harness

    def snapshot(self, now: float) -> Segment | None:
        del now
        return None


def drop_overlap(previous: str, text: str, max_words: int = 24) -> str:
    """Remove the longest word-run that `text` repeats from the tail of `previous`.

    Only used on segments that continue a force-cut, where the audio genuinely
    overlaps; applying it blindly would eat legitimate repetition.
    """
    prev_words, new_words = previous.split(), text.split()
    if not prev_words or not new_words:
        return text

    def key(word: str) -> str:
        return "".join(c for c in word.lower() if c.isalnum())

    prev_keys = [key(w) for w in prev_words[-max_words:]]
    new_keys = [key(w) for w in new_words[:max_words]]
    for n in range(min(len(prev_keys), len(new_keys)), 1, -1):
        if prev_keys[-n:] == new_keys[:n]:
            return " ".join(new_words[n:]).strip()
    return text


# --------------------------------------------------------------------------
# Provisional lines (D25) — cadence + handoff
# --------------------------------------------------------------------------
# How long the decode worker waits on a FINAL segment before it looks for a provisional
# snapshot to fill the idle time. Deliberately NOT a settings knob: D25 names two knobs
# (PARTIAL_DECODE_ENABLED, PARTIAL_DECODE_SECONDS) and this is the worker's poll interval,
# not a behaviour anyone tunes. It only costs a wakeup 20x/s while the GPU is idle.
PARTIAL_POLL_SECONDS = 0.05


@dataclass
class PartialClock:
    """Decides WHEN the open segment is due for a provisional decode (D25).

    A pure function of (segment start, now) plus the last fire, so the cadence is testable
    without audio, a model or a clock. The first provisional for a segment lands one
    interval after the segment OPENED, not one interval after the previous segment's, so a
    long pause never causes a burst of catch-up decodes.
    """

    interval: float
    segment_start: float | None = None
    last_fire: float = 0.0

    def due(self, start: float | None, now: float) -> bool:
        if start is None:            # nobody is speaking — nothing to be provisional about
            self.segment_start = None
            return False
        if start != self.segment_start:
            self.segment_start, self.last_fire = start, start
        if now - self.last_fire < self.interval:
            return False
        self.last_fire = now
        return True


class PartialMailbox:
    """A one-slot, newest-wins handoff of provisional snapshots to the decode worker.

    A queue is the wrong shape here: provisionals that pile up behind a busy GPU decode to
    text that is already stale by the time it is written, and each one delays the FINAL
    line behind it. Only the newest prefix is worth decoding, so a new snapshot replaces an
    undecoded older one and the displacement is counted (`dropped`) rather than hidden.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: Segment | None = None
        self.dropped = 0

    def put(self, seg: Segment) -> None:
        with self._lock:
            if self._item is not None:
                self.dropped += 1
            self._item = seg

    def take(self) -> Segment | None:
        with self._lock:
            item, self._item = self._item, None
            return item


@dataclass
class CaptureDriftMonitor:
    """Tracks how far the AUDIO clock falls behind WALL clock — the signature of a stall.

    The capture loop derives audio time as `frames_seen * frame_seconds`: it only advances
    when a frame is actually read from parec/PipeWire. Wall time (`time.monotonic() -
    feed_started`) advances regardless. When capture blocks, wall time keeps moving while the
    audio clock does not, so the divergence between the two is a direct, measurable symptom
    of the stall — the only thing that has ever breached the meter's 31.0 s ceiling on the
    real call (#428/D27: a 2.0 s capture stall, gap 32.31 s against a 30 s spacing). This is a
    pure function of successive `(wall_elapsed, audio_elapsed)` pairs, so it needs no audio,
    no GPU and no model to test — feed it samples and read `max_drift` / `total_drift` back.

    Sign convention: `drift = wall_elapsed - audio_elapsed`. POSITIVE means the audio clock is
    BEHIND wall clock, i.e. a stall. A healthy capture cannot make audio time run ahead of
    wall time (frames cannot be read before the real time they represent has elapsed), so a
    negative reading is not a stall and must not cancel one out later: `total_drift` only
    accumulates the GROWTH in drift (the size of each stall as it happens), never a shrink.
    That also makes an unpaced `--from-wav` replay safe to sample: audio time deliberately
    races far AHEAD of wall time there (by design — see the pacing comment above), which
    reads as a large, ever-more-negative drift, and correctly reports max/total drift of 0.0
    rather than a meaningless negative number.
    """

    max_drift: float = 0.0
    total_drift: float = 0.0
    _last_drift: float = field(default=0.0, repr=False)

    def sample(self, wall_elapsed: float, audio_elapsed: float) -> float:
        drift = wall_elapsed - audio_elapsed
        growth = drift - self._last_drift
        if growth > 0:
            self.total_drift += growth
        if drift > self.max_drift:
            self.max_drift = drift
        self._last_drift = drift
        return drift


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def mmss(seconds: float) -> str:
    seconds = max(0.0, seconds)
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


@dataclass
class RunOutputs:
    """The files a run leaves behind, all sharing one timestamp (TRK).

    The `.partial` file (D25) is the only one that may be absent: it exists when provisional
    lines are on. It is written by `write_partial` and by nothing else — the transcript and
    the plain scorer file cannot receive a provisional word, which is D19's contract holding.
    """

    stamp: str
    run_id: str
    wav_path: Path | None
    transcript_path: Path
    plain_path: Path
    wav: wave.Wave_write | None
    transcript: object
    plain: object
    lines: int = 0
    partial_path: Path | None = None
    partial: object | None = None
    partial_lines: int = 0
    # (segment start, text) of the last provisional written — repeats are not rewritten.
    last_partial: tuple[float, str] | None = None
    # Start time of the newest segment that already has a FINAL line. Anything provisional
    # at or before it has been superseded and must not be written (see `superseded`).
    last_final_start: float = -1.0

    def write_line(self, seg: Segment, text: str, wall: str, language: str = "") -> None:
        self.lines += 1
        # The speaker tag (G9/#326) and the detected language (per-question answer language)
        # go on the HUMAN transcript, which the reasoning layer parses. The _plain_ scorer
        # file stays text-only so WER scoring is unaffected.
        lang = f" ({language})" if language else ""
        line = f"[{mmss(seg.start)}-{mmss(seg.end)}] {seg.speaker}{lang}: {text}"
        print(line, flush=True)
        self.transcript.write(f"{line}\n")  # type: ignore[attr-defined]
        self.transcript.flush()  # type: ignore[attr-defined]
        os.fsync(self.transcript.fileno())  # type: ignore[attr-defined]
        self.plain.write(f"{text}\n")  # type: ignore[attr-defined]
        self.plain.flush()  # type: ignore[attr-defined]
        os.fsync(self.plain.fileno())  # type: ignore[attr-defined]
        self.last_final_start = seg.start
        logger.info("%s | segment %d | %s%s | %s", wall, seg.index, seg.speaker, lang, text)

    def superseded(self, seg: Segment) -> bool:
        """Has this segment's FINAL line already been written?

        MEASURED, not anticipated: on the 2026-09-05 real-time self-test one provisional landed
        **0.39 s after** the final line for its own segment — and carried a wrong last word
        ("terenie" where the final said "TELU"). Finals having GPU priority is exactly what makes
        this possible: a snapshot taken at 02:18 waits while the segment closes at 02:19 and its
        final is decoded first. A provisional that arrives after the text it was provisional
        ABOUT is pure noise, so it is dropped before it costs a decode or reaches the file.
        Segment starts only increase, so `<=` also covers a straggler from an older segment.
        """
        return seg.start <= self.last_final_start

    def write_partial(self, seg: Segment, text: str, language: str = "") -> bool:
        """Append a PROVISIONAL line for a still-open segment (D25). Returns whether it wrote.

        Deliberately NOT printed to the console: the terminal shows the transcript, and a
        provisional line racing the final one for the same segment on the same stream is how
        an operator ends up reading text that was already superseded. The `.partial` file is
        the seam; a consumer that wants provisional text tails it.

        The same line is not written twice for the same segment. A re-decode of a prefix that
        has not grown produces the same words, and a tailing consumer should see an update
        only when there IS one; liveness is the meter's job (P5-A), not this file's.
        """
        if self.partial is None or not text or self.superseded(seg):
            return False
        key = (round(seg.start, 3), text)
        if key == self.last_partial:
            return False
        self.last_partial = key
        self.partial_lines += 1
        lang = f" ({language})" if language else ""
        line = f"[{mmss(seg.start)}-{mmss(seg.end)}] {seg.speaker}{lang}: {text}"
        self.partial.write(f"{line}\n")  # type: ignore[attr-defined]
        self.partial.flush()  # type: ignore[attr-defined]
        os.fsync(self.partial.fileno())  # type: ignore[attr-defined]
        logger.info("provisional | %s", line)
        return True

    def close(self) -> None:
        for handle in (self.transcript, self.plain, self.partial):
            if handle is None:
                continue
            try:
                handle.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 — closing must never mask the run's result
                logger.exception("failed closing a transcript handle")
        if self.wav is not None:
            self.wav.close()


PARTIAL_HEADER = (
    "# interview_copilot PROVISIONAL lines (D25) — this is NOT the transcript.\n"
    "# Every line here is a decode of a segment that was still OPEN. The line for the same\n"
    "# segment in live_transcript_<stamp>.txt ALWAYS supersedes it.\n"
    "# Join key: the START stamp. A provisional [mm:ss-mm:ss] shares its start with the final\n"
    "# line of its segment; the end stamp is 'as of this decode' and grows as the segment does.\n"
    "# The line format is identical to the .txt, so the same tail and the same parser read both\n"
    "# (D19). A provisional may have NO final line at all — its segment can still be dropped as\n"
    "# silence, or decode to nothing.\n"
    "# D11: derived from the same disclosed recording. Delete it alongside the WAV.\n"
)


def open_outputs(
    stamp: str, run_id: str, record: bool, channels: int, header: str, partials: bool = False
) -> RunOutputs:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = OUTPUT_DIR / f"live_audio_{stamp}.wav" if record else None
    transcript_path = OUTPUT_DIR / f"live_transcript_{stamp}.txt"
    plain_path = OUTPUT_DIR / f"live_transcript_plain_{stamp}.txt"

    wav_writer: wave.Wave_write | None = None
    if wav_path is not None:
        # Written frame-by-frame rather than temp-then-rename: this is a recorder, and
        # a half-recorded call must survive a crash. wave.close() fixes up the header.
        wav_writer = wave.open(str(wav_path), "wb")
        wav_writer.setnchannels(channels)
        wav_writer.setsampwidth(2)
        wav_writer.setframerate(settings.SAMPLE_RATE)

    transcript = open(transcript_path, "w", encoding="utf-8")
    transcript.write(header)
    transcript.flush()
    plain = open(plain_path, "w", encoding="utf-8")  # scorer input: text only, no metadata

    # A provisional line is a convenience; the recording is not. If the .partial cannot be
    # opened, the run continues without it rather than failing on the one file nobody needs.
    partial_path: Path | None = None
    partial_handle: object | None = None
    if partials:
        try:
            partial_path = OUTPUT_DIR / f"live_transcript_{stamp}.partial"
            partial_handle = open(partial_path, "w", encoding="utf-8")
            partial_handle.write(PARTIAL_HEADER)
            partial_handle.flush()
        except OSError:
            logger.exception("could not open the .partial file — continuing without provisional lines")
            partial_path, partial_handle = None, None

    return RunOutputs(
        stamp, run_id, wav_path, transcript_path, plain_path, wav_writer, transcript, plain,
        partial_path=partial_path, partial=partial_handle,
    )


def append_run_row(run_id: str, start_iso: str, end_iso: str, status: str, audio_seconds: float, out: RunOutputs) -> None:
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    paths = ";".join(
        str(p.name)
        for p in (out.wav_path, out.transcript_path, out.plain_path, out.partial_path)
        if p
    )
    row = [run_id, "live_transcribe.py", start_iso, end_iso, status, f"{audio_seconds:.1f}", str(out.lines), paths]
    header = ["run_id", "script", "start_iso", "end_iso", "status", "in_count", "out_count", "output_paths"]
    existing = RUNS_CSV.read_text(encoding="utf-8") if RUNS_CSV.exists() else ""
    tmp = RUNS_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if not existing:
            writer.writerow(header)
        fh.write(existing)
        writer.writerow(row)
    tmp.replace(RUNS_CSV)


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------
def decode_segment(seg: Segment, transcriber: object) -> object | None:
    """Decode one segment on the loaded model. None = dropped as silence, or decode failed.

    Shared by the final and the provisional path, so a provisional cannot end up with a
    different silence floor or a different error posture than the line that supersedes it.
    It never raises: one bad decode must not end the call, and a bad PROVISIONAL decode must
    not cost the call a final line.
    """
    audio = np.frombuffer(seg.pcm, dtype=np.int16).astype(np.float32) / 32768.0
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    kind = "provisional" if seg.provisional else "segment"
    if peak < settings.SEGMENT_MIN_PEAK:
        logger.info("%s %d dropped: peak %.5f below floor (silence)", kind, seg.index, peak)
        return None
    try:
        return transcriber.transcribe_array(audio, sample_rate=settings.SAMPLE_RATE)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — one bad decode must not end the call
        logger.exception("%s %d failed to decode — continuing", kind, seg.index)
        return None


def decode_provisional(
    snap: Segment, transcriber: object, out: RunOutputs, dedupe: bool, previous: str, stats: dict
) -> None:
    """Decode one provisional snapshot and write it to the `.partial` file (D25).

    Every exit path is accounted for in `stats` so the run can report what the provisional
    path actually cost, rather than what the D25 model predicts it costs.
    """
    if out.superseded(snap):
        # Its final line beat it to the file — do not spend a decode on text nobody can use.
        stats["partial_superseded"] += 1
        logger.info("provisional for segment starting %.2fs dropped: its final line already landed",
                    snap.start)
        return
    result = decode_segment(snap, transcriber)
    if result is None:
        return
    stats["partial_decoded"] += 1
    stats["partial_decode_seconds"] += result.latency_seconds
    text = result.text.strip()
    if dedupe and snap.continued:
        # De-duplicated against the last FINAL text; a provisional never becomes the
        # baseline the next final is de-duplicated against.
        text = drop_overlap(previous, text)
    if out.write_partial(snap, text, language=result.language):
        stats["partial_written"] += 1


def new_stats() -> dict:
    """The counters a run accumulates. One definition, so a caller cannot hand the worker a
    dict missing a key the worker writes (which is a KeyError inside a daemon thread — i.e.
    a silently dead decoder in the middle of an interview)."""
    return {
        "decoded": 0, "decode_seconds": 0.0,
        "partial_decoded": 0, "partial_decode_seconds": 0.0, "partial_written": 0,
        "partial_superseded": 0,
        # How long finals waited in the queue, and how much of that a provisional explains.
        "final_waits": 0, "final_wait_max": 0.0,
        "final_behind_partial": 0, "final_behind_partial_max": 0.0,
        "final_behind_partial_seconds": 0.0,
    }


class LatencyTrace:
    """One JSONL row per FINAL segment: what the P5 meter's allowance has to cover (D27/#428).

    The meter measures the wall-clock gap between LINE ARRIVALS, so the quantity that decides
    `METER_DECODE_ALLOWANCE_SECONDS` is not the decode time — it is **close -> write**: from the
    instant the capture loop hands a closed segment to the worker, to the instant its line is on
    disk. That spans the queue wait (including any in-flight provisional, D25), the decode
    (including D26's language probes), and the write itself. Nothing else in the run summary
    measures it end to end.

    Segments that produce NO line are recorded too, with `dropped: true` — they lengthen the gap
    to the next arrival, which is exactly what the meter feels.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fh = path.open("w", encoding="utf-8")
        self.rows = 0

    def record(
        self, seg: Segment, result: object | None, queue_wait: float, behind_partial: float,
        dropped: bool,
    ) -> None:
        now = time.monotonic()
        row = {
            "index": seg.index,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "duration": round(seg.end - seg.start, 3),
            "speaker": seg.speaker,
            "continued": bool(seg.continued),
            "dropped": bool(dropped),
            "queued_at": round(seg.queued_at, 4),
            "written_at": round(now, 4),
            "close_to_write": round(now - seg.queued_at, 4) if seg.queued_at else None,
            "queue_wait": round(queue_wait, 4),
            "behind_partial": round(behind_partial, 4),
            "decode_seconds": round(getattr(result, "latency_seconds", 0.0), 4),
            "language": getattr(result, "language", ""),
            "code_switch": bool(getattr(result, "code_switch", False)),
            "decode_passes": int(getattr(result, "decode_passes", 0)),
        }
        self.fh.write(json.dumps(row) + "\n")
        self.fh.flush()
        self.rows += 1

    def close(self) -> None:
        self.fh.close()


def transcribe_worker(
    work: queue.Queue, transcriber: object, out: RunOutputs, dedupe: bool, stats: dict,
    mailbox: PartialMailbox | None = None, trace: "LatencyTrace | None" = None,
) -> None:
    """Decode segments off the queue so capture never blocks on the GPU.

    With a `mailbox` (D25) this same worker — and therefore the same single Whisper, which is
    the whole point of the decision — also decodes provisional snapshots, but only in the
    gaps: a FINAL segment is taken first every time, and a provisional is picked up only when
    the queue is empty at that instant. That ordering is why a provisional cannot push a final
    past the P5 meter's 31.0 s ceiling by queueing ahead of it. The one cost a final can still
    pay is sitting behind ONE provisional decode already in flight (0.248 + 0.0144*audio_s,
    measured max 0.96 s on the real call), which the ceiling's 1.0 s allowance does not cover
    twice — so it is stated in OPEN_DESIGN P5 rather than assumed away.
    """
    previous = ""
    # Monotonic time the most recent provisional decode RETURNED. A final enqueued before
    # that instant was, by definition, sitting behind a decode already in flight — the one
    # cost the final-first ordering cannot remove, and Done-when 3's second number.
    partial_done_at = 0.0
    while True:
        if mailbox is None:
            seg = work.get()
        else:
            try:
                seg = work.get(timeout=PARTIAL_POLL_SECONDS)
            except queue.Empty:
                snap = mailbox.take()
                if snap is not None:
                    decode_provisional(snap, transcriber, out, dedupe, previous, stats)
                    partial_done_at = time.monotonic()
                continue
        if seg is None:
            return
        wait = behind = 0.0
        if seg.queued_at:
            wait = time.monotonic() - seg.queued_at
            stats["final_waits"] += 1
            stats["final_wait_max"] = max(stats["final_wait_max"], wait)
            # Attribute only the part of the wait that a provisional can explain: the
            # overhang of an in-flight provisional decode past this final's enqueue.
            behind = max(0.0, min(wait, partial_done_at - seg.queued_at))
            if behind > 0:
                stats["final_behind_partial"] += 1
                stats["final_behind_partial_max"] = max(
                    stats["final_behind_partial_max"], behind
                )
                stats["final_behind_partial_seconds"] += behind
        result = decode_segment(seg, transcriber)
        if result is None:
            if trace is not None:
                trace.record(seg, None, wait, behind, dropped=True)
            continue
        stats["decoded"] += 1
        stats["decode_seconds"] += result.latency_seconds
        text = result.text.strip()
        if dedupe and seg.continued:
            text = drop_overlap(previous, text)
        if not text:
            if trace is not None:
                trace.record(seg, result, wait, behind, dropped=True)
            continue
        previous = text
        out.write_line(seg, text, datetime.now().strftime("%H:%M:%S"), language=result.language)
        # D27/#428: the meter's whole budget is this number — the wall time from a segment CLOSING
        # to its line being on disk. Recorded here, after the write, because the meter measures
        # arrivals and not decodes.
        if trace is not None:
            trace.record(seg, result, wait, behind, dropped=False)


def run_loop(args: argparse.Namespace) -> int:
    import webrtcvad

    from scripts.stt import Transcriber

    frame_seconds = settings.VAD_FRAME_MS / 1000.0
    frame_bytes = int(settings.SAMPLE_RATE * frame_seconds) * 2

    # --- resolve inputs -------------------------------------------------
    wav_input: np.ndarray | None = None
    remote_name = mic_name = None
    if args.from_wav:
        from faster_whisper.audio import decode_audio

        wav_input = decode_audio(args.from_wav, sampling_rate=settings.SAMPLE_RATE)
        remote_name = f"file:{Path(args.from_wav).name}"
    else:
        ok, hint = BACKEND.available()
        if not ok:
            logger.error(hint)
            return 2
        remote_name = args.source if args.source != "auto" else default_monitor()
        if not remote_name:
            logger.error("could not resolve a monitor / system-audio source — run --list and pass --source")
            return 2
        if BACKEND.name == "parec" and remote_name not in source_names():
            logger.error("source %r is not in `pactl list short sources` — run --list", remote_name)
            return 2
        if not args.no_mic:
            mic_name = args.mic if args.mic != "auto" else default_mic()
            if not mic_name:
                logger.warning("no microphone resolved — recording the monitor only (your voice will be missing)")

    # --- model FIRST, so the first minute of the call is not eaten by a load ---
    logger.info("loading STT model before capture starts (once, outside the loop)")
    transcriber = Transcriber()

    gate_remote = ChannelGate(webrtcvad.Vad(settings.VAD_AGGRESSIVENESS), "remote")
    gate_mic = ChannelGate(webrtcvad.Vad(settings.VAD_AGGRESSIVENESS), "mic")

    segmenter: VadSegmenter | FixedSegmenter
    if args.segmentation == "vad":
        segmenter = VadSegmenter(frame_seconds=frame_seconds)
        policy = (
            f"VAD(aggr={settings.VAD_AGGRESSIVENESS}) pause>={settings.SEGMENT_SILENCE_SECONDS}s "
            f"min={settings.SEGMENT_MIN_SECONDS}s max={settings.SEGMENT_MAX_SECONDS}s"
        )
    else:
        segmenter = FixedSegmenter(
            frame_seconds=frame_seconds,
            window_seconds=args.fixed_seconds,
            overlap_seconds=args.overlap_seconds,
        )
        policy = f"FIXED {args.fixed_seconds}s overlap={args.overlap_seconds}s (comparison mode)"

    # D25 provisional lines. CLI > env > config (CFG). Never in `fixed` mode: that segmenter
    # is the A/B harness, and a provisional prefix of a fixed window measures nothing.
    pace = max(0.0, getattr(args, "pace", 0.0))
    partials_on = settings.PARTIAL_DECODE_ENABLED if args.partials is None else args.partials
    partials_on = partials_on and args.segmentation == "vad"
    partial_seconds = args.partial_seconds or settings.PARTIAL_DECODE_SECONDS
    partial_policy = f"every {partial_seconds:g}s -> .partial (D25)" if partials_on else "off"

    run_id = f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    start_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    channels = 2 if mic_name else 1
    header = (
        f"# interview_copilot live transcript\n"
        f"# run_id     : {run_id}\n"
        f"# started    : {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"# remote src : {remote_name}\n"
        f"# mic src    : {mic_name or '(none)'}\n"
        f"# segmentation: {policy}\n"
        f"# provisional : {partial_policy}\n"
        f"# model      : {settings.STT_MODEL} lang={settings.STT_LANGUAGE} beam={settings.STT_BEAM_SIZE}\n"
        f"# D11: recording disclosed to the other party. Delete the WAV after scoring.\n"
        f"# (timestamps are mm:ss from run start; the _plain_ file is the scorer input)\n"
    )
    out = open_outputs(
        stamp, run_id, record=not args.no_record, channels=channels, header=header,
        partials=partials_on,
    )
    partials_on = partials_on and out.partial is not None  # the .partial may have failed to open
    logger.info("run %s START | source=%s mic=%s | %s | provisional=%s",
                run_id, remote_name, mic_name, policy, partial_policy)
    logger.info("outputs: %s | %s | %s | %s",
                out.wav_path, out.transcript_path, out.plain_path, out.partial_path)

    stop = threading.Event()

    def on_sigint(signum: int, frame: object) -> None:
        if stop.is_set():  # a second Ctrl-C means "now"
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt
        stop.set()
        print("\n[stopping — finishing pending decodes; Ctrl-C again to force]", flush=True)

    signal.signal(signal.SIGINT, on_sigint)

    stats = new_stats()
    work: queue.Queue = queue.Queue()
    mailbox = PartialMailbox() if partials_on else None
    partial_clock = PartialClock(interval=partial_seconds)
    trace = LatencyTrace(Path(args.latency_trace)) if getattr(args, "latency_trace", "") else None
    if trace is not None:
        logger.info("latency trace -> %s (#428)", trace.path)
    worker = threading.Thread(
        target=transcribe_worker,
        args=(work, transcriber, out, args.dedupe, stats, mailbox, trace),
        daemon=True,
    )
    worker.start()

    streams: list[audio_backend.CaptureStream] = []
    player: subprocess.Popen | None = None
    frames_seen = 0
    status = "ok"
    drift_monitor = CaptureDriftMonitor()
    try:
        if args.selftest_sink:
            player = BACKEND.play_into_sink(args.selftest_sink, args.selftest_wav)
            if player is not None:
                logger.info("self-test: playing %s into sink %s", args.selftest_wav, args.selftest_sink)

        if wav_input is None:
            streams.append(BACKEND.open_stream(remote_name, frame_bytes, "remote"))
            if mic_name:
                streams.append(BACKEND.open_stream(mic_name, frame_bytes, "mic"))

        # Show the resolved sources: `auto` picks the DEFAULT input, which on this box
        # is the webcam mic, not your external mic. Check this line before the call starts.
        print(f"\n  them (monitor) : {remote_name}")
        print(f"  you   (mic)    : {mic_name or '(none — your own voice will NOT be recorded)'}")
        print(f"  segmentation   : {policy}")
        print("\nREADY — listening. Speak or start the call. Ctrl-C to stop.\n", flush=True)

        wav_cursor = 0
        drain_deadline: float | None = None
        feed_started = time.monotonic()
        while not stop.is_set():
            if wav_input is not None:
                chunk = wav_input[wav_cursor : wav_cursor + frame_bytes // 2]
                wav_cursor += frame_bytes // 2
                if len(chunk) < frame_bytes // 2:
                    break
                remote = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                mic_frame = None
                if pace > 0:
                    # A live call delivers audio at 1x, and the provisional cadence is a race
                    # between audio time and the GPU: unpaced, a replay outruns the decoder by
                    # ~45x, every snapshot is displaced by a fresher one and NO provisional line
                    # is ever written. Pacing is what makes --from-wav able to exercise D25 at
                    # all; it changes nothing about what the loop decides, only when.
                    behind = (wav_cursor / settings.SAMPLE_RATE) / pace - (
                        time.monotonic() - feed_started
                    )
                    if behind > 0:
                        time.sleep(behind)
            else:
                remote = streams[0].read()
                mic_frame = streams[1].read() if len(streams) > 1 else None
                if all(s.dead for s in streams):
                    logger.error("every capture stream died — stopping")
                    status = "streams_died"
                    break

            frames_seen += 1
            now = frames_seen * frame_seconds
            drift_monitor.sample(time.monotonic() - feed_started, now)

            if out.wav is not None:
                if mic_frame is not None:
                    stereo = np.empty(frame_bytes, dtype=np.int16)
                    stereo[0::2] = np.frombuffer(remote, dtype=np.int16)
                    stereo[1::2] = np.frombuffer(mic_frame, dtype=np.int16)
                    out.wav.writeframes(stereo.tobytes())
                else:
                    out.wav.writeframes(remote)

            # Either party talking opens/holds a segment; each channel is judged
            # against its own noise floor so a hissy mic cannot pin the loop open.
            # Both gates run every frame (not short-circuited) so the per-channel vote
            # that decides the speaker tag is complete, and the mic floor stays current.
            remote_speech = gate_remote.is_speech(remote)
            mic_speech = gate_mic.is_speech(mic_frame) if mic_frame is not None else False

            seg = segmenter.push(mix(remote, mic_frame), remote_speech, mic_speech, now)
            if seg is not None:
                seg.queued_at = time.monotonic()
                work.put(seg)

            # D25: hand the decoder a prefix of the segment that is still open. `now` is
            # AUDIO time, not wall time, so a --from-wav replay produces the provisionals the
            # live call would have produced. The snapshot is a ~1 MB memcpy every
            # PARTIAL_DECODE_SECONDS on a 20 ms frame budget, and the mailbox never blocks,
            # so the capture loop still cannot be held up by the GPU.
            if mailbox is not None and partial_clock.due(segmenter.open_start(), now):
                snapshot = segmenter.snapshot(now)
                if snapshot is not None:
                    mailbox.put(snapshot)

            if args.seconds and now >= args.seconds:
                break
            if player is not None and player.poll() is not None:
                # The self-test WAV finished; drain a moment so the VAD sees the
                # trailing silence and closes the last segment.
                drain_deadline = drain_deadline or now + settings.SEGMENT_MAX_SILENCE_SECONDS + 0.5
                if now >= drain_deadline:
                    break
    except KeyboardInterrupt:
        status = "interrupted"
        print("\n[forced stop]", flush=True)
    except Exception:  # noqa: BLE001 — report, then still close the files cleanly
        status = "error"
        logger.exception("capture loop failed")
    finally:
        for stream in streams:
            stream.close()
        if player is not None and player.poll() is None:
            player.terminate()
        tail = segmenter.flush(frames_seen * frame_seconds)
        if tail is not None:
            tail.queued_at = time.monotonic()
            work.put(tail)
        pending = work.qsize()
        if pending:
            print(f"[finishing {pending} pending segment(s)...]", flush=True)
        work.put(None)
        worker.join()
        audio_seconds = frames_seen * frame_seconds
        out.close()
        if trace is not None:
            trace.close()
            print(f"latency trace   : {trace.path} ({trace.rows} row(s))")
        end_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        append_run_row(run_id, start_iso, end_iso, status, audio_seconds, out)
        dropped = mailbox.dropped if mailbox is not None else 0
        logger.info(
            "run %s END | status=%s in=%.1fs out=%d lines (decoded %d segments in %.2fs) "
            "| provisional: %d decoded in %.2fs, %d written, %d superseded before decode, "
            "%d snapshots displaced | capture drift: max=%.2fs total=%.2fs",
            run_id, status, audio_seconds, out.lines, stats["decoded"], stats["decode_seconds"],
            stats["partial_decoded"], stats["partial_decode_seconds"], stats["partial_written"],
            stats["partial_superseded"], dropped,
            drift_monitor.max_drift, drift_monitor.total_drift,
        )
        print("\n=== RUN COMPLETE ===")
        print(f"audio captured : {audio_seconds:.1f}s ({stats['decoded']} segments decoded)")
        print(f"transcript     : {out.transcript_path}")
        print(f"scorer input   : {out.plain_path}")
        # Capture drift (#458): wall clock minus audio clock (frames_seen * frame_seconds),
        # sampled every frame by CaptureDriftMonitor. Positive = audio behind wall = a
        # capture (parec/PipeWire) stall — the only thing that has ever breached the meter's
        # 31.0 s ceiling on the real call (D27/#428: a 2.0 s stall, gap 32.31 s vs 30 s spacing).
        # max = the single worst stall seen; total = the sum of all stall growth over the run.
        print(
            f"capture drift  : max {drift_monitor.max_drift:.2f}s, "
            f"total {drift_monitor.total_drift:.2f}s (positive = capture behind wall clock)"
        )
        if out.partial_path:
            # `displaced` is not a failure: it is the count of prefixes the decoder was too
            # busy to reach before a fresher one arrived, which is exactly what a newest-wins
            # mailbox is for. It runs high on a --from-wav replay, where audio time outruns
            # the GPU by design.
            print(
                f"provisional    : {out.partial_path}\n"
                f"                 {stats['partial_written']} line(s) written from "
                f"{stats['partial_decoded']} decode(s) in {stats['partial_decode_seconds']:.2f}s"
                f" ({dropped} displaced, {stats['partial_superseded']} superseded before decode)"
                f" — superseded by the transcript (D25)"
            )
            # The two numbers D25 is answerable for. Duty is GPU-seconds per audio-second, so
            # it is comparable to the 2.6% the final path costs and to D25's ~10% estimate.
            # The delay is the one a provisional can impose on a CLOSING segment's line, which
            # is the number that can hurt: it eats into the meter's decode allowance.
            if audio_seconds > 0:
                final_duty = 100.0 * stats["decode_seconds"] / audio_seconds
                partial_duty = 100.0 * stats["partial_decode_seconds"] / audio_seconds
                print(
                    f"                 GPU duty: finals {final_duty:.1f}% + provisional "
                    f"{partial_duty:.1f}% of audio time"
                )
            print(
                f"                 delay imposed on a closing line: "
                f"{stats['final_behind_partial']}/{stats['final_waits']} finals waited behind an "
                f"in-flight provisional, worst {stats['final_behind_partial_max']:.2f}s "
                f"(worst queue wait of any kind {stats['final_wait_max']:.2f}s)"
            )
        if out.wav_path:
            print(f"recording      : {out.wav_path}   <- delete after scoring (D11)")
        print(
            "\nscore it with:\n"
            f"  python scripts/score_transcript.py --reference scripts/inputs/test_script_pl.txt "
            f"--hypothesis-file {out.plain_path.relative_to(PROJECT_ROOT)}"
        )
    return 0 if status in ("ok", "interrupted") else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--list", action="store_true", help="list capture sources and exit")
    parser.add_argument("--source", default="auto", help="monitor source to capture (default: auto = default sink's monitor)")
    parser.add_argument("--mic", default="auto", help="microphone source (default: auto = default input)")
    parser.add_argument("--no-mic", action="store_true", help="capture the monitor only (your own voice will NOT be recorded)")
    parser.add_argument("--seconds", type=float, default=0.0, help="auto-stop after N seconds (0 = run until Ctrl-C)")
    parser.add_argument("--latency-trace", default="", metavar="FILE",
                        help="#428: write one JSONL row per final segment with its close->write latency")
    parser.add_argument("--segmentation", choices=("vad", "fixed"), default="vad", help="segmentation policy ('fixed' is A/B comparison only)")
    parser.add_argument("--fixed-seconds", type=float, default=8.0, help="window length for --segmentation fixed")
    parser.add_argument("--overlap-seconds", type=float, default=0.0, help="overlap carried between fixed windows")
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false", help="keep overlapping words in continued segments")
    parser.add_argument("--no-partials", dest="partials", action="store_false", default=None, help="do not write D25 provisional lines (default: PARTIAL_DECODE_ENABLED)")
    parser.add_argument("--partial-seconds", type=float, default=0.0, help="provisional decode cadence in seconds (0 = PARTIAL_DECODE_SECONDS)")
    parser.add_argument("--from-wav", help="offline: push a WAV file through the loop instead of live audio")
    parser.add_argument("--pace", type=float, default=0.0, help="with --from-wav: feed at N x real time (1.0 = a real call's clock; 0 = as fast as possible). Required to exercise the D25 cadence offline")
    parser.add_argument("--no-record", action="store_true", help="do not write the audio WAV (offline scoring runs)")
    parser.add_argument("--selftest-sink", help="no-human proof: play --selftest-wav into this SINK and capture its .monitor")
    parser.add_argument("--selftest-wav", default=str(OUTPUT_DIR / "piper_test.wav"), help="WAV used by --selftest-sink")
    args = parser.parse_args()

    if args.list:
        list_sources()
        sys.exit(0)
    if args.selftest_sink and args.source == "auto":
        args.source = f"{args.selftest_sink}.monitor"
    if args.from_wav:
        args.no_mic = True

    try:
        sys.exit(run_loop(args))
    except Exception:  # noqa: BLE001 — top-level runner (guides/coding_pipeline.md)
        logger.exception("live_transcribe failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
