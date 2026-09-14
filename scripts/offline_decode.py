"""Offline best-effort re-decode of a recorded two-channel call (G8, #324).

Why this exists: the live loop decodes 4–30 s VAD segments with `large-v3-turbo`
under a latency budget. That budget is a *choice*, and the question this script
answers is what the same audio yields when latency is not a constraint —
whole-file `large-v3`, a wider beam, faster-whisper's own VAD, and per-segment
language detection instead of ours.

IMPORTANT — what this is NOT (D24): the output is **not a reference transcript**. Both
sides are Whisper, so a diff against the live transcript measures DIVERGENCE, not
word error rate. It localises where the live pipeline lost content; it cannot say
which side is right. A true WER needs a human ear (see `scripts/inputs/`).

SI1: decoding is local (faster-whisper on this GPU). Nothing is uploaded.
D11: the input WAV is a disclosed recording — this script only reads it.

Usage (lease the GPU — CLAUDE.md §GPU Leases):
    python -m commons.coordination.gpu run --vram 6000 --label "#324 wer redecode" -- \
        python scripts/offline_decode.py scripts/outputs/live_audio_20260902_100033.wav
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
import wave
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings  # noqa: E402
from scripts.logger import get_logger  # noqa: E402

logger = get_logger("offline_decode")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "outputs"
RUNS_CSV = PROJECT_ROOT / "logs" / "runs.csv"

# The live recorder writes L = remote (interviewer), R = mic (candidate) —
# live_transcribe.py `open_outputs`. Decoding the channels separately keeps
# speaker attribution exact instead of inferring it from VAD votes.
CHANNEL_SPEAKERS: dict[int, str] = {0: "them", 1: "you"}


@dataclass
class DecodedSegment:
    """One offline-decoded segment, timed from the start of the recording."""

    start: float
    end: float
    speaker: str
    language: str
    language_probability: float
    text: str
    avg_logprob: float
    no_speech_prob: float


def read_wav_channels(path: Path) -> tuple[list[np.ndarray], int]:
    """Load a PCM WAV as a list of mono float32 channels in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    if sampwidth not in dtype_map:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")
    data = np.frombuffer(frames, dtype=dtype_map[sampwidth]).astype(np.float32)
    data /= float(np.iinfo(dtype_map[sampwidth]).max)
    if n_channels > 1:
        data = data.reshape(-1, n_channels)
        channels = [np.ascontiguousarray(data[:, c]) for c in range(n_channels)]
    else:
        channels = [data]
    return channels, sample_rate


def decode_channel(
    model: object,
    audio: np.ndarray,
    speaker: str,
    beam_size: int,
    language: str | None,
) -> list[DecodedSegment]:
    """Decode one channel whole-file. `language=None` → per-segment detection."""
    segments_iter, info = model.transcribe(  # type: ignore[attr-defined]
        audio,
        beam_size=beam_size,
        language=language,
        multilingual=language is None,
        vad_filter=True,
        condition_on_previous_text=True,
        word_timestamps=False,
        log_progress=False,
    )
    logger.info(
        "channel %s: decoding (whole-file lang=%s detected=%s p=%.2f)",
        speaker, language or "auto", info.language, info.language_probability,
    )
    out: list[DecodedSegment] = []
    for s in segments_iter:  # lazy: decoding happens here
        text = s.text.strip()
        if not text:
            continue
        out.append(
            DecodedSegment(
                start=round(float(s.start), 2),
                end=round(float(s.end), 2),
                speaker=speaker,
                language=str(getattr(s, "language", None) or info.language),
                language_probability=round(float(info.language_probability), 3),
                text=text,
                avg_logprob=round(float(s.avg_logprob), 3),
                no_speech_prob=round(float(s.no_speech_prob), 3),
            )
        )
        if len(out) % 25 == 0:
            logger.info("channel %s: %d segments, t=%.0fs", speaker, len(out), out[-1].end)
    return out


def append_run_row(
    run_id: str, start_iso: str, end_iso: str, status: str,
    audio_seconds: float, n_segments: int, paths: list[Path],
) -> None:
    """Trackability (CLAUDE.md TRK): one row per run in logs/runs.csv."""
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = [
        run_id, "offline_decode.py", start_iso, end_iso, status,
        f"{audio_seconds:.1f}", str(n_segments), ";".join(p.name for p in paths),
    ]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("wav", help="recorded call WAV (stereo L=remote R=mic, or mono)")
    parser.add_argument("--model", default="large-v3", help="faster-whisper model (default: large-v3)")
    parser.add_argument("--beam", type=int, default=8, help="beam size (default: 8; live uses 5)")
    parser.add_argument("--compute-type", default=settings.STT_COMPUTE_TYPE)
    parser.add_argument("--device", default=settings.STT_DEVICE)
    parser.add_argument(
        "--language", default=None,
        help="force a decode language (default: per-segment detection, multilingual=True)",
    )
    parser.add_argument("--limit-seconds", type=float, default=None, help="decode only the first N s (smoke test)")
    args = parser.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        logger.error("no such WAV: %s", wav_path)
        sys.exit(1)

    run_id = f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    start_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    logger.info(
        "run %s START | wav=%s model=%s beam=%d lang=%s",
        run_id, wav_path.name, args.model, args.beam, args.language or "auto",
    )

    channels, sample_rate = read_wav_channels(wav_path)
    if sample_rate != settings.SAMPLE_RATE:
        raise ValueError(f"expected {settings.SAMPLE_RATE} Hz, got {sample_rate} Hz")
    if args.limit_seconds:
        keep = int(args.limit_seconds * sample_rate)
        channels = [c[:keep] for c in channels]
    audio_seconds = len(channels[0]) / sample_rate
    logger.info("loaded %.1fs of audio, %d channel(s)", audio_seconds, len(channels))

    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    logger.info("model %s loaded in %.1fs", args.model, time.perf_counter() - t0)

    segments: list[DecodedSegment] = []
    for idx, audio in enumerate(channels):
        speaker = CHANNEL_SPEAKERS.get(idx, f"ch{idx}") if len(channels) > 1 else "them"
        t_ch = time.perf_counter()
        got = decode_channel(model, audio, speaker, args.beam, args.language)
        logger.info(
            "channel %s done: %d segments in %.1fs (rtf %.3f)",
            speaker, len(got), time.perf_counter() - t_ch,
            (time.perf_counter() - t_ch) / audio_seconds if audio_seconds else 0.0,
        )
        segments.extend(got)
    segments.sort(key=lambda s: (s.start, s.speaker))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"offline_decode_{stamp}.json"
    txt_path = OUTPUT_DIR / f"offline_transcript_{stamp}.txt"
    plain_path = OUTPUT_DIR / f"offline_transcript_plain_{stamp}.txt"

    meta = {
        "run_id": run_id,
        "source_wav": wav_path.name,
        "audio_seconds": round(audio_seconds, 1),
        "model": args.model,
        "beam_size": args.beam,
        "compute_type": args.compute_type,
        "language": args.language or "auto (multilingual per-segment)",
        "vad_filter": True,
        "note": "OFFLINE best-effort decode. NOT a reference transcript — machine vs machine.",
    }
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"meta": meta, "segments": [asdict(s) for s in segments]}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    tmp.replace(json_path)

    header_lines = [f"# {k}: {v}" for k, v in meta.items()]
    txt_path.write_text(
        "\n".join(header_lines)
        + "\n"
        + "\n".join(
            f"[{int(s.start)//60:02d}:{int(s.start)%60:02d}-{int(s.end)//60:02d}:{int(s.end)%60:02d}] "
            f"{s.speaker} ({s.language}): {s.text}"
            for s in segments
        )
        + "\n",
        encoding="utf-8",
    )
    plain_path.write_text("\n".join(s.text for s in segments) + "\n", encoding="utf-8")

    end_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    append_run_row(run_id, start_iso, end_iso, "ok", audio_seconds, len(segments),
                   [json_path, txt_path, plain_path])
    logger.info("run %s END | %d segments -> %s", run_id, len(segments), txt_path.name)
    print(txt_path)


if __name__ == "__main__":
    main()
