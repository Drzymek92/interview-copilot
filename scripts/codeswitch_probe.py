"""Measure the G8(d) straddling-segment code-switch failure (#397).

A measurement instrument, not a fix. It answers three questions about a recorded
run, deterministically and without touching the live loop:

1. **What audio did the live loop actually decode?** `replay_segments()` pushes the
   recorded stereo WAV (L = remote, R = mic) back through the *same* `ChannelGate` +
   `VadSegmenter` the live run used, so segment boundaries are reproduced rather than
   guessed from the transcript's mm:ss stamps. `--verify` checks that reproduction
   against a transcript before any GPU work is done.
2. **Where does the language vote actually land?** `--probe` reports
   `detect_language` on the whole segment *and* on sliding sub-windows, so a segment
   that straddles a switch shows up as sub-windows that disagree.
3. **What does each language cost?** `--probe` decodes the segment forced to each
   candidate language and reports the duration-weighted `avg_logprob` of each, which
   is the score candidate (a) would arbitrate on.

Usage (GPU work must go through the lease board — see CLAUDE.md):

    python scripts/codeswitch_probe.py --verify --wav W --transcript T
    python scripts/codeswitch_probe.py --probe --wav W --transcript T --segments 46,55
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings  # noqa: E402
from scripts.live_transcribe import ChannelGate, Segment, VadSegmenter, mix  # noqa: E402
from scripts.logger import get_logger  # noqa: E402

logger = get_logger("codeswitch_probe")

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "outputs"

LINE_RE = re.compile(
    r"^\[(?P<s_m>\d+):(?P<s_s>\d\d)-(?P<e_m>\d+):(?P<e_s>\d\d)\]\s+"
    r"(?P<speaker>them|you)(?:\s+\((?P<lang>[a-z-]+)\))?:\s*(?P<text>.*)$"
)


@dataclass
class TranscriptLine:
    """One parsed line of a live transcript (`.txt`), 1-based by file line number."""

    lineno: int
    start: int  # whole seconds, as printed
    end: int
    speaker: str
    language: str
    text: str


def parse_transcript(path: Path) -> list[TranscriptLine]:
    """Parse the human transcript into lines, skipping the `#` header."""
    out: list[TranscriptLine] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        m = LINE_RE.match(raw.strip())
        if not m:
            continue
        out.append(
            TranscriptLine(
                lineno=i,
                start=int(m["s_m"]) * 60 + int(m["s_s"]),
                end=int(m["e_m"]) * 60 + int(m["e_s"]),
                speaker=m["speaker"],
                language=m["lang"] or "",
                text=m["text"],
            )
        )
    return out


def replay_segments(wav_path: Path) -> Iterator[Segment]:
    """Re-derive the live loop's segments from the recorded WAV.

    The recorder wrote one WAV frame per capture-loop iteration from the very frames
    the gates judged (L = remote, R = mic), so feeding those channels back through the
    same gates and segmenter reproduces the boundaries exactly — including the pre-roll,
    the 30 s force-cut and its carry-over. Mono recordings (monitor-only runs) are fed
    as `remote` with no mic, which is what the loop itself does there.
    """
    import webrtcvad

    frame_seconds = settings.VAD_FRAME_MS / 1000.0
    samples_per_frame = int(settings.SAMPLE_RATE * frame_seconds)

    gate_remote = ChannelGate(webrtcvad.Vad(settings.VAD_AGGRESSIVENESS), "remote")
    gate_mic = ChannelGate(webrtcvad.Vad(settings.VAD_AGGRESSIVENESS), "mic")
    segmenter = VadSegmenter(frame_seconds=frame_seconds)

    with wave.open(str(wav_path), "rb") as wf:
        if wf.getframerate() != settings.SAMPLE_RATE or wf.getsampwidth() != 2:
            raise ValueError(
                f"{wav_path.name}: expected 16-bit {settings.SAMPLE_RATE} Hz PCM, got "
                f"{wf.getsampwidth() * 8}-bit {wf.getframerate()} Hz"
            )
        n_channels = wf.getnchannels()
        frames_seen = 0
        while True:
            block = wf.readframes(samples_per_frame)
            if len(block) < samples_per_frame * 2 * n_channels:
                break
            if n_channels == 2:
                inter = np.frombuffer(block, dtype=np.int16)
                remote = inter[0::2].tobytes()
                mic_frame: bytes | None = inter[1::2].tobytes()
            else:
                remote, mic_frame = block, None

            frames_seen += 1
            now = frames_seen * frame_seconds
            remote_speech = gate_remote.is_speech(remote)
            mic_speech = gate_mic.is_speech(mic_frame) if mic_frame is not None else False
            seg = segmenter.push(mix(remote, mic_frame), remote_speech, mic_speech, now)
            if seg is not None:
                yield seg
        tail = segmenter.flush(frames_seen * frame_seconds)
        if tail is not None:
            yield tail


def segment_audio(seg: Segment) -> np.ndarray:
    """The exact mono float32 buffer the live loop handed the transcriber."""
    return np.frombuffer(seg.pcm, dtype=np.int16).astype(np.float32) / 32768.0


def collect_segments(wav_path: Path) -> list[Segment]:
    """Replayed segments the live loop would have HANDED THE DECODER (silence dropped)."""
    kept: list[Segment] = []
    for seg in replay_segments(wav_path):
        audio = segment_audio(seg)
        if float(np.max(np.abs(audio))) < settings.SEGMENT_MIN_PEAK:
            continue
        kept.append(seg)
    return kept


def align(lines: list[TranscriptLine], segs: list[Segment]) -> dict[int, Segment]:
    """Map transcript line ordinal (1-based) -> the segment that produced it.

    Not a 1:1 zip: the live worker writes NO line for a segment that decodes to empty
    text (`if not text: continue`), so the replay legitimately has more segments than the
    transcript has lines. Matching is on the printed stamps, which `mmss` truncates.
    """
    by_stamp: dict[tuple[int, int], list[Segment]] = {}
    for seg in segs:
        by_stamp.setdefault((int(seg.start), int(seg.end)), []).append(seg)
    out: dict[int, Segment] = {}
    for i, line in enumerate(lines, start=1):
        bucket = by_stamp.get((line.start, line.end))
        if bucket:
            out[i] = bucket.pop(0)
    return out


def verify(wav_path: Path, transcript_path: Path) -> int:
    """Check the replayed segmentation against a transcript. Returns an exit code."""
    lines = parse_transcript(transcript_path)
    segs = collect_segments(wav_path)
    matched = align(lines, segs)
    unmatched_lines = [i for i in range(1, len(lines) + 1) if i not in matched]
    extra = len(segs) - len(matched)

    print(f"transcript lines      : {len(lines)}")
    print(f"replayed segments     : {len(segs)}")
    print(f"lines matched to audio: {len(matched)}")
    print(f"transcript lines with NO replayed segment: {len(unmatched_lines)} {unmatched_lines[:10]}")
    print(f"replayed segments with no line (empty decode): {extra}")
    ok = not unmatched_lines
    print("VERDICT:", "every transcript line is backed by an exactly-reproduced segment"
          if ok else "DIVERGES — do not trust slices")
    return 0 if ok else 1


def _weighted_avg_logprob(fw_segments: list) -> tuple[float, float]:
    """Duration-weighted mean `avg_logprob` over faster-whisper segments, and total dur."""
    total = sum(max(0.0, s.end - s.start) for s in fw_segments)
    if not fw_segments or total <= 0:
        return float("nan"), 0.0
    weighted = sum(s.avg_logprob * max(0.0, s.end - s.start) for s in fw_segments)
    return weighted / total, total


def probe(
    wav_path: Path,
    transcript_path: Path,
    wanted: list[int],
    window: float,
    hop: float,
    languages: tuple[str, ...],
) -> dict:
    """Report the language vote and the per-language decode score for chosen segments.

    `wanted` are 1-based transcript line ordinals (as `--verify` numbers them), which
    `verify` has already shown to line up with the replayed segments.
    """
    from faster_whisper import WhisperModel

    lines = parse_transcript(transcript_path)
    matched = align(lines, collect_segments(wav_path))
    if not wanted:  # `--segments all`: the whole call, failures and control alike
        wanted = sorted(matched)
    missing = [o for o in wanted if o not in matched]
    if missing:
        raise SystemExit(f"transcript lines {missing} have no reproduced segment — run --verify")
    by_ordinal = {o: matched[o] for o in wanted}

    logger.info("loading %s on %s", settings.STT_MODEL, settings.STT_DEVICE)
    model = WhisperModel(
        settings.STT_MODEL, device=settings.STT_DEVICE, compute_type=settings.STT_COMPUTE_TYPE
    )

    report: dict = {
        "model": settings.STT_MODEL,
        "beam_size": settings.STT_BEAM_SIZE,
        "window_seconds": window,
        "hop_seconds": hop,
        "wav": wav_path.name,
        "transcript": transcript_path.name,
        "segments": [],
    }
    for n_done, ordinal in enumerate(sorted(by_ordinal), start=1):
        seg = by_ordinal[ordinal]
        if len(by_ordinal) > 10 and n_done % 20 == 0:
            logger.info("probed %d/%d segments", n_done, len(by_ordinal))
        audio = segment_audio(seg)
        line = lines[ordinal - 1] if ordinal - 1 < len(lines) else None

        lang, prob, all_probs = model.detect_language(audio)
        top = sorted(all_probs, key=lambda t: -t[1])[:4]
        entry: dict = {
            "ordinal": ordinal,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "duration": round(len(audio) / settings.SAMPLE_RATE, 2),
            "speaker": seg.speaker,
            "live_language_tag": line.language if line else "",
            "live_text": line.text if line else "",
            "full_detect": {"lang": lang, "prob": round(prob, 4),
                            "top": [[t, round(p, 4)] for t, p in top]},
            "windows": [],
            "decodes": {},
        }
        # Where in the segment does the vote change? A straddling segment shows up as
        # sub-windows that disagree; a homogeneous one votes the same way throughout.
        n = len(audio)
        step = int(hop * settings.SAMPLE_RATE)
        span = int(window * settings.SAMPLE_RATE)
        offset = 0
        while offset < n:
            chunk = audio[offset : offset + span]
            if len(chunk) < int(1.0 * settings.SAMPLE_RATE):
                break
            w_lang, w_prob, w_all = model.detect_language(chunk)
            probs = dict(w_all)
            entry["windows"].append(
                {
                    "t0": round(offset / settings.SAMPLE_RATE, 2),
                    "t1": round((offset + len(chunk)) / settings.SAMPLE_RATE, 2),
                    "lang": w_lang,
                    "prob": round(w_prob, 4),
                    "p_pl": round(float(probs.get("pl", 0.0)), 4),
                    "p_en": round(float(probs.get("en", 0.0)), 4),
                }
            )
            offset += step
        # What each forced language costs: the score candidate (a) would arbitrate on.
        for lang_code in languages:
            seg_iter, _info = model.transcribe(
                audio, beam_size=settings.STT_BEAM_SIZE, language=lang_code, vad_filter=True
            )
            fw = list(seg_iter)
            mean_lp, dur = _weighted_avg_logprob(fw)
            entry["decodes"][lang_code] = {
                "avg_logprob": None if np.isnan(mean_lp) else round(mean_lp, 4),
                "decoded_seconds": round(dur, 2),
                "n_subsegments": len(fw),
                "text": " ".join(s.text.strip() for s in fw).strip(),
            }
        report["segments"].append(entry)
    return report



def replay_decode(
    wav_path: Path, transcript_path: Path, mode: str, out_path: Path, limit: int
) -> dict:
    """Re-decode every reproduced segment through the REAL `Transcriber` seam.

    Boundaries are held FIXED at the ones the live run produced, so a difference between
    two runs of this harness is attributable to `STT_CODESWITCH_MODE` alone and not to
    segmentation — which is the comparison D21 and #400 make impossible to read off a
    `--from-wav` replay. It writes a transcript in the live `.txt` line format so
    `diff_transcripts.py` scores it unchanged.
    """
    import os

    os.environ["STT_CODESWITCH_MODE"] = mode
    import importlib

    from config import settings as _settings

    importlib.reload(_settings)
    from scripts import stt as _stt

    importlib.reload(_stt)

    lines = parse_transcript(transcript_path)
    matched = align(lines, collect_segments(wav_path))
    ordinals = sorted(matched)[: limit or None]

    transcriber = _stt.Transcriber()
    rows: list[dict] = []
    out_lines = [
        f"# codeswitch replay | mode={mode} | model={settings.STT_MODEL}\n",
        f"# source: {wav_path.name} + {transcript_path.name} (live segment boundaries held fixed)\n",
    ]
    for ordinal in ordinals:
        seg = matched[ordinal]
        audio = segment_audio(seg)
        result = transcriber.transcribe_array(audio, sample_rate=settings.SAMPLE_RATE)
        text = result.text.strip()
        rows.append(
            {
                "ordinal": ordinal,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "duration": round(result.audio_seconds, 2),
                "speaker": seg.speaker,
                "language": result.language,
                "languages": list(result.languages),
                "code_switch": bool(result.code_switch),
                "decode_passes": int(result.decode_passes),
                "latency_seconds": round(result.latency_seconds, 4),
                "avg_logprob": round(result.avg_logprob, 4),
                "text": text,
            }
        )
        if text:
            out_lines.append(
                f"[{int(seg.start) // 60:02d}:{int(seg.start) % 60:02d}-"
                f"{int(seg.end) // 60:02d}:{int(seg.end) % 60:02d}] "
                f"{seg.speaker} ({result.language}): {text}\n"
            )
    out_path.write_text("".join(out_lines), encoding="utf-8")
    switched = [r for r in rows if r["code_switch"]]
    return {
        "mode": mode,
        "model": settings.STT_MODEL,
        "wav": wav_path.name,
        "transcript_out": str(out_path),
        "n_segments": len(rows),
        "audio_seconds": round(sum(r["duration"] for r in rows), 2),
        "decode_seconds": round(sum(r["latency_seconds"] for r in rows), 3),
        "n_code_switch": len(switched),
        "code_switch_ordinals": [r["ordinal"] for r in switched],
        "extra_decode_seconds_on_switched": round(
            sum(r["latency_seconds"] for r in switched), 3
        ),
        "segments": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", required=True, help="recorded stereo WAV from the live run")
    ap.add_argument("--transcript", required=True, help="the live_transcript_*.txt of that run")
    ap.add_argument("--verify", action="store_true", help="reproduce segmentation only (no GPU)")
    ap.add_argument("--probe", action="store_true", help="detect + per-language decode (GPU)")
    ap.add_argument("--replay", default="", metavar="MODE",
                    help="re-decode every segment through Transcriber under STT_CODESWITCH_MODE=MODE (GPU)")
    ap.add_argument("--limit", type=int, default=0, help="with --replay: only the first N segments")
    ap.add_argument("--segments", default="", help="comma-separated 1-based transcript line ordinals, or 'all'")
    ap.add_argument("--quiet", action="store_true", help="write the JSON, skip the per-segment dump")
    ap.add_argument("--window", type=float, default=6.0, help="sub-window seconds for detection")
    ap.add_argument("--hop", type=float, default=3.0, help="sub-window hop seconds")
    ap.add_argument("--languages", default="pl,en", help="languages to force-decode under")
    ap.add_argument("--out", default="", help="write the JSON report here (default: auto-stamped)")
    args = ap.parse_args()

    wav_path, transcript_path = Path(args.wav), Path(args.transcript)
    if args.verify:
        sys.exit(verify(wav_path, transcript_path))
    if args.replay:
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        base = Path(args.out) if args.out else OUTPUT_DIR / f"codeswitch_replay_{args.replay}_{stamp}"
        base.parent.mkdir(parents=True, exist_ok=True)
        summary = replay_decode(
            wav_path, transcript_path, args.replay, base.with_suffix(".txt"), args.limit
        )
        base.with_suffix(".json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"mode={summary['mode']}  segments={summary['n_segments']}  "
              f"audio={summary['audio_seconds']:.0f}s  decode={summary['decode_seconds']:.1f}s  "
              f"code_switch={summary['n_code_switch']} {summary['code_switch_ordinals']}")
        print(f"transcript: {base.with_suffix('.txt')}")
        print(f"summary   : {base.with_suffix('.json')}")
        return
    if not args.probe:
        ap.error("choose --verify, --probe or --replay")

    if args.segments.strip().lower() == "all":
        wanted: list[int] = []
    else:
        wanted = [int(x) for x in args.segments.split(",") if x.strip()]
        if not wanted:
            ap.error("--probe needs --segments (ordinals, or 'all')")
    langs = tuple(x.strip() for x in args.languages.split(",") if x.strip())
    report = probe(wav_path, transcript_path, wanted, args.window, args.hop, langs)

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    out_path = Path(args.out) if args.out else OUTPUT_DIR / f"codeswitch_probe_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %s", out_path)
    if args.quiet:
        print(f"report: {out_path}")
        return
    for entry in report["segments"]:
        print(f"\n=== #{entry['ordinal']} [{entry['start']:.1f}-{entry['end']:.1f}] "
              f"{entry['speaker']} live-tag={entry['live_language_tag']} "
              f"dur={entry['duration']:.1f}s")
        fd = entry["full_detect"]
        print(f"  full-segment detect: {fd['lang']} p={fd['prob']:.3f}  top={fd['top']}")
        for w in entry["windows"]:
            print(f"    [{w['t0']:5.1f}-{w['t1']:5.1f}] {w['lang']} p={w['prob']:.3f} "
                  f"(pl={w['p_pl']:.3f} en={w['p_en']:.3f})")
        for lang_code, d in entry["decodes"].items():
            lp = d["avg_logprob"]
            print(f"  decode[{lang_code}] avg_logprob={lp} ({d['n_subsegments']} sub, "
                  f"{d['decoded_seconds']:.1f}s)")
            print(f"    {d['text'][:400]}")
    print(f"\nreport: {out_path}")


if __name__ == "__main__":
    main()
