"""Day-1 audio-capture spike (D17) — monitor-source capture proven 2026-08-31.

The riskiest de-risk (G2): prove we can capture the interviewer's voice off a
Teams call via a PipeWire/PulseAudio **monitor** source, push it through the
local Whisper wrapper (`scripts/stt.py`), and get a transcript with acceptable
end-to-end latency.

STATUS: **PASS.** The monitor->STT path is verified end-to-end by
`--selftest-sink` (speech WAV -> sink -> sink.monitor -> Whisper): exact
transcript, 0.37 s decode for an 8 s window (rtf 0.05), model resident. What
remains machine-state-dependent is only which SINK Teams plays to — see step 1.

(The Day-1 note that the agent sandbox has no audio server is obsolete: the
PipeWire socket at $XDG_RUNTIME_DIR/pulse/native IS reachable, so `--list` and
`--selftest-sink` run fine headlessly.)

--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------
Prereqs (one-time) — all satisfied on lab as of 2026-08-31:
    * pip deps installed (see config/requirements.txt) — faster-whisper.
    * `pactl` + `parec` + `paplay` (apt: `sudo apt install pulseaudio-utils`) —
      these do the capture; PortAudio/sounddevice is NOT on the capture path.

1. List every capture source (mic + monitors) and pick the Teams one:

       python scripts/spike_capture.py --list

   Monitor sources end in `.monitor` — they are the *output* of a sink (what you
   HEAR). To capture the interviewer, pick the monitor of the sink Teams plays to
   (usually your headphones/speakers), e.g. `alsa_output.pci-....analog-stereo.monitor`.
   TIP: start a Teams test call (or play any audio through those speakers), run
   `--list` again, and note which monitor shows recent activity. To be certain
   Teams routes to a known sink, use `pavucontrol` → Playback tab → set Microsoft
   Teams' output to that sink while this spike captures its `.monitor`.

2. Capture N seconds from that source and transcribe:

       python scripts/spike_capture.py --source "alsa_output.pci-....analog-stereo.monitor" --seconds 8

   Speak (or have the call speak) during the capture window. The script prints the
   transcript and the end-to-end latency (capture-stop → transcript-ready).

3. To capture YOUR mic instead (the other channel), pass its source name from
   `--list` (the non-monitor input, e.g. `alsa_input....`).

--------------------------------------------------------------------------------
NOTES
--------------------------------------------------------------------------------
* Capture routes through `parec --device=<source>` (pulseaudio-utils), NOT
  PortAudio. Verified 2026-08-31: the conda-forge PortAudio build on lab exposes
  no `pulse` device at all (only raw ALSA hw devices), so the original
  `device="pulse"` + PULSE_SOURCE approach raised "No input device matching
  'pulse'". `parec` speaks the native PipeWire/Pulse protocol and selects an
  exact source without touching the system default.
* `--selftest-sink <sink>` proves the whole monitor->STT path with no human and
  no call: it plays a known speech WAV into the sink and captures its monitor.
* Day-2/3 import `scripts.stt.Transcriber`, NOT this spike. This file is throwaway
  proof; the reusable wrapper is stt.py.
* SI1 (local-only): capture + transcription are entirely local; nothing is sent
  off-box here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings  # noqa: E402
from scripts.logger import get_logger  # noqa: E402

logger = get_logger("spike_capture")


def _have_pactl() -> bool:
    import shutil

    return shutil.which("pactl") is not None


def list_sources() -> None:
    """Print PortAudio devices and PulseAudio/PipeWire sources (monitors flagged)."""
    print("=== PortAudio devices (sounddevice) ===")
    try:
        import sounddevice as sd

        print(sd.query_devices())
    except Exception as exc:  # noqa: BLE001 — self-diagnosing spike
        print(f"  [sounddevice unavailable] {exc}")
        print("  -> Install PortAudio (conda-forge portaudio / apt libportaudio2).")

    print("\n=== PulseAudio/PipeWire sources (pactl) ===")
    if not _have_pactl():
        print("  [pactl not found] install pulseaudio-utils:")
        print("      sudo apt install pulseaudio-utils")
        print("  Is an audio server running? (this must run in your desktop session)")
        return
    try:
        out = subprocess.run(
            ["pactl", "list", "sources", "short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        print(f"  [pactl failed] {exc.stderr or exc}")
        print("  No audio server reachable — run this at the graphical desktop.")
        return
    if not out:
        print("  (no sources reported)")
        return
    for line in out.splitlines():
        is_monitor = ".monitor" in line
        tag = "  <-- MONITOR (capture interviewer/output here)" if is_monitor else ""
        print(f"  {line}{tag}")
    print(
        "\nPick the `.monitor` of the sink Teams plays to for the interviewer voice;"
        "\npick the plain input source for your mic."
    )


def capture_pcm(source: str, seconds: float) -> np.ndarray:
    """Capture `seconds` of mono float32 audio from a pactl `source` via `parec`.

    Uses `parec` (pulseaudio-utils) rather than PortAudio: the conda-forge
    PortAudio build on this box exposes NO `pulse` device (verified 2026-08-31 —
    `sd.query_devices()` lists only raw ALSA hw devices), so `device="pulse"`
    raises "No input device matching 'pulse'". `parec` talks the native protocol,
    selects an exact source with --device, and needs no extra install.
    """
    sample_rate = settings.SAMPLE_RATE
    channels = settings.CHANNELS
    want_bytes = int(seconds * sample_rate) * channels * 2  # s16le = 2 bytes

    cmd = [
        "parec",
        f"--device={source}",
        "--format=s16le",
        f"--rate={sample_rate}",
        f"--channels={channels}",
        "--raw",
    ]
    logger.info(
        "capturing %.1fs from source=%r at %d Hz (parec)", seconds, source, sample_rate
    )

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buf = bytearray()
    deadline = time.monotonic() + seconds + 5.0  # grace for stream setup
    try:
        assert proc.stdout is not None
        while len(buf) < want_bytes and time.monotonic() < deadline:
            chunk = proc.stdout.read(min(8192, want_bytes - len(buf)))
            if not chunk:
                break
            buf.extend(chunk)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    if len(buf) < want_bytes:
        err = (
            proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        ).strip()
        logger.warning(
            "short capture: got %d/%d bytes%s",
            len(buf),
            want_bytes,
            f" — parec: {err}" if err else "",
        )
    return np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32) / 32768.0


def capture_and_transcribe(source: str, seconds: float) -> int:
    """Capture `seconds` from `source` (a pactl source name) and transcribe it."""
    if not _have_pactl():
        logger.error(
            "pactl not found — install pulseaudio-utils and run in the desktop session."
        )
        return 2
    import shutil

    if shutil.which("parec") is None:
        logger.error("parec not found — install pulseaudio-utils (it ships parec).")
        return 2

    sample_rate = settings.SAMPLE_RATE
    try:
        audio = capture_pcm(source, seconds)
    except Exception as exc:  # noqa: BLE001 — self-diagnosing
        logger.error("capture failed: %s", exc)
        logger.error(
            "Check: is '%s' a valid source (--list)? Is audio playing to it?", source
        )
        return 2

    peak = float(np.abs(audio).max()) if audio.size else 0.0
    logger.info("captured %d samples, peak amplitude=%.4f", audio.size, peak)
    if peak < 1e-4:
        logger.warning(
            "captured near-silence (peak=%.6f) — is audio actually routing to '%s'? "
            "Play/say something during the capture window.",
            peak,
            source,
        )

    # End-to-end latency = capture-stop -> transcript-ready.
    from scripts.stt import Transcriber

    transcriber = Transcriber()
    t0 = time.perf_counter()
    result = transcriber.transcribe_array(audio, sample_rate=sample_rate)
    e2e = time.perf_counter() - t0

    print("\n=== SPIKE RESULT ===")
    print(f"source           : {source}")
    print(f"captured audio   : {result.audio_seconds:.2f}s")
    print(f"transcript       : {result.text!r}")
    print(
        f"decode latency   : {result.latency_seconds:.2f}s (rtf={result.realtime_factor:.2f})"
    )
    print(f"end-to-end (post-capture) : {e2e:.2f}s")
    return 0


def loopback_selftest(sink: str, wav: str, seconds: float) -> int:
    """Prove the monitor->STT path with no human present.

    Plays a known speech WAV into `sink` while capturing `sink`.monitor, then
    transcribes. This exercises the exact code path a real Teams call uses — the
    only thing it does not prove is that Teams is routed to that sink.
    """
    import shutil

    if shutil.which("paplay") is None:
        logger.error("paplay not found — install pulseaudio-utils.")
        return 2
    if not Path(wav).is_file():
        logger.error("self-test WAV not found: %s", wav)
        return 2

    source = f"{sink}.monitor"
    logger.info(
        "loopback self-test: playing %s -> sink %r, capturing %r", wav, sink, source
    )
    player = subprocess.Popen(
        ["paplay", f"--device={sink}", wav],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        rc = capture_and_transcribe(source, seconds)
    finally:
        if player.poll() is None:
            player.terminate()
    return rc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--list",
        action="store_true",
        help="list audio devices + pactl sources and exit",
    )
    parser.add_argument("--source", help="pactl source name to capture (see --list)")
    parser.add_argument(
        "--seconds", type=float, default=8.0, help="capture duration (default 8s)"
    )
    parser.add_argument(
        "--selftest-sink",
        help="loopback self-test: play --selftest-wav into this SINK and capture its .monitor",
    )
    parser.add_argument(
        "--selftest-wav",
        default=str(
            Path(__file__).resolve().parents[1] / "scripts/outputs/piper_test.wav"
        ),
        help="speech WAV used by the loopback self-test",
    )
    args = parser.parse_args()

    if args.selftest_sink:
        sys.exit(loopback_selftest(args.selftest_sink, args.selftest_wav, args.seconds))

    if args.list or not args.source:
        list_sources()
        if not args.list:
            print(
                "\nNo --source given. Re-run with --source <name> to capture+transcribe."
            )
        sys.exit(0)

    sys.exit(capture_and_transcribe(args.source, args.seconds))


if __name__ == "__main__":
    main()
