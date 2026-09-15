"""Pluggable audio-capture backend — so `live_transcribe.py` runs on Linux, Windows and macOS.

The rest of the pipeline consumes fixed-size **int16 mono PCM frames at `settings.SAMPLE_RATE`
(16 kHz)** and nothing else. A backend's whole job is to produce those frames from two sources —
the *monitor* (the other person's voice = system/loopback audio) and the *mic* — and to enumerate
what sources exist. Everything above this file (VAD, segmentation, STT) is platform-agnostic.

    Linux    `parec` on a PulseAudio/PipeWire monitor source (the original, best-tested path).
    Windows  sounddevice/PortAudio: WASAPI **loopback** of an output device for the monitor,
             a normal input device for the mic. No extra software needed.
    macOS    sounddevice/PortAudio input devices. macOS has **no built-in loopback**, so the
             monitor must be a virtual audio device (e.g. BlackHole) that the user routes the
             call's audio into; it then appears as a normal input device here.

Select with `settings.AUDIO_BACKEND` / `COPILOT_AUDIO_BACKEND` (auto | parec | sounddevice).

NOTE: the sounddevice path is implemented against the documented PortAudio/WASAPI APIs but has
been unit-tested (format/mixdown/resample/framing) rather than hardware-verified on Windows/macOS
— see tests/test_audio_backend.py. Report issues with real capture on those platforms.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from config import settings
from scripts.logger import get_logger

logger = get_logger("audio_backend")

# A source addressed with this prefix is captured as WASAPI loopback of the named OUTPUT device
# (Windows): what that device is *playing* becomes the capture. Stripped before device resolution.
LOOPBACK_PREFIX = "loopback:"


class CaptureStream(Protocol):
    """One capture channel. `read()` returns exactly `frame_bytes` of s16le mono @ 16 kHz; it must
    never block forever and never raise — a dead stream degrades to silence (a tool that dies
    mid-call is worse than one that records half a conversation)."""

    dead: bool

    def read(self) -> bytes: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------
# Frame math shared by the non-parec backends
# --------------------------------------------------------------------------
def to_mono_16k(pcm_int16: np.ndarray, src_rate: int, channels: int,
                target_rate: int = settings.SAMPLE_RATE) -> np.ndarray:
    """Down-mix interleaved int16 to mono and linearly resample to `target_rate`.

    Per-chunk linear resampling (no cross-chunk state) is deliberately simple: for 16 kHz speech
    STT the edge artefacts at ~100 ms chunk boundaries are inaudible to Whisper, and it adds no
    dependency beyond numpy. Callers should feed reasonably large chunks (≥ ~50 ms)."""
    if pcm_int16.size == 0:
        return np.zeros(0, dtype=np.int16)
    frames = pcm_int16.reshape(-1, channels).astype(np.float32)
    mono = frames.mean(axis=1)
    if src_rate != target_rate and mono.size > 1:
        n_out = max(1, int(round(mono.size * target_rate / src_rate)))
        idx = np.linspace(0.0, mono.size - 1, n_out)
        mono = np.interp(idx, np.arange(mono.size, dtype=np.float32), mono)
    return np.clip(np.rint(mono), -32768, 32767).astype(np.int16)


class _FrameBuffer:
    """Accumulates variable-size byte chunks and hands out fixed `frame_bytes` frames."""

    def __init__(self, frame_bytes: int) -> None:
        self.frame_bytes = frame_bytes
        self._buf = bytearray()
        self._lock = threading.Lock()

    def push(self, data: bytes) -> None:
        with self._lock:
            self._buf.extend(data)

    def pop_frame(self) -> bytes | None:
        with self._lock:
            if len(self._buf) < self.frame_bytes:
                return None
            frame = bytes(self._buf[: self.frame_bytes])
            del self._buf[: self.frame_bytes]
            return frame


@dataclass
class Source:
    """One enumerable capture source."""

    name: str
    kind: str  # "monitor" | "mic" | "output" | "input"
    detail: str = ""


# --------------------------------------------------------------------------
# Backend interface
# --------------------------------------------------------------------------
class AudioBackend(Protocol):
    name: str
    supports_playback: bool

    def available(self) -> tuple[bool, str]: ...

    def source_names(self) -> list[str]: ...

    def default_monitor(self) -> str | None: ...

    def default_mic(self) -> str | None: ...

    def print_sources(self) -> None: ...

    def open_stream(self, source: str, frame_bytes: int, label: str) -> CaptureStream: ...

    def play_into_sink(self, sink: str, wav: str) -> subprocess.Popen | None: ...


# --------------------------------------------------------------------------
# Linux — parec / pactl (the original path, moved here unchanged)
# --------------------------------------------------------------------------
class ParecStream:
    """One `parec` capture stream. Never fatal: a dead stream degrades to silence."""

    def __init__(self, source: str, frame_bytes: int, label: str) -> None:
        self.source = source
        self.label = label
        self.frame_bytes = frame_bytes
        self.dead = False
        cmd = [
            "parec", f"--device={source}", "--format=s16le",
            f"--rate={settings.SAMPLE_RATE}", "--channels=1", "--raw",
        ]
        logger.info("%s: parec --device=%s", label, source)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def read(self) -> bytes:
        if self.dead:
            return b"\x00" * self.frame_bytes
        assert self.proc.stdout is not None
        data = self.proc.stdout.read(self.frame_bytes)
        if not data or len(data) < self.frame_bytes:
            err = (self.proc.stderr.read().decode(errors="replace") if self.proc.stderr else "").strip()
            logger.warning("%s: stream ended (%s) — continuing with silence", self.label, err or "eof")
            self.dead = True
            return b"\x00" * self.frame_bytes
        return data

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                logger.warning("%s: parec ignored SIGTERM — killing", self.label)
                self.proc.kill()
                self.proc.wait(timeout=3)
        for pipe in (self.proc.stdout, self.proc.stderr):
            if pipe is not None:
                pipe.close()
        logger.info("%s: parec stopped (rc=%s)", self.label, self.proc.returncode)


class ParecBackend:
    name = "parec"
    supports_playback = True

    def available(self) -> tuple[bool, str]:
        if shutil.which("parec") is None or shutil.which("pactl") is None:
            return False, "parec/pactl not found — install pulseaudio-utils (Linux only)"
        return True, ""

    def _pactl(self, *args: str) -> str:
        return subprocess.run(
            ["pactl", *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    def source_names(self) -> list[str]:
        return [ln.split("\t")[1]
                for ln in self._pactl("list", "short", "sources").splitlines() if "\t" in ln]

    def default_monitor(self) -> str | None:
        try:
            sink = self._pactl("get-default-sink")
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        monitor = f"{sink}.monitor"
        return monitor if monitor in self.source_names() else None

    def default_mic(self) -> str | None:
        try:
            src = self._pactl("get-default-source")
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        if src.endswith(".monitor") or src not in self.source_names():
            return None
        return src

    def print_sources(self) -> None:
        ok, hint = self.available()
        if not ok:
            print(hint)
            return
        print("=== capture sources (pactl) ===")
        for line in self._pactl("list", "short", "sources").splitlines():
            tag = "   <-- MONITOR (the other person's voice)" if ".monitor" in line else ""
            print(f"  {line}{tag}")
        print(f"\nauto --source : {self.default_monitor()}")
        print(f"auto --mic    : {self.default_mic()}")

    def open_stream(self, source: str, frame_bytes: int, label: str) -> CaptureStream:
        return ParecStream(source, frame_bytes, label)

    def play_into_sink(self, sink: str, wav: str) -> subprocess.Popen | None:
        return subprocess.Popen(
            ["paplay", f"--device={sink}", wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )


# --------------------------------------------------------------------------
# Windows / macOS — sounddevice (PortAudio)
# --------------------------------------------------------------------------
_LOOPBACK_HINTS = ("blackhole", "loopback", "aggregate", "soundflower", "vb-audio", "cable")


def _import_sounddevice():
    import sounddevice as sd  # lazy: importing needs the native PortAudio lib present
    return sd


class SoundDeviceStream:
    """A PortAudio input (or WASAPI-loopback) stream normalised to s16le mono @ 16 kHz frames."""

    def __init__(self, source: str, frame_bytes: int, label: str) -> None:
        sd = _import_sounddevice()
        self.label = label
        self.frame_bytes = frame_bytes
        self.dead = False
        self._buf = _FrameBuffer(frame_bytes)
        self._silence = b"\x00" * frame_bytes

        loopback = source.startswith(LOOPBACK_PREFIX)
        dev_ref = source[len(LOOPBACK_PREFIX):] if loopback else source
        device = _resolve_device(sd, dev_ref, want_output=loopback)
        info = sd.query_devices(device)
        self._src_rate = int(info["default_samplerate"])
        self._channels = int(info["max_output_channels" if loopback else "max_input_channels"]) or 1

        extra = None
        if loopback:
            try:
                extra = sd.WasapiSettings(loopback=True)  # Windows WASAPI: capture what it plays
            except Exception:  # noqa: BLE001 — non-WASAPI host has no loopback settings
                logger.warning("%s: WASAPI loopback unavailable on this host API; opening as input", label)

        def _callback(indata, _frames, _time, status) -> None:
            if status:
                logger.debug("%s: sounddevice status %s", label, status)
            pcm = np.frombuffer(bytes(indata), dtype=np.int16)
            self._buf.push(to_mono_16k(pcm, self._src_rate, self._channels).tobytes())

        # ~100 ms blocks keep per-chunk resampling artefacts negligible.
        blocksize = max(256, int(self._src_rate * 0.1))
        logger.info("%s: sounddevice device=%r rate=%d ch=%d loopback=%s",
                    label, device, self._src_rate, self._channels, loopback)
        try:
            self._stream = sd.RawInputStream(
                samplerate=self._src_rate, blocksize=blocksize, device=device,
                channels=self._channels, dtype="int16", callback=_callback,
                extra_settings=extra,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001 — any open failure degrades to silence, never fatal
            logger.error("%s: could not open capture stream (%s) — continuing with silence", label, exc)
            self._stream = None
            self.dead = True

    def read(self) -> bytes:
        if self.dead:
            return self._silence
        # Wait up to ~1 s of frames for enough audio; underrun → silence (keep the call alive).
        for _ in range(50):
            frame = self._buf.pop_frame()
            if frame is not None:
                return frame
            time.sleep(0.02)
        return self._silence

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: error closing stream: %s", self.label, exc)
        logger.info("%s: sounddevice stopped", self.label)


def _resolve_device(sd, ref: str, want_output: bool):
    """Resolve a device reference (index, or case-insensitive name substring) to a PortAudio id."""
    if ref in ("", "auto", "default"):
        idx = sd.default.device[1 if want_output else 0]
        if idx is None or idx < 0:
            raise RuntimeError("no default audio device")
        return idx
    if ref.isdigit():
        return int(ref)
    key = "max_output_channels" if want_output else "max_input_channels"
    low = ref.lower()
    for i, d in enumerate(sd.query_devices()):
        if low in d["name"].lower() and d[key] > 0:
            return i
    raise RuntimeError(f"no {'output' if want_output else 'input'} device matching {ref!r}")


class SoundDeviceBackend:
    name = "sounddevice"
    supports_playback = False

    def available(self) -> tuple[bool, str]:
        try:
            _import_sounddevice()
        except Exception as exc:  # noqa: BLE001
            return False, (f"sounddevice/PortAudio not available ({exc}). "
                           "pip install sounddevice; on Linux also install libportaudio2.")
        return True, ""

    def _devices(self):
        sd = _import_sounddevice()
        return sd, list(sd.query_devices())

    def source_names(self) -> list[str]:
        _sd, devs = self._devices()
        return [d["name"] for d in devs]

    def default_monitor(self) -> str | None:
        sd, devs = self._devices()
        # macOS: a virtual loopback device shows up as an INPUT — prefer one by name.
        for d in devs:
            if d["max_input_channels"] > 0 and any(h in d["name"].lower() for h in _LOOPBACK_HINTS):
                return d["name"]
        # Windows: loopback-capture the default OUTPUT device.
        if sys.platform == "win32":
            out = sd.default.device[1]
            if out is not None and out >= 0:
                return LOOPBACK_PREFIX + devs[out]["name"]
        return None

    def default_mic(self) -> str | None:
        sd, devs = self._devices()
        idx = sd.default.device[0]
        if idx is None or idx < 0:
            return None
        name = devs[idx]["name"]
        if any(h in name.lower() for h in _LOOPBACK_HINTS):
            return None  # the default input IS a loopback — not a real mic
        return name

    def print_sources(self) -> None:
        ok, hint = self.available()
        if not ok:
            print(hint)
            return
        _sd, devs = self._devices()
        print("=== capture devices (sounddevice / PortAudio) ===")
        for i, d in enumerate(devs):
            ins, outs = d["max_input_channels"], d["max_output_channels"]
            role = []
            if ins > 0:
                role.append("input")
            if outs > 0:
                role.append("output")
            loop = "  <-- looks like a loopback device" if any(
                h in d["name"].lower() for h in _LOOPBACK_HINTS) else ""
            print(f"  [{i}] {d['name']}  ({'/'.join(role)}){loop}")
        print(f"\nauto --source : {self.default_monitor()}")
        print(f"auto --mic    : {self.default_mic()}")
        if sys.platform == "darwin" and self.default_monitor() is None:
            print("\nmacOS has no built-in loopback: install BlackHole (or similar), route the call's\n"
                  "audio into it, then pass its device name as --source. See SETUP.md.")

    def open_stream(self, source: str, frame_bytes: int, label: str) -> CaptureStream:
        return SoundDeviceStream(source, frame_bytes, label)

    def play_into_sink(self, sink: str, wav: str) -> subprocess.Popen | None:
        logger.warning("--selftest-sink playback is only supported on the parec (Linux) backend")
        return None


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def select_backend(name: str = "auto") -> AudioBackend:
    """Pick a backend by name, or auto-detect from the platform."""
    name = (name or "auto").lower()
    if name == "parec":
        return ParecBackend()
    if name == "sounddevice":
        return SoundDeviceBackend()
    if name != "auto":
        logger.warning("unknown AUDIO_BACKEND %r — falling back to auto", name)
    return ParecBackend() if sys.platform.startswith("linux") else SoundDeviceBackend()
