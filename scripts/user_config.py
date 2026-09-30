"""The UI-editable settings registry + the saved layer (D36).

A **whitelist**: only the knobs named in `KNOBS` can be changed from the hub's Settings view or a
mode's ⚙ drawer. Saved values go to the gitignored `config/user_settings.json`, keyed by the
environment-variable name `config/settings.py` reads, and are folded in **below env** — the house
CFG ladder with the file as its config rung: **CLI > env > saved > default**.

Provenance per knob, as the UI shows it:
  default — neither the environment nor the saved file sets it
  saved   — the saved file sets it (and the environment does not)
  env     — the user's own environment sets it: LOCKED, the saved value cannot win (D36)

Applying a change = write the file (temp + rename), then `importlib.reload(settings)`: every module
reads `settings.X` at use time, so the next mode start sees the new value, and recorder children
inherit it through the exported env. Host/port and credentials are deliberately absent (SI1).
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402

GROUPS: dict[str, str] = {
    "general": "General",
    "call": "Call",
    "train": "Training",
    "review": "Review",
    "generate": "Generate context",
}


@dataclass(frozen=True)
class Knob:
    env: str                    # the environment-variable name settings.py reads (= file key)
    attr: str                   # the settings.py attribute it lands in
    label: str
    kind: str                   # "bool" | "int" | "float" | "str" | "choice"
    groups: tuple[str, ...]
    default: str                # the settings.py literal default, as its env string
    help: str = ""
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None


KNOBS: tuple[Knob, ...] = (
    # -- general: devices + models + transcription ------------------------------------------
    Knob("COPILOT_MIC", "COPILOT_MIC", "Microphone (your voice)", "str", ("general",), "auto",
         "PulseAudio/PipeWire source name. 'auto' picks the WEBCAM mic on lab — set your external mic for a real call."),
    Knob("COPILOT_SOURCE", "COPILOT_SOURCE", "Interviewer audio (Teams output monitor)", "str", ("general",), "auto",
         "The monitor of the sink Teams plays to."),
    Knob("OLLAMA_MODEL", "LOCAL_MODEL", "Local model", "str", ("general",), "interview-copilot:14b",
         "The Ollama model used by every local call."),
    Knob("CLOUD_MODEL", "CLOUD_MODEL", "Cloud model", "str", ("general",), "",
         "Empty = the provider default. Used only when a mode's backend is set to cloud (opt-in, announced)."),
    Knob("STT_MODEL", "STT_MODEL", "Speech-to-text model", "choice", ("general",), "large-v3-turbo",
         "Multilingual Whisper only — distil models are English-only.", ("large-v3-turbo", "large-v3")),
    Knob("STT_LANGUAGE", "STT_LANGUAGE", "Main spoken language", "choice", ("general",), "pl",
         "The fallback / forced language of the conversation.", ("pl", "en")),
    Knob("STT_DETECT_LANGUAGE", "STT_DETECT_LANGUAGE", "Detect language per segment", "bool", ("general",), "1",
         "Off = force the main spoken language on every segment."),
    Knob("COPILOT_OPEN_BROWSER", "COPILOT_OPEN_BROWSER", "Open the browser on launch", "bool", ("general",), "1"),
    # -- call ----------------------------------------------------------------------------------
    Knob("REASONING_BACKEND", "REASONING_BACKEND", "Answers backend at start", "choice", ("call",), "local",
         "Cloud sends the question + context to your BYOK provider (announced). Flippable live in the call.",
         ("local", "cloud")),
    Knob("SUGGESTION_LANGUAGE", "SUGGESTION_LANGUAGE", "Suggestion language", "choice", ("call",), "match",
         "'match' answers in the language the question was asked in.", ("match", "en", "pl")),
    Knob("ANSWER_SPEAKER", "ANSWER_SPEAKER", "Whose turns get suggestions", "choice", ("call",), "them",
         "'them' = the interviewer only.", ("them", "you", "any")),
    Knob("SALIENCE_GATE_ENABLED", "SALIENCE_GATE_ENABLED", "Salience gate (D23)", "bool", ("call",), "1",
         "Off = every detected question fires a suggestion (noisier)."),
    Knob("SUGGESTION_COOLDOWN_SECONDS", "SUGGESTION_COOLDOWN_SECONDS", "Cooldown between suggestions (s)", "float",
         ("call",), "8.0", minimum=0, maximum=120),
    Knob("SUGGESTION_MAX_TOKENS", "SUGGESTION_MAX_TOKENS", "Suggestion length cap (tokens)", "int", ("call",), "220",
         minimum=60, maximum=1000),
    Knob("SUGGESTION_STALE_LINES", "SUGGESTION_STALE_LINES", "Mark a suggestion stale after N lines", "int",
         ("call",), "2", minimum=1, maximum=20),
    Knob("DASHBOARD_SHOW_PARTIALS", "DASHBOARD_SHOW_PARTIALS", "Show provisional (still-spoken) text", "bool",
         ("call", "train"), "1"),
    Knob("SEGMENT_MAX_SECONDS", "SEGMENT_MAX_SECONDS", "Force-cut a segment after (s)", "float", ("call",), "30.0",
         "D28: 30 s stays the default; lower only knowingly (accuracy trade).", minimum=10, maximum=60),
    # -- train ---------------------------------------------------------------------------------
    Knob("TRAINING_QUESTION_COUNT", "TRAINING_QUESTION_COUNT", "Questions per practice run", "int", ("train",), "6",
         minimum=1, maximum=30),
    Knob("SCORING_BACKEND", "SCORING_BACKEND", "Scoring backend", "choice", ("train", "review"), "local",
         "Cloud sends your answers + CV to your BYOK provider (announced).", ("local", "cloud")),
    Knob("SCORING_MODEL", "SCORING_MODEL", "Scoring model", "str", ("train", "review"), "",
         "Empty = the backend's default model."),
    Knob("SCORING_TIMEOUT_SECONDS", "SCORING_TIMEOUT_SECONDS", "Scoring timeout (s)", "float", ("train", "review"),
         "180", minimum=10, maximum=1200),
    # -- review --------------------------------------------------------------------------------
    Knob("REVIEW_SYNTHESIZE_RUBRICS", "REVIEW_SYNTHESIZE_RUBRICS", "Write a rubric per question first", "bool",
         ("review",), "0", "Slower (one extra model call per question); off = holistic grading."),
    # -- generate ------------------------------------------------------------------------------
    Knob("GENERATE_BACKEND", "GENERATE_BACKEND", "Generation backend", "choice", ("generate",), "local",
         "Cloud sends the job description to your BYOK provider (announced).", ("local", "cloud")),
    Knob("GENERATE_MODEL", "GENERATE_MODEL", "Generation model", "str", ("generate",), "",
         "Empty = the backend's default model."),
    Knob("GENERATE_QUESTIONS_PER_COMPETENCY", "GENERATE_QUESTIONS_PER_COMPETENCY", "Questions per competency",
         "int", ("generate",), "3", minimum=1, maximum=10),
    Knob("GENERATE_TIMEOUT_SECONDS", "GENERATE_TIMEOUT_SECONDS", "Generation timeout (s)", "float", ("generate",),
         "480", minimum=30, maximum=1800),
)
BY_ENV: dict[str, Knob] = {k.env: k for k in KNOBS}


class ConfigError(ValueError):
    """A rejected value — surfaced to the UI as a 400 with this message."""


def normalize(knob: Knob, value: object) -> str:
    """Validate one incoming value and return it as the env string settings.py parses."""
    if knob.kind == "bool":
        if isinstance(value, bool):
            return "1" if value else "0"
        text = str(value).strip().lower()
        if text in ("1", "true", "on", "yes"):
            return "1"
        if text in ("0", "false", "off", "no"):
            return "0"
        raise ConfigError(f"{knob.label}: expected on/off, got {value!r}")
    text = str(value).strip()
    if knob.kind in ("int", "float"):
        try:
            number = int(text) if knob.kind == "int" else float(text)
        except ValueError:
            raise ConfigError(f"{knob.label}: expected a{'n integer' if knob.kind == 'int' else ' number'}, "
                              f"got {value!r}") from None
        if knob.minimum is not None and number < knob.minimum:
            raise ConfigError(f"{knob.label}: must be ≥ {knob.minimum:g}")
        if knob.maximum is not None and number > knob.maximum:
            raise ConfigError(f"{knob.label}: must be ≤ {knob.maximum:g}")
        return str(number)
    if knob.kind == "choice" and text not in knob.choices:
        raise ConfigError(f"{knob.label}: must be one of {', '.join(knob.choices)}")
    if "\n" in text or len(text) > 200:
        raise ConfigError(f"{knob.label}: one short line only")
    return text


def _display(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return "" if value is None else str(value)


def provenance(knob: Knob, saved: dict[str, str] | None = None) -> str:
    if knob.env in settings.ORIGINAL_ENV_KEYS:
        return "env"
    saved = settings.load_saved_settings() if saved is None else saved
    return "saved" if knob.env in saved else "default"


def describe(group: str | None = None) -> list[dict]:
    """The rows a settings panel renders: current value, default, provenance, constraints."""
    saved = settings.load_saved_settings()
    rows = []
    for knob in KNOBS:
        if group is not None and group not in knob.groups:
            continue
        rows.append({
            "env": knob.env, "label": knob.label, "kind": knob.kind, "help": knob.help,
            "groups": list(knob.groups), "choices": list(knob.choices),
            "min": knob.minimum, "max": knob.maximum, "default": knob.default,
            "value": _display(getattr(settings, knob.attr, knob.default)),
            "source": provenance(knob, saved),
        })
    return rows


def _write(saved: dict[str, str]) -> None:
    path = settings.USER_SETTINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(sorted(saved.items())), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _reload() -> None:
    importlib.reload(settings)


def save(updates: dict[str, object]) -> list[dict]:
    """Validate every update first (all-or-nothing), then write + apply. A value equal to the
    default is stored as a removal, so the file only ever holds real deviations."""
    unknown = [name for name in updates if name not in BY_ENV]
    if unknown:
        raise ConfigError(f"not a UI-editable setting: {', '.join(sorted(unknown))}")
    normalized = {name: normalize(BY_ENV[name], value) for name, value in updates.items()}
    locked = [BY_ENV[n].label for n in normalized if BY_ENV[n].env in settings.ORIGINAL_ENV_KEYS]
    if locked:
        raise ConfigError("set by your environment, so a saved value would never apply: "
                          + ", ".join(locked))
    saved = {k: v for k, v in settings.load_saved_settings().items() if k in BY_ENV}
    for name, value in normalized.items():
        if value == normalize(BY_ENV[name], BY_ENV[name].default):
            saved.pop(name, None)
        else:
            saved[name] = value
    _write(saved)
    _reload()
    return describe()


def reset(names: list[str] | None = None) -> list[dict]:
    """Drop saved values (all of them when `names` is None) and re-apply."""
    saved = {k: v for k, v in settings.load_saved_settings().items() if k in BY_ENV}
    for name in (list(saved) if names is None else names):
        saved.pop(name, None)
    _write(saved)
    _reload()
    return describe()
