"""D36 — the UI-saved settings layer: whitelist, validation, precedence, provenance."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
import user_config as uc  # noqa: E402


@pytest.fixture
def saved_file(tmp_path, monkeypatch):
    """Point the saved layer at a temp file, with no knob in the 'original' environment."""
    path = tmp_path / "user_settings.json"
    monkeypatch.setattr(settings, "USER_SETTINGS_FILE", path)
    clean = frozenset(k for k in settings.ORIGINAL_ENV_KEYS if k not in uc.BY_ENV)
    monkeypatch.setattr(settings, "ORIGINAL_ENV_KEYS", clean)
    for knob in uc.KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    # reload() would re-read COPILOT_USER_SETTINGS; keep it pointed at the temp file
    monkeypatch.setenv("COPILOT_USER_SETTINGS", str(path))
    yield path
    for name in list(settings._INJECTED):
        os.environ.pop(name, None)
    settings._INJECTED.clear()
    monkeypatch.undo()
    importlib.reload(settings)


def test_registry_defaults_match_settings_literals(saved_file):
    importlib.reload(settings)
    for knob in uc.KNOBS:
        current = uc._display(getattr(settings, knob.attr))
        assert uc.normalize(knob, current) == uc.normalize(knob, knob.default), knob.env


def test_registry_never_exposes_the_bind_or_credentials():
    names = set(uc.BY_ENV)
    assert not names & {"DASHBOARD_HOST", "DASHBOARD_PORT", "COPILOT_USER_SETTINGS"}
    assert not any(n.endswith(settings._CREDENTIAL_SUFFIXES) for n in names)


def test_saved_file_cannot_set_the_bind(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"DASHBOARD_HOST": "0.0.0.0", "OPENAI_API_KEY": "x",
                                "TRAINING_QUESTION_COUNT": "3", "SUGGESTION_MAX_TOKENS": "300"}))
    assert settings.load_saved_settings(path) == {"TRAINING_QUESTION_COUNT": "3",
                                                  "SUGGESTION_MAX_TOKENS": "300"}


def test_corrupt_saved_file_is_an_empty_layer(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("{not json")
    assert settings.load_saved_settings(path) == {}


def test_save_applies_and_reports_saved(saved_file):
    rows = {r["env"]: r for r in uc.save({"TRAINING_QUESTION_COUNT": 9})}
    assert settings.TRAINING_QUESTION_COUNT == 9
    assert rows["TRAINING_QUESTION_COUNT"]["source"] == "saved"
    assert json.loads(saved_file.read_text()) == {"TRAINING_QUESTION_COUNT": "9"}


def test_saving_the_default_removes_the_entry(saved_file):
    uc.save({"TRAINING_QUESTION_COUNT": 9})
    uc.save({"TRAINING_QUESTION_COUNT": 6})
    assert json.loads(saved_file.read_text()) == {}
    assert settings.TRAINING_QUESTION_COUNT == 6


def test_reset_drops_saved_values(saved_file):
    uc.save({"SALIENCE_GATE_ENABLED": False, "TRAINING_QUESTION_COUNT": 4})
    assert settings.SALIENCE_GATE_ENABLED is False
    uc.reset(["SALIENCE_GATE_ENABLED"])
    assert settings.SALIENCE_GATE_ENABLED is True and settings.TRAINING_QUESTION_COUNT == 4
    uc.reset()
    assert settings.TRAINING_QUESTION_COUNT == 6


def test_env_wins_and_is_locked(saved_file, monkeypatch):
    monkeypatch.setenv("TRAINING_QUESTION_COUNT", "2")
    monkeypatch.setattr(settings, "ORIGINAL_ENV_KEYS",
                        settings.ORIGINAL_ENV_KEYS | {"TRAINING_QUESTION_COUNT"})
    saved_file.write_text(json.dumps({"TRAINING_QUESTION_COUNT": "9"}))
    importlib.reload(settings)
    assert settings.TRAINING_QUESTION_COUNT == 2
    row = next(r for r in uc.describe("train") if r["env"] == "TRAINING_QUESTION_COUNT")
    assert row["source"] == "env"
    with pytest.raises(uc.ConfigError, match="environment"):
        uc.save({"TRAINING_QUESTION_COUNT": 5})


@pytest.mark.parametrize("env,value", [
    ("TRAINING_QUESTION_COUNT", "0"),        # below minimum
    ("TRAINING_QUESTION_COUNT", "many"),     # not a number
    ("REASONING_BACKEND", "gpt"),            # not a choice
    ("SALIENCE_GATE_ENABLED", "maybe"),      # not a bool
    ("COPILOT_MIC", "a\nb"),                 # multi-line
])
def test_invalid_values_are_rejected(saved_file, env, value):
    with pytest.raises(uc.ConfigError):
        uc.save({env: value})
    assert not saved_file.exists()


def test_unknown_setting_is_rejected_all_or_nothing(saved_file):
    with pytest.raises(uc.ConfigError, match="not a UI-editable"):
        uc.save({"TRAINING_QUESTION_COUNT": 3, "DASHBOARD_HOST": "0.0.0.0"})
    assert not saved_file.exists()


def test_describe_filters_by_group():
    groups = {r["env"] for r in uc.describe("generate")}
    assert "GENERATE_BACKEND" in groups and "COPILOT_MIC" not in groups
    assert {r["env"] for r in uc.describe()} == set(uc.BY_ENV)


def test_reload_ignores_dotenv_only_keys(saved_file, monkeypatch):
    """A settings save reloads the module; values that only config/.env supplied (loaded by llm_client
    AFTER the first import) must not suddenly apply — a fresh process never saw them in settings."""
    importlib.reload(settings)
    before = settings.LOCAL_MODEL
    monkeypatch.setenv("OLLAMA_MODEL", "some-old-dotenv-model:8b")
    monkeypatch.setattr(settings, "DOTENV_KEYS", {"OLLAMA_MODEL"})
    uc.save({"TRAINING_QUESTION_COUNT": 5})               # triggers the reload
    assert settings.LOCAL_MODEL == before
    uc.save({"OLLAMA_MODEL": "picked-in-ui:14b"})          # ...but a value saved in the UI does apply
    assert settings.LOCAL_MODEL == "picked-in-ui:14b"
    assert "OLLAMA_MODEL" not in settings.DOTENV_KEYS
