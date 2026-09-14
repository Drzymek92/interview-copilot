"""Shared pytest fixtures for interview_copilot.

Currently minimal: each test module sets up what it needs (a tmp bundle, a fake runner, the
shipped example bundle). Add cross-module fixtures here as they earn their place.
"""
from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to tests/fixtures/ (question set for the trigger measurement)."""
    return Path(__file__).parent / "fixtures"
