"""Shared pytest configuration for interview_copilot.

Intentionally minimal: each test module sets up what it needs (a tmp bundle, a fake runner, or the
shipped example bundle). Add cross-module fixtures here only when more than one module needs them.
"""
import os

# D36: tests must never read the user's real saved settings (config/user_settings.json).
os.environ.setdefault("COPILOT_USER_SETTINGS", "/nonexistent/interview_copilot_user_settings.json")

