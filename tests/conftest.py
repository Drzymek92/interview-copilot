"""Shared pytest configuration for interview_copilot.

Intentionally minimal: each test module sets up what it needs (a tmp bundle, a fake runner, or the
shipped example bundle). Add cross-module fixtures here only when more than one module needs them.
"""
