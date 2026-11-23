"""scripts/ollama_keepalive.py — the Windows launcher's model-warm pinger (no network, no model)."""

from __future__ import annotations

import json
import socket
import urllib.error

from scripts import ollama_keepalive as ka


def test_native_root_strips_the_openai_suffix() -> None:
    assert ka.native_root("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434"
    assert ka.native_root("http://127.0.0.1:11434/v1/") == "http://127.0.0.1:11434"
    assert ka.native_root("http://host:1") == "http://host:1"


def test_request_refreshes_keep_alive_without_a_prompt() -> None:
    req = ka.build_request("http://127.0.0.1:11434", "m:8b")
    assert req.full_url == "http://127.0.0.1:11434/api/generate"
    assert req.get_method() == "POST"
    assert json.loads(req.data) == {"model": "m:8b", "keep_alive": "3h"}


def test_ping_never_raises_when_ollama_is_down(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(ka.urllib.request, "urlopen", boom)
    assert ka.ping("http://127.0.0.1:9", "m") is False


def test_resolve_target_defaults_to_ipv4(monkeypatch) -> None:
    from scripts import llm_client   # imported first: its .env load must not decide this test

    monkeypatch.setattr(llm_client, "_cfg", lambda _name, default: default)
    root, model = ka.resolve_target()
    assert root == "http://127.0.0.1:11434"
    assert model


def test_resolve_target_follows_ollama_base_url(monkeypatch) -> None:
    from scripts import llm_client  # noqa: F401 — load its .env first; the env var set after wins

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://10.0.0.5:11434/v1")
    root, _model = ka.resolve_target()
    assert root == "http://10.0.0.5:11434"


def test_hub_watch_stops_after_the_hub_goes_away() -> None:
    states = iter([False, True, True, False])
    watch = ka.HubWatch(lambda: next(states), wait=100, clock=lambda: 0.0)
    assert [watch.should_stop() for _ in range(4)] == [False, False, False, True]


def test_hub_watch_gives_up_if_the_hub_never_starts() -> None:
    now = {"t": 0.0}
    watch = ka.HubWatch(lambda: False, wait=10, clock=lambda: now["t"])
    assert watch.should_stop() is False
    now["t"] = 11.0
    assert watch.should_stop() is True


def test_run_pings_on_interval_and_returns_when_the_hub_stops(monkeypatch) -> None:
    pings: list[str] = []
    monkeypatch.setattr(ka, "ping", lambda root, model: pings.append(model) or True)
    ups = iter([True] * 30 + [False])
    watch = ka.HubWatch(lambda: next(ups), wait=100, clock=lambda: 0.0)
    ka.run("http://x", "m", interval=120, watch=watch, sleep=lambda _s: None, poll=5)
    # 30 polls x 5 s = 150 s of hub uptime -> pings at t=0 and t=120
    assert pings == ["m", "m"]


def test_hub_up_detects_a_listening_socket() -> None:
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert ka.hub_up("127.0.0.1", port) is True
    finally:
        srv.close()
    assert ka.hub_up("127.0.0.1", port) is False
