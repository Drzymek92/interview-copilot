"""Keep the local Ollama model resident while the copilot runs (the Windows launcher's pinger).

Ollama unloads a model after 5 idle minutes, and the OpenAI-compatible endpoint the client uses
cannot pass `keep_alive`, so a quiet gap mid-interview makes the next question pay a cold load +
prefill (measured: ~10 s to the first token cold vs ~0.16 s warm). This loop refreshes the timer
through Ollama's native API every `--interval` seconds: an empty prompt only loads/refreshes the
model, it generates nothing.

`scripts/launch_copilot.sh` does the same with a curl loop; `scripts/launch_copilot.bat` starts
this script in the background instead, so no shell quoting is needed on Windows. The model and
endpoint are resolved exactly as the app resolves them (env > saved Settings > default for the
model; `OLLAMA_BASE_URL` from the environment or `config/.env`).

    python scripts/ollama_keepalive.py                 # loop until Ctrl-C / the console closes
    python scripts/ollama_keepalive.py --follow-hub    # ...or until the hub stops (the launcher)
    python scripts/ollama_keepalive.py --once          # one ping, exit 0 if Ollama answered

`--follow-hub` makes the pinger exit on its own once the hub (DASHBOARD_HOST:DASHBOARD_PORT) has
been seen up and then stops answering, or if the hub never comes up within `--hub-wait` seconds —
so a launcher started from an already-open terminal does not leave it running.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KEEP_ALIVE = "3h"
DEFAULT_INTERVAL_SECONDS = 120.0
REQUEST_TIMEOUT_SECONDS = 20.0


def resolve_target() -> tuple[str, str]:
    """(native API root, model) as the app would use them. Imports llm_client for its .env load."""
    from config import settings
    from scripts import llm_client

    base = llm_client._cfg("OLLAMA_BASE_URL", llm_client.DEFAULT_OLLAMA_BASE_URL)
    return native_root(base), settings.LOCAL_MODEL


def native_root(base_url: str) -> str:
    """The OpenAI-compatible base (`…/v1`) → Ollama's native API root."""
    return base_url.rstrip("/").removesuffix("/v1")


def build_request(root: str, model: str) -> urllib.request.Request:
    body = json.dumps({"model": model, "keep_alive": KEEP_ALIVE}).encode("utf-8")
    return urllib.request.Request(f"{root}/api/generate", data=body, method="POST",
                                  headers={"Content-Type": "application/json"})


def ping(root: str, model: str, timeout: float = REQUEST_TIMEOUT_SECONDS) -> bool:
    """One refresh. Never raises: a down Ollama just means the next ping tries again."""
    try:
        with urllib.request.urlopen(build_request(root, model), timeout=timeout) as resp:
            resp.read()
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def hub_up(host: str, port: int, timeout: float = 1.0) -> bool:
    """True if something accepts a TCP connection on the hub's address."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class HubWatch:
    """Decides when a `--follow-hub` pinger should stop: the hub was up and is now gone, or it
    never came up within `wait` seconds. `probe` is injectable for tests."""

    def __init__(self, probe: Callable[[], bool], wait: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._probe, self._wait, self._clock = probe, wait, clock
        self._start = clock()
        self.seen = False

    def should_stop(self) -> bool:
        if self._probe():
            self.seen = True
            return False
        return self.seen or (self._clock() - self._start) > self._wait


def run(root: str, model: str, interval: float, watch: HubWatch | None,
        sleep: Callable[[float], None] = time.sleep, poll: float = 5.0) -> None:
    """Ping every `interval` s; with a watch, check the hub every `poll` s and return when told."""
    next_ping = 0.0
    elapsed = 0.0
    while True:
        if watch is not None and watch.should_stop():
            return
        if elapsed >= next_ping:
            ping(root, model)
            next_ping = elapsed + max(1.0, interval)
        step = min(poll, max(1.0, interval)) if watch is not None else max(1.0, interval)
        sleep(step)
        elapsed += step


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--once", action="store_true", help="ping once and exit")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SECONDS,
                        help="seconds between pings (default %(default)s)")
    parser.add_argument("--follow-hub", action="store_true",
                        help="exit when the hub stops answering (or never starts)")
    parser.add_argument("--hub-wait", type=float, default=180.0,
                        help="with --follow-hub: give up if the hub is not up after this many "
                             "seconds (default %(default)s)")
    args = parser.parse_args(argv)
    root, model = resolve_target()
    if args.once:
        return 0 if ping(root, model) else 1
    watch = None
    if args.follow_hub:
        from config import settings
        watch = HubWatch(lambda: hub_up(settings.DASHBOARD_HOST, settings.DASHBOARD_PORT),
                         wait=args.hub_wait)
    try:
        run(root, model, args.interval, watch)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
