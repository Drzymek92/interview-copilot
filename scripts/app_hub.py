"""The interview copilot hub (D35): one loopback server, an opening menu, one view per job.

    python scripts/app_hub.py            # what the desktop shortcut runs (scripts/launch_copilot.sh)

`/` is the menu — **Start a call · Train · Review · Generate context · Settings** — and each mode is
its own function-centred page (`/call`, `/train`, `/review`, `/generate`, `/settings`) with a ⚙
drawer for that mode's settings (the D36 saved layer, `scripts/user_config.py`).

What runs where:
  call / train  → a `ModeManager` runs AT MOST ONE capture mode at a time. It builds the existing
                  `AppController` / `TrainingController` + `DashboardState` + `EventBus` unchanged
                  (D18/D31/D34 behaviour does not move) for the bundle picked in the view. Ending
                  the mode — or shutting the hub down — stops its recorder: capture is never
                  orphaned. Reloading or closing the page does NOT stop it (a stray refresh must
                  never end capture mid-interview); the menu shows a running mode instead.
  review / generate → background jobs on this server (`review.review`, `generate_context.run`),
                  polled via `/api/jobs/<id>`; each is a model call that can take minutes.

SI1: binds 127.0.0.1 only (`_assert_loopback`, not overridable from the UI — host/port are not in
the D36 registry). Every endpoint is same-origin + loopback. Cloud stays opt-in and announced.
SI2/D11: every page carries the permanent disclosure banner; there is no hide control.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import shutil
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response  # noqa: E402

from config import settings  # noqa: E402
from scripts import user_config  # noqa: E402
from scripts.dashboard import (  # noqa: E402
    OUTPUT_DIR, AppController, DashboardState, EventBus, RunInfo, TrainingController,
    _assert_loopback, _open_browser, _tick_loop, dispatch_control, meter_ceiling,
    scorecard_payload,
)
from scripts.llm_client import announce_backend, cloud_ready, model_for  # noqa: E402
from scripts.doc_text import DocTextError, extract_text  # noqa: E402
from scripts.logger import get_logger  # noqa: E402
from scripts.reasoning import SESSIONS_DIR, load_bundle  # noqa: E402

logger = get_logger("app_hub")

UI_DIR = Path(__file__).resolve().parent / "ui"
PAGES = {"/": "menu.html", "/call": "call.html", "/train": "train.html", "/review": "review.html",
         "/generate": "generate.html", "/settings": "settings.html"}
STATIC = {"app.css": "text/css", "common.js": "text/javascript"}
RUNS_LOG = OUTPUT_DIR / "hub_runs.jsonl"     # which bundle each captured interview used (local)
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_\-]{0,63}$")   # no dots or slashes: never a path
QA_NAME_RE = re.compile(r"^interview_qa_[0-9_]+\.jsonl$")
REPORT_NAME_RE = re.compile(r"^(interview_review|training_report)_[0-9A-Za-z_]+\.md$")
MODES = ("call", "train")
# D37: the bundle every mode falls back to when none is picked — a generic AI tech job. The template is
# tracked in the repo (scripts/inputs/ is gitignored) and seeded into the sessions dir when missing.
DEFAULT_SESSION = "default_ai_tech"
DEFAULT_TEMPLATE = PROJECT_ROOT / "config" / "bundles" / DEFAULT_SESSION / "bundle.json"


class HubError(Exception):
    """A refused request — surfaced as JSON {"error": ...} with `status`."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------
# small local records
# --------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_run(record: dict, path: Path | None = None) -> None:
    """Append one line to the hub's run log. Best-effort: a failed write must never break capture."""
    target = path or RUNS_LOG
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": _now_iso(), **record}, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("could not append to %s", target)


def read_runs(path: Path | None = None) -> list[dict]:
    target = path or RUNS_LOG
    rows: list[dict] = []
    try:
        for line in target.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    except OSError:
        pass
    return rows


def ensure_default_bundle(sessions_dir: Path | None = None) -> Path | None:
    """Seed `<sessions>/default_ai_tech/bundle.json` from the tracked template if it is missing.
    Never overwrites: a user who edits their copy keeps it."""
    target = (sessions_dir or SESSIONS_DIR) / DEFAULT_SESSION / "bundle.json"
    if target.is_file() or not DEFAULT_TEMPLATE.is_file():
        return target if target.is_file() else None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DEFAULT_TEMPLATE, target)
        logger.info("seeded the default bundle at %s", target)
        return target
    except OSError:
        logger.exception("could not seed the default bundle")
        return None


def resolve_session(session: str | None) -> str:
    """'' / None → the default bundle (D37)."""
    return (session or "").strip() or DEFAULT_SESSION


def list_sessions(sessions_dir: Path | None = None) -> list[dict]:
    """Every context bundle on disk, newest first, with what the pickers show about it."""
    root = sessions_dir or SESSIONS_DIR
    out: list[dict] = []
    for manifest in sorted(root.glob("*/bundle.json")):
        sid = manifest.parent.name
        row = {"id": sid, "modified": datetime.fromtimestamp(manifest.stat().st_mtime).isoformat(timespec="minutes")}
        try:
            bundle = load_bundle(sid, sessions_dir=root)
        except Exception as exc:        # a broken bundle is listed as broken, never hidden
            out.append({**row, "ok": False, "error": str(exc)})
            continue
        out.append({**row, "ok": True, "default": sid == DEFAULT_SESSION, "role": bundle.role, "company": bundle.company,
                    "placeholders": list(bundle.placeholders), "plan_steps": len(bundle.plan),
                    "questions": len(bundle.question_bank), "answer_bank": len(bundle.answer_bank),
                    "has_resume": bool(bundle.resume.strip()),
                    "empty": not any((bundle.job_description.strip(), bundle.resume.strip(),
                                      bundle.plan, bundle.answer_bank))})
    # the default first (it is what "nothing selected" means), then newest first, `blank` last
    first = [r for r in out if r["id"] == DEFAULT_SESSION]
    last = [r for r in out if r["id"] == "blank"]
    rest = sorted((r for r in out if r["id"] not in (DEFAULT_SESSION, "blank")),
                  key=lambda r: r["modified"], reverse=True)
    return first + rest + last


def list_interviews(output_dir: Path | None = None, runs: list[dict] | None = None) -> list[dict]:
    """Captured Q&A logs (D33), newest first, joined with the hub's run log for their bundle and
    any review already written for them."""
    root = output_dir or OUTPUT_DIR
    runs = read_runs() if runs is None else runs
    session_for = {r.get("qa"): r.get("session") for r in runs if r.get("kind") == "call" and r.get("qa")}
    reviews: dict[str, list[str]] = {}
    for r in runs:
        if r.get("kind") == "review" and r.get("qa") and r.get("report"):
            reviews.setdefault(r["qa"], []).append(r["report"])
    out = []
    for path in sorted(root.glob("interview_qa_*.jsonl"), reverse=True):
        pairs = answered = 0
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict) and str(obj.get("question", "")).strip():
                    pairs += 1
                    answered += bool(str(obj.get("answer", "")).strip())
        except OSError:
            continue
        stamp = path.stem.removeprefix("interview_qa_")
        out.append({"name": path.name, "stamp": stamp, "pairs": pairs, "answered": answered,
                    "session": session_for.get(path.name),
                    "reports": [n for n in reviews.get(path.name, []) if (root / n).exists()]})
    return out


def list_reports(output_dir: Path | None = None) -> list[dict]:
    root = output_dir or OUTPUT_DIR
    out = []
    for path in sorted(root.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        if REPORT_NAME_RE.match(path.name):
            out.append({"name": path.name,
                        "kind": "review" if path.name.startswith("interview_review") else "training",
                        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="minutes")})
    return out


def ollama_status(timeout: float = 0.6) -> dict:
    """Is the local model server up, and which models are resident? Loopback only; never raises."""
    base = (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{base}/api/ps", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {"reachable": True, "loaded": [m.get("name", "") for m in data.get("models", [])]}
    except Exception:
        return {"reachable": False, "loaded": []}


def audio_sources(timeout: float = 2.0) -> dict:
    """PulseAudio/PipeWire capture sources, split into microphones and output monitors, for the
    Settings view's device pickers. Read-only (`pactl list short sources`); never raises."""
    def pactl(*args: str) -> str:
        return subprocess.run(["pactl", *args], check=True, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    try:
        names = [ln.split("\t")[1] for ln in pactl("list", "short", "sources").splitlines() if "\t" in ln]
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "mics": [], "monitors": [], "default_mic": None, "default_monitor": None}
    def default(kind: str) -> str | None:
        try:
            return pactl(kind) or None
        except (OSError, subprocess.SubprocessError):
            return None
    sink = default("get-default-sink")
    src = default("get-default-source")
    return {"available": True,
            "mics": [n for n in names if not n.endswith(".monitor")],
            "monitors": [n for n in names if n.endswith(".monitor")],
            "default_mic": src if src and not src.endswith(".monitor") else None,
            "default_monitor": f"{sink}.monitor" if sink and f"{sink}.monitor" in names else None}


# --------------------------------------------------------------------------
# capture modes (call / train): at most one at a time
# --------------------------------------------------------------------------
@dataclass
class ActiveMode:
    name: str
    session: str
    controller: AppController | TrainingController
    state: DashboardState
    bus: EventBus
    info: RunInfo
    ceiling: float
    started: str = field(default_factory=_now_iso)

    @property
    def capturing(self) -> bool:
        return bool(getattr(self.controller, "transcription_on", False))


class ModeManager:
    """Owns the one active capture mode. Not a governance seat (fw:D2): it builds and tears down
    the existing controllers; every interview decision stays where D19/D23/D30/D32 put it."""

    def __init__(self, runs_log: Path | None = None) -> None:
        self._lock = threading.RLock()
        self.active: ActiveMode | None = None
        self.runs_log = runs_log

    @staticmethod
    def _args(mode: str) -> argparse.Namespace:
        """The Namespace the controllers expect, filled from settings (CLI > env > saved > default,
        D36) at the moment the mode starts — so a Settings change applies from the next start."""
        backend: str | None
        if mode == "call":
            backend = settings.REASONING_BACKEND if settings.REASONING_BACKEND == "local" or cloud_ready() else "local"
        else:
            backend = None      # the trainer resolves SCORING_BACKEND itself
        return argparse.Namespace(
            backend=backend, model=None, no_salience=not settings.SALIENCE_GATE_ENABLED,
            suggestion_language=None, answer_speaker=None, from_end=False, partials=None,
            no_suggestions=False, questions=None, mode="", ceiling_seconds=0.0,
        )

    def start(self, mode: str, session: str) -> dict:
        if mode not in MODES:
            raise HubError(f"unknown mode {mode!r}", 404)
        session = resolve_session(session)
        if not SESSION_ID_RE.match(session):
            raise HubError("pick a context bundle first")
        with self._lock:
            if self.active is not None:
                if self.active.name == mode and self.active.session == session:
                    return self.status()
                if self.active.capturing:
                    raise HubError(
                        f"a {self.active.name} is capturing audio right now — end it before "
                        f"starting a {mode}", 409)
                self._stop_locked()
            try:
                bundle = load_bundle(session)
            except FileNotFoundError as exc:
                raise HubError(str(exc), 404) from exc
            except ValueError as exc:
                raise HubError(f"bundle {session!r} could not be loaded: {exc}") from exc
            args = self._args(mode)
            ceiling = meter_ceiling()
            partials_on = settings.DASHBOARD_SHOW_PARTIALS
            state = DashboardState(plan=bundle.plan)
            bus = EventBus()
            if mode == "call":
                controller = AppController(args, state, bus, bundle, ceiling, partials_on)
                info = RunInfo(source="— (transcription off)", mode="live (hub)", session=session,
                               salience="on (D23)" if settings.SALIENCE_GATE_ENABLED else "off (ungated)",
                               ceiling_seconds=ceiling, suggestions=True,
                               provisional="on (D25)" if partials_on else "off")
            else:
                controller = TrainingController(args, state, bus, bundle, ceiling, partials_on)
                info = RunInfo(source="— (training)", mode="training", session=session,
                               salience="n/a (training)", ceiling_seconds=ceiling, suggestions=False,
                               provisional="on (D25)" if partials_on else "off")
            self.active = ActiveMode(mode, session, controller, state, bus, info, ceiling)
            logger.info("mode %s started on bundle %s", mode, session)
            return self.status()

    def stop(self) -> dict:
        with self._lock:
            self._stop_locked()
            return self.status()

    def _stop_locked(self) -> None:
        active, self.active = self.active, None
        if active is None:
            return
        try:
            active.controller.shutdown()       # SIGINTs the recorder if one is running
        except Exception:
            logger.exception("mode %s: shutdown raised", active.name)
        active.bus.publish({"kind": "mode_ended", "mode": active.name})
        logger.info("mode %s ended (bundle %s)", active.name, active.session)

    def note_capture_started(self) -> None:
        """After a call's transcription switch goes ON: record which bundle this Q&A log belongs
        to, so the Review view can default to it (the D33 log itself stays unchanged)."""
        with self._lock:
            active = self.active
        if active is None or active.name != "call":
            return
        qa = getattr(active.controller, "_last_qa_path", None)
        source = getattr(active.controller, "_current_path", None)
        append_run({"kind": "call", "session": active.session,
                    "qa": qa.name if qa else None, "transcript": source.name if source else None},
                   self.runs_log)

    def status(self) -> dict:
        with self._lock:
            a = self.active
            if a is None:
                return {"active": None}
            return {"active": {"mode": a.name, "session": a.session, "capturing": a.capturing,
                               "started": a.started}}


# --------------------------------------------------------------------------
# background jobs (review / generate)
# --------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "running"       # running | done | error
    started: float = field(default_factory=time.time)
    finished: float | None = None
    banner: str = ""
    error: str = ""
    result: dict = field(default_factory=dict)

    def view(self) -> dict:
        d = asdict(self)
        d["elapsed"] = round((self.finished or time.time()) - self.started, 1)
        return d


class Jobs:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def running(self, kind: str) -> Job | None:
        with self._lock:
            return next((j for j in self._jobs.values() if j.kind == kind and j.status == "running"), None)

    def submit(self, kind: str, label: str, fn, banner: str = "") -> Job:
        if self.running(kind) is not None:
            raise HubError(f"a {kind} job is already running — wait for it to finish", 409)
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, label=label, banner=banner)
        with self._lock:
            self._jobs[job.id] = job

        def run() -> None:
            try:
                job.result = fn() or {}
                job.status = "done"
            except Exception as exc:     # surfaced to the view verbatim; the traceback goes to the log
                logger.exception("%s job %s failed", kind, job.id)
                job.error = str(exc) or exc.__class__.__name__
                job.status = "error"
            finally:
                job.finished = time.time()

        threading.Thread(target=run, name=f"job-{kind}-{job.id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise HubError("no such job", 404)
        return job

    def latest(self, kind: str) -> Job | None:
        with self._lock:
            jobs = [j for j in self._jobs.values() if j.kind == kind]
        return max(jobs, key=lambda j: j.started) if jobs else None


def _egress_banner(backend_setting: str) -> str:
    """The SI1 banner for a job that will use `backend_setting`, or '' when it stays local."""
    if backend_setting == "cloud":
        return announce_backend("cloud", model_for("cloud"))
    return ""


def run_review_job(qa_name: str, session: str, output_dir: Path | None = None,
                   runs_log: Path | None = None) -> dict:
    from scripts import review as review_mod

    root = output_dir or OUTPUT_DIR
    report = review_mod.review(session, root / qa_name, synthesize_rubrics=settings.REVIEW_SYNTHESIZE_RUBRICS,
                               output_dir=root)
    card = scorecard_payload(report)
    name = report.path.name if report.path else None
    append_run({"kind": "review", "qa": qa_name, "session": session, "report": name}, runs_log)
    return {"report": name, **card}


def run_generate_job(jd_text: str, session_id: str, role: str, company: str,
                     spoken_language: str, sessions_dir: Path | None = None, cv_text: str = "") -> dict:
    from scripts import generate_context as gen

    args = argparse.Namespace(
        jd=jd_text, session=session_id, company=company, role=role, company_notes="", cv=cv_text,
        questions_per_competency=settings.GENERATE_QUESTIONS_PER_COMPETENCY, backend=None,
        model=None, spoken_language=spoken_language, suggestion_language="match",
    )
    target = gen.run(args, sessions_dir=sessions_dir)
    bundle = load_bundle(session_id, sessions_dir=sessions_dir)
    return {"session": session_id, "path": str(target), "role": bundle.role, "company": bundle.company,
            "plan": [s.title for s in bundle.plan], "questions": len(bundle.question_bank),
            "has_resume": bool(bundle.resume.strip()), "placeholders": list(bundle.placeholders)}


# --------------------------------------------------------------------------
# the app
# --------------------------------------------------------------------------
def _err(exc: HubError) -> JSONResponse:
    return JSONResponse({"error": str(exc)}, status_code=exc.status)


def create_hub(manager: ModeManager | None = None, jobs: Jobs | None = None,
               clock=time.monotonic) -> FastAPI:
    manager = manager or ModeManager()
    jobs = jobs or Jobs()
    ensure_default_bundle()
    app = FastAPI(title="interview copilot hub", docs_url=None, redoc_url=None)
    app.state.manager = manager
    app.state.jobs = jobs

    for route, filename in PAGES.items():
        def page(filename: str = filename) -> HTMLResponse:
            return HTMLResponse((UI_DIR / filename).read_text(encoding="utf-8"),
                                headers={"Cache-Control": "no-store"})
        app.add_api_route(route, page, methods=["GET"], response_class=HTMLResponse,
                          include_in_schema=False)

    @app.get("/static/{name}")
    async def static(name: str) -> Response:
        if name not in STATIC:
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(UI_DIR / name, media_type=STATIC[name], headers={"Cache-Control": "no-store"})

    # -- status / catalogues ---------------------------------------------------
    @app.get("/api/status")
    async def status() -> JSONResponse:
        ollama = await asyncio.to_thread(ollama_status)
        return JSONResponse({
            **manager.status(), "cloud_available": cloud_ready(), "ollama": ollama,
            "local_model": settings.LOCAL_MODEL, "mic": settings.COPILOT_MIC,
            "source": settings.COPILOT_SOURCE,
            "jobs": {k: (j.view() if (j := jobs.running(k)) else None) for k in ("review", "generate")},
        })

    @app.get("/api/devices")
    async def devices() -> JSONResponse:
        return JSONResponse(await asyncio.to_thread(audio_sources))

    @app.get("/api/sessions")
    async def sessions() -> JSONResponse:
        return JSONResponse({"sessions": await asyncio.to_thread(list_sessions)})

    @app.get("/api/interviews")
    async def interviews() -> JSONResponse:
        return JSONResponse({"interviews": await asyncio.to_thread(list_interviews),
                             "reports": await asyncio.to_thread(list_reports)})

    @app.get("/api/reports/{name}")
    async def report(name: str) -> JSONResponse:
        if not REPORT_NAME_RE.match(name) or not (OUTPUT_DIR / name).is_file():
            return JSONResponse({"error": "no such report"}, status_code=404)
        return JSONResponse({"name": name, "markdown": (OUTPUT_DIR / name).read_text(encoding="utf-8")})

    # -- settings (D36) ----------------------------------------------------------
    @app.get("/api/config")
    async def get_config(group: str | None = None) -> JSONResponse:
        if group is not None and group not in user_config.GROUPS:
            return JSONResponse({"error": f"unknown group {group!r}"}, status_code=404)
        return JSONResponse({"groups": user_config.GROUPS, "knobs": user_config.describe(group),
                             "file": str(settings.USER_SETTINGS_FILE)})

    @app.post("/api/config")
    async def set_config(payload: dict) -> JSONResponse:
        updates = payload.get("updates")
        if not isinstance(updates, dict) or not updates:
            return JSONResponse({"error": "nothing to save"}, status_code=400)
        try:
            knobs = user_config.save(updates)
        except user_config.ConfigError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "knobs": knobs, "active": manager.status()["active"]})

    @app.post("/api/config/reset")
    async def reset_config(payload: dict) -> JSONResponse:
        names = payload.get("names")
        if names is not None and not (isinstance(names, list) and all(n in user_config.BY_ENV for n in names)):
            return JSONResponse({"error": "not UI-editable settings"}, status_code=400)
        return JSONResponse({"ok": True, "knobs": user_config.reset(names)})

    # -- capture modes ------------------------------------------------------------
    @app.post("/api/mode")
    async def start_mode(payload: dict) -> JSONResponse:
        try:
            st = await asyncio.to_thread(manager.start, str(payload.get("mode", "")),
                                         str(payload.get("session", "")))
        except HubError as exc:
            return _err(exc)
        return JSONResponse({"ok": True, **st})

    @app.post("/api/mode/stop")
    async def stop_mode() -> JSONResponse:
        return JSONResponse({"ok": True, **await asyncio.to_thread(manager.stop)})

    @app.post("/api/control")
    async def control(payload: dict) -> JSONResponse:
        active = manager.active
        if active is None:
            return JSONResponse({"error": "no call or training is running"}, status_code=409)
        mode = str(payload.get("mode", ""))
        if mode and mode != active.name:
            return JSONResponse({"error": f"the running mode is {active.name}, not {mode}"}, status_code=409)
        resp = await dispatch_control(active.controller, payload)
        if (active.name == "call" and payload.get("switch") == "transcription" and payload.get("value")
                and resp.status_code == 200):
            await asyncio.to_thread(manager.note_capture_started)
        return resp

    @app.websocket("/ws")
    async def stream(socket: WebSocket) -> None:
        await socket.accept()
        active = manager.active
        if active is None:
            await socket.send_text(json.dumps({"kind": "idle"}))
            await socket.close()
            return
        active.bus.bind(asyncio.get_running_loop())
        queue = active.bus.subscribe()
        try:
            await socket.send_text(json.dumps({
                **active.state.tick(clock(), active.ceiling), **active.state.snapshot(),
                "info": asdict(active.info), "kind": "hello", "mode": active.name,
                "session": active.session, "controls": active.controller.ui_state(),
            }))
            ticker = asyncio.create_task(_tick_loop(socket, active.state, active.ceiling, clock))
            try:
                while True:
                    event = await queue.get()
                    await socket.send_text(json.dumps(event))
                    if event.get("kind") == "mode_ended":
                        break
            finally:
                ticker.cancel()
            await socket.close()
        except WebSocketDisconnect:
            pass
        except RuntimeError as exc:
            # A tab closed mid-send: Starlette raises RuntimeError ("... once a close message has been
            # sent") rather than WebSocketDisconnect. A normal disconnect, not an error (e2e round 1).
            if "close message" not in str(exc) and "not connected" not in str(exc).lower():
                logger.exception("hub websocket closed on an error")
        except Exception:
            logger.exception("hub websocket closed on an error")
        finally:
            active.bus.unsubscribe(queue)

    # -- jobs: review / generate -------------------------------------------------
    @app.post("/api/review")
    async def start_review(payload: dict) -> JSONResponse:
        qa = str(payload.get("qa", ""))
        session = resolve_session(str(payload.get("session", "")))
        if not QA_NAME_RE.match(qa) or not (OUTPUT_DIR / qa).is_file():
            return JSONResponse({"error": "pick a captured interview"}, status_code=400)
        if not SESSION_ID_RE.match(session) or not (SESSIONS_DIR / session / "bundle.json").is_file():
            return JSONResponse({"error": "pick the context bundle the interview used"}, status_code=400)
        try:
            job = jobs.submit("review", f"{qa} · {session}", lambda: run_review_job(qa, session),
                              banner=_egress_banner(settings.SCORING_BACKEND))
        except HubError as exc:
            return _err(exc)
        return JSONResponse({"ok": True, "job": job.view()})

    @app.post("/api/extract")
    async def extract(payload: dict) -> JSONResponse:
        # D37: a JD or CV file → plain text, parsed locally (PyMuPDF for PDF). The browser sends the file
        # base64-encoded; nothing is written to disk and nothing leaves this machine (SI1).
        name = str(payload.get("filename", ""))[:200]
        raw = str(payload.get("data_b64", ""))
        if len(raw) > 14_500_000:                     # ~10 MB after decoding
            return JSONResponse({"error": "the file is larger than 10 MB"}, status_code=400)
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            return JSONResponse({"error": "the upload was corrupted — try again"}, status_code=400)
        try:
            result = await asyncio.to_thread(extract_text, data, name)
        except DocTextError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "filename": name, **result})

    @app.post("/api/generate")
    async def start_generate(payload: dict) -> JSONResponse:
        jd_text = str(payload.get("jd_text", "")).strip()
        cv_text = str(payload.get("cv_text", "")).strip()
        session_id = str(payload.get("session_id", "")).strip()
        role = str(payload.get("role", "")).strip()[:200]
        company = str(payload.get("company", "")).strip()[:200]
        spoken = str(payload.get("spoken_language", "pl"))
        if len(jd_text) < 80:
            return JSONResponse({"error": "paste the full job description (at least a few sentences)"},
                                status_code=400)
        if len(jd_text) > 60_000:
            return JSONResponse({"error": "that job description is too long (60k characters max)"},
                                status_code=400)
        if len(cv_text) > 60_000:
            return JSONResponse({"error": "that CV is too long (60k characters max)"}, status_code=400)
        if not SESSION_ID_RE.match(session_id):
            return JSONResponse({"error": "bundle name: letters, digits, _ or - only (max 64)"},
                                status_code=400)
        if session_id == DEFAULT_SESSION:
            return JSONResponse({"error": f"{DEFAULT_SESSION!r} is the built-in default — pick another name"},
                                status_code=400)
        if spoken not in ("pl", "en"):
            return JSONResponse({"error": "spoken language must be pl or en"}, status_code=400)
        if (SESSIONS_DIR / session_id / "bundle.json").exists() and not payload.get("overwrite"):
            return JSONResponse({"error": f"a bundle named {session_id!r} already exists",
                                 "exists": True}, status_code=409)
        try:
            job = jobs.submit("generate", session_id,
                              lambda: run_generate_job(jd_text, session_id, role, company, spoken,
                                                       cv_text=cv_text),
                              banner=_egress_banner(settings.GENERATE_BACKEND))
        except HubError as exc:
            return _err(exc)
        return JSONResponse({"ok": True, "job": job.view()})

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str) -> JSONResponse:
        try:
            return JSONResponse({"job": jobs.get(job_id).view()})
        except HubError as exc:
            return _err(exc)

    @app.get("/api/jobs")
    async def job_latest(kind: str) -> JSONResponse:
        job = jobs.latest(kind)
        return JSONResponse({"job": job.view() if job else None})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--host", default=settings.DASHBOARD_HOST, help="loopback only (D18/SI1)")
    parser.add_argument("--port", type=int, default=settings.DASHBOARD_PORT)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--open-path", default="/",
                        help="page the browser opens on, e.g. /call?session=example_ai_engineer (default: the menu)")
    args = parser.parse_args()
    _assert_loopback(args.host)

    manager = ModeManager()
    app = create_hub(manager)
    url = f"http://{args.host}:{args.port}"
    print(f"interview copilot: {url}   (D18/SI1 — loopback only)", flush=True)
    print(f"  cloud   : {'available (opt-in per mode)' if cloud_ready() else 'not configured — local only'}", flush=True)
    print(f"  devices : source={settings.COPILOT_SOURCE}  mic={settings.COPILOT_MIC}"
          + ("   (set your external mic in Settings for a real call)" if settings.COPILOT_MIC == "auto" else ""),
          flush=True)
    logger.info("hub starting on %s:%s", args.host, args.port)
    open_path = args.open_path if args.open_path.startswith("/") else "/" + args.open_path
    if settings.COPILOT_OPEN_BROWSER and not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.0), _open_browser(url + open_path)),
                         name="open-browser", daemon=True).start()

    import uvicorn
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        manager.stop()               # never orphan capture on exit


if __name__ == "__main__":
    main()
