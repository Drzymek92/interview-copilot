"""D35 — the hub: pages, the one-capture-mode manager, the jobs, and the input guards.

No subprocess, GPU or model: modes are opened but transcription is never switched on (a capturing
mode is simulated by flipping the controller's flag), and jobs run injected callables."""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from config import settings  # noqa: E402
from scripts import app_hub as hub  # noqa: E402
from scripts import reasoning  # noqa: E402
from scripts import user_config as uc  # noqa: E402

PAGES = ["/", "/call", "/train", "/review", "/generate", "/settings"]


def _bundle(root: Path, sid: str, **extra) -> None:
    (root / sid).mkdir(parents=True)
    body = {"schema_version": 2, "session_id": sid, "role": extra.pop("role", ""), "company": "",
            "job_description": "", "company_brief": "", "resume": "", "answer_bank": [], "plan": [],
            "honesty_boundary": [], **extra}
    (root / sid / "bundle.json").write_text(json.dumps(body))


@pytest.fixture
def env(tmp_path, monkeypatch):
    sessions = tmp_path / "sessions"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    _bundle(sessions, "blank")
    _bundle(sessions, "acme_20260926", role="AI Engineer",
            plan=[{"id": "p1", "title": "Intro", "key_points": [], "done_signals": []}],
            resume={"status": "placeholder"})
    monkeypatch.setattr(reasoning, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(hub, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(hub, "OUTPUT_DIR", outputs)
    monkeypatch.setattr(hub, "RUNS_LOG", outputs / "hub_runs.jsonl")
    monkeypatch.setattr(hub, "ollama_status", lambda timeout=0.6: {"reachable": False, "loaded": []})
    manager = hub.ModeManager(runs_log=outputs / "hub_runs.jsonl")
    client = TestClient(hub.create_hub(manager))
    yield {"client": client, "manager": manager, "sessions": sessions, "outputs": outputs}
    manager.stop()


# -- pages -------------------------------------------------------------------------------------
@pytest.mark.parametrize("route", PAGES)
def test_every_page_is_served_with_the_permanent_disclosure(env, route):
    r = env["client"].get(route)
    assert r.status_code == 200
    assert "Disclosed AI assistance (D11)" in r.text          # SI2 — on every page, no toggle
    assert "/static/app.css" in r.text
    assert "fonts.googleapis" not in r.text and "cdn" not in r.text.lower()   # SI1: no egress


def test_mode_pages_link_home_and_offer_their_own_settings(env):
    for route, group in [("/call", "call"), ("/train", "train"), ("/review", "review"), ("/generate", "generate")]:
        text = env["client"].get(route).text
        assert 'href="/"' in text
        assert f'initSettingsDrawer(el("gear"), "{group}"' in text


def test_static_whitelist(env):
    c = env["client"]
    assert c.get("/static/app.css").status_code == 200
    assert "[hidden]" in c.get("/static/app.css").text     # the stacked-panels bug stays fixed
    assert c.get("/static/common.js").status_code == 200
    assert c.get("/static/..%2Fapp_hub.py").status_code == 404
    assert c.get("/static/app_hub.py").status_code == 404


# -- catalogues ----------------------------------------------------------------------------------
def test_sessions_list_puts_the_default_first_and_flags_placeholders(env):
    rows = env["client"].get("/api/sessions").json()["sessions"]
    assert rows[0]["id"] == hub.DEFAULT_SESSION and rows[0]["default"] is True     # seeded by create_hub (D37)
    assert rows[-1]["id"] == "blank" and rows[-1]["empty"] is True
    acme = next(r for r in rows if r["id"] == "acme_20260926")
    assert acme["placeholders"] == ["resume"] and acme["plan_steps"] == 1


def test_broken_bundle_is_listed_not_hidden(env):
    (env["sessions"] / "broken").mkdir()
    (env["sessions"] / "broken" / "bundle.json").write_text("{not json")
    rows = env["client"].get("/api/sessions").json()["sessions"]
    assert next(r for r in rows if r["id"] == "broken")["ok"] is False


def test_interviews_join_the_run_log(env):
    out = env["outputs"]
    (out / "interview_qa_20260926_101500.jsonl").write_text(
        json.dumps({"question": "What is RAG?", "answer": "Retrieval…"}) + "\n"
        + json.dumps({"question": "Why us?", "answer": ""}) + "\nnot json\n")
    (out / "interview_review_20260926_120000_abc123.md").write_text("# Interview review\n")
    hub.append_run({"kind": "call", "session": "acme_20260926", "qa": "interview_qa_20260926_101500.jsonl"},
                   out / "hub_runs.jsonl")
    hub.append_run({"kind": "review", "qa": "interview_qa_20260926_101500.jsonl",
                    "report": "interview_review_20260926_120000_abc123.md"}, out / "hub_runs.jsonl")
    data = env["client"].get("/api/interviews").json()
    [row] = data["interviews"]
    assert (row["pairs"], row["answered"], row["session"]) == (2, 1, "acme_20260926")
    assert row["reports"] == ["interview_review_20260926_120000_abc123.md"]
    assert data["reports"][0]["kind"] == "review"


def test_reports_endpoint_refuses_anything_but_report_names(env):
    c = env["client"]
    (env["outputs"] / "training_report_20260926_1.md").write_text("# Mock interview\n- ok")
    assert c.get("/api/reports/training_report_20260926_1.md").json()["markdown"].startswith("# Mock")
    for bad in ["hub_runs.jsonl", "..%2F..%2Fconfig%2F.env", "live_transcript_1.txt"]:
        assert c.get(f"/api/reports/{bad}").status_code == 404


# -- capture modes -------------------------------------------------------------------------------
def test_open_a_call_then_end_it(env):
    c = env["client"]
    r = c.post("/api/mode", json={"mode": "call", "session": "acme_20260926"})
    assert r.status_code == 200 and r.json()["active"]["mode"] == "call"
    with c.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["kind"] == "hello" and hello["mode"] == "call"
        assert hello["session"] == "acme_20260926" and hello["controls"]["transcription"] is False
        assert [s["title"] for s in hello["plan"]] == ["Intro"]
    assert c.post("/api/mode/stop").json()["active"] is None
    with c.websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"kind": "idle"}


def test_ending_a_mode_tells_open_sockets(env):
    c = env["client"]
    c.post("/api/mode", json={"mode": "call", "session": "blank"})
    with c.websocket_connect("/ws") as ws:
        assert ws.receive_json()["kind"] == "hello"
        env["manager"].stop()
        kinds = []
        for _ in range(50):
            msg = ws.receive_json()
            kinds.append(msg["kind"])
            if msg["kind"] == "mode_ended":
                break
        assert kinds[-1] == "mode_ended"


def test_only_one_capture_mode_at_a_time(env):
    c, m = env["client"], env["manager"]
    c.post("/api/mode", json={"mode": "call", "session": "blank"})
    m.active.controller.transcription_on = True          # simulate live capture
    r = c.post("/api/mode", json={"mode": "train", "session": "blank"})
    assert r.status_code == 409 and "capturing" in r.json()["error"]
    assert m.active.name == "call"
    m.active.controller.transcription_on = False
    r = c.post("/api/mode", json={"mode": "train", "session": "blank"})
    assert r.status_code == 200 and m.active.name == "train"


def test_underscore_bundle_ids_open(env):
    _bundle(env["sessions"], "_e2e_14b_20260917")
    assert env["client"].post("/api/mode", json={"mode": "call", "session": "_e2e_14b_20260917"}).status_code == 200


def test_reopening_the_same_mode_is_idempotent(env):
    c, m = env["client"], env["manager"]
    c.post("/api/mode", json={"mode": "call", "session": "blank"})
    first = m.active
    c.post("/api/mode", json={"mode": "call", "session": "blank"})
    assert m.active is first


@pytest.mark.parametrize("payload,status", [
    ({"mode": "call", "session": ""}, 200),          # empty = the default bundle (D37)
    ({"mode": "call", "session": "../etc"}, 400),
    ({"mode": "call", "session": "nope"}, 404),
    ({"mode": "karaoke", "session": "blank"}, 404),
])
def test_mode_start_guards(env, payload, status):
    assert env["client"].post("/api/mode", json=payload).status_code == status


def test_control_needs_the_matching_running_mode(env):
    c = env["client"]
    assert c.post("/api/control", json={"switch": "suggestions", "value": False}).status_code == 409
    c.post("/api/mode", json={"mode": "call", "session": "blank"})
    assert c.post("/api/control", json={"mode": "train", "switch": "training", "value": "start"}).status_code == 409
    r = c.post("/api/control", json={"mode": "call", "switch": "suggestions", "value": False})
    assert r.status_code == 200 and r.json()["suggestions"] is False


def test_mode_args_follow_the_settings(env, monkeypatch):
    monkeypatch.setattr(settings, "SALIENCE_GATE_ENABLED", False)
    monkeypatch.setattr(settings, "REASONING_BACKEND", "cloud")
    monkeypatch.setattr(hub, "cloud_ready", lambda: False)
    args = hub.ModeManager._args("call")
    assert args.no_salience is True
    assert args.backend == "local"          # cloud asked for but not configured → stays local (SI1)
    assert hub.ModeManager._args("train").backend is None


def test_capture_start_is_recorded_for_review(env):
    m = env["manager"]
    m.start("call", "acme_20260926")
    m.active.controller._last_qa_path = env["outputs"] / "interview_qa_20260926_130000.jsonl"
    m.active.controller._current_path = env["outputs"] / "live_transcript_20260926_130000.txt"
    m.note_capture_started()
    [row] = hub.read_runs(env["outputs"] / "hub_runs.jsonl")
    assert row["session"] == "acme_20260926" and row["qa"] == "interview_qa_20260926_130000.jsonl"


# -- settings (D36) through the hub ----------------------------------------------------------------
@pytest.fixture
def saved(tmp_path, monkeypatch):
    path = tmp_path / "user_settings.json"
    monkeypatch.setattr(settings, "USER_SETTINGS_FILE", path)
    monkeypatch.setattr(settings, "ORIGINAL_ENV_KEYS",
                        frozenset(k for k in settings.ORIGINAL_ENV_KEYS if k not in uc.BY_ENV))
    for knob in uc.KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    monkeypatch.setenv("COPILOT_USER_SETTINGS", str(path))
    yield path
    for name in list(settings._INJECTED):
        monkeypatch.delenv(name, raising=False)
    settings._INJECTED.clear()
    monkeypatch.undo()
    importlib.reload(settings)


def test_config_roundtrip(env, saved):
    c = env["client"]
    groups = c.get("/api/config?group=train").json()
    assert {k["env"] for k in groups["knobs"]} >= {"TRAINING_QUESTION_COUNT", "SCORING_BACKEND"}
    r = c.post("/api/config", json={"updates": {"TRAINING_QUESTION_COUNT": "4"}})
    assert r.status_code == 200 and settings.TRAINING_QUESTION_COUNT == 4
    row = next(k for k in r.json()["knobs"] if k["env"] == "TRAINING_QUESTION_COUNT")
    assert row["source"] == "saved"
    r = c.post("/api/config/reset", json={"names": ["TRAINING_QUESTION_COUNT"]})
    assert settings.TRAINING_QUESTION_COUNT == 6


def test_config_refuses_the_bind(env, saved):
    c = env["client"]
    r = c.post("/api/config", json={"updates": {"DASHBOARD_HOST": "0.0.0.0"}})
    assert r.status_code == 400
    assert c.post("/api/config/reset", json={"names": ["DASHBOARD_HOST"]}).status_code == 400
    assert c.get("/api/config?group=nope").status_code == 404
    assert not saved.exists()


# -- jobs ----------------------------------------------------------------------------------------
def _wait(client, job_id: str) -> dict:
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()["job"]
        if job["status"] != "running":
            return job
        time.sleep(0.01)
    raise AssertionError("job never finished")


def test_review_job_validates_then_runs(env, monkeypatch):
    c = env["client"]
    (env["outputs"] / "interview_qa_20260926_101500.jsonl").write_text(
        json.dumps({"question": "Q?", "answer": "A."}) + "\n")
    assert c.post("/api/review", json={"qa": "../x.jsonl", "session": "blank"}).status_code == 400
    assert c.post("/api/review", json={"qa": "interview_qa_1.jsonl", "session": "blank"}).status_code == 400
    assert c.post("/api/review", json={"qa": "interview_qa_20260926_101500.jsonl", "session": "nope"}).status_code == 400
    monkeypatch.setattr(hub, "run_review_job", lambda qa, session: {"report": "r.md", "qa": qa, "session": session})
    r = c.post("/api/review", json={"qa": "interview_qa_20260926_101500.jsonl", "session": "blank"})
    job = _wait(c, r.json()["job"]["id"])
    assert job["status"] == "done" and job["result"]["session"] == "blank"
    assert c.get("/api/jobs?kind=review").json()["job"]["id"] == job["id"]


def test_failed_job_reports_its_error(env):
    jobs = hub.Jobs()
    job = jobs.submit("review", "x", lambda: (_ for _ in ()).throw(ValueError("no answered pairs")))
    for _ in range(200):
        if job.status != "running":
            break
        time.sleep(0.01)
    assert job.status == "error" and "no answered pairs" in job.error


def test_one_job_per_kind(env):
    jobs = hub.Jobs()
    gate = __import__("threading").Event()
    jobs.submit("generate", "a", lambda: gate.wait(2) and {})
    with pytest.raises(hub.HubError):
        jobs.submit("generate", "b", lambda: {})
    gate.set()


@pytest.mark.parametrize("payload,status", [
    ({"jd_text": "too short", "session_id": "x1"}, 400),
    ({"jd_text": "x" * 200, "session_id": "bad id!"}, 400),
    ({"jd_text": "x" * 200, "session_id": "ok_1", "spoken_language": "de"}, 400),
    ({"jd_text": "x" * 200, "session_id": "acme_20260926"}, 409),
])
def test_generate_guards(env, payload, status):
    assert env["client"].post("/api/generate", json=payload).status_code == status


def test_generate_job_runs_with_the_form_values(env, monkeypatch):
    seen = {}

    def fake(jd_text, session_id, role, company, spoken, cv_text=""):
        seen.update(jd=len(jd_text), sid=session_id, role=role, company=company, spoken=spoken)
        return {"session": session_id}

    monkeypatch.setattr(hub, "run_generate_job", fake)
    c = env["client"]
    r = c.post("/api/generate", json={"jd_text": "y" * 300, "session_id": "newco_1", "company": "NewCo",
                                      "role": "", "spoken_language": "en"})
    assert _wait(c, r.json()["job"]["id"])["status"] == "done"
    assert seen == {"jd": 300, "sid": "newco_1", "role": "", "company": "NewCo", "spoken": "en"}


def test_status_reports_mode_devices_and_cloud(env):
    s = env["client"].get("/api/status").json()
    assert s["active"] is None and "cloud_available" in s and s["ollama"]["reachable"] is False
    assert s["jobs"] == {"review": None, "generate": None}


def test_run_generate_job_writes_and_summarises_a_bundle(tmp_path, monkeypatch):
    from scripts import generate_context as gen
    sections = {"role": "LLM Engineer", "job_description": {"summary": "Build RAG"}, "company_brief": "Acme",
                "plan": [{"id": "p1", "title": "Intro", "key_points": [], "done_signals": []}],
                "question_bank": [{"id": "q1", "competency": "RAG", "question": "What is RAG?",
                                   "rubric": {"criteria": [{"id": "c1", "label": "defines it", "weight": 1}]}}]}
    monkeypatch.setattr(gen, "generate_sections", lambda *a, **k: sections)
    monkeypatch.setattr(reasoning, "SESSIONS_DIR", tmp_path)
    res = hub.run_generate_job("x" * 200, "acme_1", "", "Acme", "pl", sessions_dir=tmp_path)
    assert res["session"] == "acme_1" and res["role"] == "LLM Engineer"
    assert res["plan"] == ["Intro"] and res["questions"] == 1
    assert set(res["placeholders"]) == {"resume", "answer_bank", "honesty_boundary"}   # never invented (D22)


def test_run_review_job_records_the_report(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import review as review_mod
    report = SimpleNamespace(path=tmp_path / "interview_review_x.md", answers=[], total=0.0, max_score=0.0,
                             normalized=0.0)
    seen = {}
    monkeypatch.setattr(review_mod, "review", lambda session, qa, **k: seen.update(session=session, qa=qa.name, **k) or report)
    monkeypatch.setattr(settings, "REVIEW_SYNTHESIZE_RUBRICS", True)
    res = hub.run_review_job("interview_qa_1.jsonl", "acme", output_dir=tmp_path, runs_log=tmp_path / "runs.jsonl")
    assert seen["synthesize_rubrics"] is True and seen["session"] == "acme"
    assert res["report"] == "interview_review_x.md" and res["overall"]["answered"] == 0
    [row] = hub.read_runs(tmp_path / "runs.jsonl")
    assert row["kind"] == "review" and row["report"] == "interview_review_x.md"


# -- D37: default bundle, extract, CV ---------------------------------------------------------------
def test_default_bundle_is_seeded_and_never_overwritten(env):
    target = env["sessions"] / hub.DEFAULT_SESSION / "bundle.json"
    assert target.is_file()
    target.write_text(target.read_text().replace("generic default", "my edited default"))
    hub.ensure_default_bundle(env["sessions"])
    assert "my edited default" in target.read_text()


def test_no_bundle_selected_runs_on_the_default(env):
    c = env["client"]
    r = c.post("/api/mode", json={"mode": "train", "session": ""})
    assert r.status_code == 200 and r.json()["active"]["session"] == hub.DEFAULT_SESSION
    assert len(env["manager"].active.controller.bundle.question_bank) >= 5    # trainable without generation


def test_review_without_a_bundle_uses_the_default(env, monkeypatch):
    c = env["client"]
    (env["outputs"] / "interview_qa_20260926_101500.jsonl").write_text(json.dumps({"question": "Q?", "answer": "A."}) + "\n")
    monkeypatch.setattr(hub, "run_review_job", lambda qa, session: {"session": session})
    r = c.post("/api/review", json={"qa": "interview_qa_20260926_101500.jsonl", "session": ""})
    assert _wait(c, r.json()["job"]["id"])["result"]["session"] == hub.DEFAULT_SESSION


def test_default_name_cannot_be_overwritten_by_generate(env):
    r = env["client"].post("/api/generate", json={"jd_text": "x" * 200, "session_id": hub.DEFAULT_SESSION})
    assert r.status_code == 400


def test_extract_endpoint_reads_text_and_refuses_junk(env):
    import base64
    c = env["client"]
    b64 = base64.b64encode("Senior AI Engineer\nPython, RAG".encode()).decode()
    r = c.post("/api/extract", json={"filename": "jd.txt", "data_b64": b64})
    assert r.status_code == 200 and r.json()["text"].startswith("Senior AI Engineer")
    assert c.post("/api/extract", json={"filename": "jd.txt", "data_b64": "@@@"}).status_code == 400
    assert c.post("/api/extract", json={"filename": "cv.exe", "data_b64": b64}).status_code == 400


def test_extract_endpoint_reads_a_pdf(env):
    import base64
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Jan Kowalski CV")
    data = doc.tobytes()
    doc.close()
    r = env["client"].post("/api/extract", json={"filename": "cv.pdf", "data_b64": base64.b64encode(data).decode()})
    assert r.status_code == 200 and "Jan Kowalski CV" in r.json()["text"] and r.json()["pages"] == 1


def test_generate_passes_the_cv(env, monkeypatch):
    seen = {}
    monkeypatch.setattr(hub, "run_generate_job", lambda *a, cv_text="", **k: seen.update(cv=cv_text) or {})
    c = env["client"]
    r = c.post("/api/generate", json={"jd_text": "y" * 300, "cv_text": "My CV", "session_id": "cv_1"})
    _wait(c, r.json()["job"]["id"])
    assert seen["cv"] == "My CV"
    assert c.post("/api/generate", json={"jd_text": "y" * 300, "cv_text": "z" * 60001, "session_id": "cv_2"}).status_code == 400


def test_socket_closed_by_the_client_is_not_logged_as_an_error(env, caplog):
    """e2e round 1: closing a tab mid-stream raised RuntimeError('... close message has been sent') and was
    logged with a traceback. It is a normal disconnect."""
    c = env["client"]
    c.post("/api/mode", json={"mode": "call", "session": "blank"})
    with c.websocket_connect("/ws") as ws:
        assert ws.receive_json()["kind"] == "hello"
    time.sleep(0.6)                       # let a tick hit the closed socket
    assert "hub websocket closed on an error" not in caplog.text
