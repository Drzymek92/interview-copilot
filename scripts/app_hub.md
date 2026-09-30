# `scripts/app_hub.py` — the hub: menu + one view per job (D35)

Read this instead of scanning the script (~600 lines) and `scripts/ui/`. Settings layer: `scripts/user_config.py`
(D36). The mode engines it drives are unchanged: `scripts/dashboard.md` (AppController / TrainingController /
DashboardState / EventBus), `scripts/training.md`, `scripts/review.md`, `scripts/generate_context.md`.

## QUICKSTART

```
cd ~/Desktop/Claude_Projects/projects/interview_copilot
scripts/launch_copilot.sh                       # what the desktop shortcuts run (+ Ollama keep-alive)
python scripts/app_hub.py --no-browser          # the hub alone → http://127.0.0.1:8765
python scripts/app_hub.py --open-path "/call?session=example_ai_engineer"   # open straight on Call, bundle preselected
```

## Pages (`scripts/ui/`)

| Route | Page | One job |
|---|---|---|
| `/` | `menu.html` | five tiles + a running-mode banner + a status strip (model loaded? cloud? mic/source) |
| `/call` | `call.html` | setup (bundle + checklist) → live: suggestion (hero POINT) · transcript · optional plan rail; Start listening / suggestions / answers switches; End call → "Review it" link |
| `/train` | `train.html` | setup → one question card + live answer → Next / Finish → scorecard |
| `/review` | `review.html` | captured interviews + reports → grade one (job) or read a report (`?qa=` / `?report=` deep links) |
| `/generate` | `generate.html` | bundle name → JD → your CV (paste or PDF/.txt/.md via `/api/extract`) → company/role → job → bundle summary + "Practise with it" / "Use it for a call" |
| `/settings` | `settings.html` | every D36 knob grouped, with provenance badges + reset |

Shared: `app.css` (tokens copied from `dashboard_ui.html`; `[hidden] { display:none !important }` is
load-bearing) and `common.js` (`api`, `h`, bundle picker, the settings form + ⚙ drawer, `pollJob`,
`renderScorecard`, a tiny textContent-only markdown renderer). Every call-/CV-derived string is painted via
`textContent`, never `innerHTML`. No library, CDN or webfont (SI1).

## Server shape

- **`ModeManager`** — at most one capture mode (`call` | `train`). `start(mode, session)` loads the bundle and
  builds the existing controller with a Namespace filled from settings **at that moment** (`_args`), so a Settings
  change applies from the next start. Starting another mode while one is **capturing** → 409; while idle → the old
  one is shut down. `stop()` = `controller.shutdown()` (SIGINTs the recorder) + a `mode_ended` broadcast that ends
  open sockets. A page reload does NOT stop a mode (deliberate — see gotchas).
- **`/ws`** — `idle` when no mode is open; otherwise the dashboard's hello (+ `mode`, `session`) then the mode's
  EventBus stream and meter ticks, until `mode_ended`.
- **`POST /api/control`** — forwards `{switch, value}` to the active controller via `dashboard.dispatch_control`
  (the same function the legacy `POST /control` uses). Optional `mode` must match the running one. A successful
  call-transcription start appends `{kind: call, session, qa, transcript}` to `scripts/outputs/hub_runs.jsonl` so
  Review can default to the right bundle (the D33 pair log itself is unchanged).
- **`Jobs`** — one running job per kind (`review`, `generate`) in a daemon thread; `GET /api/jobs/<id>` polls it,
  `GET /api/jobs?kind=` recovers a running one after a reload. `run_review_job` (→ `review.review`, honours
  `REVIEW_SYNTHESIZE_RUBRICS`, logs `{kind: review, qa, report}`) and `run_generate_job` (→ `generate_context.run`).
- **Catalogues** — `/api/sessions` (the default first, `blank` last, broken bundles listed not hidden), `/api/interviews`
  (`interview_qa_*.jsonl` joined with the run log), `/api/reports[/<name>]` (strict name regex), `/api/devices`
  (`pactl` sources split into mics / monitors), `/api/status` (active mode, cloud, Ollama `/api/ps`, devices, jobs),
  `/api/config` (D36; GET by group, POST updates, POST reset).

## The default bundle (D37)

`DEFAULT_SESSION = "default_ai_tech"`. `create_hub()` seeds it from `config/bundles/default_ai_tech/bundle.json`
when missing (`ensure_default_bundle`, never overwrites). `resolve_session("")` → the default, used by
`ModeManager.start` and `/api/review`; `list_sessions` puts it first with `default: true`, and the pickers fold it
into the "Default — generic AI tech job (no bundle selected)" option (value `""`). `/api/generate` refuses the name.

`POST /api/extract {filename, data_b64}` → `{text, pages, chars, truncated, kind}` via `scripts/doc_text.py`
(PDF/.txt/.md, ≤10 MB, stateless). `/api/generate` takes an optional `cv_text` (≤60k), copied verbatim into the
bundle's `resume` by `generate_context.run(cv=...)`.

## Input guards (SI1 — nothing here takes a path)

`SESSION_ID_RE` (letters/digits/`_`/`-`, ≤64, no dots/slashes) · `QA_NAME_RE` · `REPORT_NAME_RE` · `/static/`
whitelist of two files · JD 80–60 000 chars · spoken language pl|en · existing bundle name → 409 unless
`overwrite`.

## Verifying without a mic or a model

`tests/test_app_hub.py` drives everything through `TestClient` with a temp sessions/outputs dir (capturing is
simulated by flipping `controller.transcription_on`; jobs run injected callables). For a visual check, a small
harness can wrap `create_hub()`, open a call and push a past transcript through `dashboard.make_sink(state, bus)`
plus canned `suggestion` events — no recorder, no GPU (this is how session 23 verified every view).

## Gotchas

- **Reload ≠ stop.** Capture ends only on End call / Abandon / the menu's "End it", or hub exit.
- **Settings apply at the next mode start**; an env var always wins (🔒 in the UI). `REASONING_BACKEND=cloud`
  without a configured key opens the call on local (SI1).
- **The menu polls `/api/status` only** (every 5 s); bundles and captures are read once per page load —
  `list_sessions` loads every bundle and logs a line per bundle.
- The Ollama keep-alive pinger lives in `launch_copilot.sh`, not here; a hub started directly lets the model idle out.
