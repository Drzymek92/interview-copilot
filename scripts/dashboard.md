# `scripts/dashboard.py` — the D18 dashboard + the P5 meter

Subtask #322. Read this instead of scanning the script (~710 lines).
Companions: `scripts/reasoning.md` (the loop it drives), `scripts/live_transcribe.md` (the producer
it never touches), `scripts/replay_transcript.py` (how it is driven without a live call).

## QUICKSTART

```
cd ~/Desktop/Claude_Projects/projects/interview_copilot
ollama ps                                          # the GPU is shared — check before a call
python scripts/dashboard.py --session example_ai_engineer --watch --wait-seconds 120
```
…then start the transcript loop in a second terminal (`live_transcribe.md` quickstart) and open
**http://127.0.0.1:8765**. Ctrl-C to stop; it never touches the recording.

No model, no GPU (transcript + meter only — the practice/dry-run mode):
```
python scripts/dashboard.py --no-suggestions --follow scripts/outputs/live_transcript_<stamp>.txt
```
The ungated comparison (D23 off, every detected question fires): add `--no-salience`.

Drive it from the real 42-min call without a live interview:
```
python scripts/replay_transcript.py scripts/outputs/live_transcript_20260902_100033.txt --speed 8
python scripts/dashboard.py --session example_ai_engineer --follow scripts/outputs/replay_transcript_<stamp>.txt \
       --ceiling-seconds 3.875        # <- the number the replay printed. See "Time-compressed replays".
```

## Single-app mode (`--app`, #607) — the desktop shortcut

`python scripts/dashboard.py --app --session <id>` is the one process the desktop shortcut
(`scripts/launch_copilot.sh` → `~/Desktop/interview-copilot.desktop`) launches. It adds an
`AppController` that OWNS the recorder and a loopback-only `POST /control` endpoint behind three
on-screen switches. Everything else on this page is unchanged; without `--app` none of it exists.

- **The controller is not a governance seat (fw:D2)** — it starts/stops processes and flips flags,
  it decides nothing. **D19 holds**: the worker still only *tails* the transcript; the controller
  manages the recorder's lifecycle out of band and never writes the file.
- **Transcription switch = a TRUE capture stop** (the user's explicit choice). ON spawns
  `live_transcribe.py` with `COPILOT_SOURCE`/`COPILOT_MIC` and waits (≤`COPILOT_RECORDER_WAIT_SECONDS`)
  for its new stamped transcript to appear, then starts the ambient + provisional tails on it. OFF
  SIGINTs the recorder (we hold the `Popen` — **never `pgrep`**, per the lab process-kill hazard) and
  sets the run's `stop_event`, ending the tails. Each ON rotates to a **new** file, so the tails are
  restarted — that is what `stop_event` on `follow_transcript`/`tail_partial` is for (`None` = the
  CLI/replay path, byte-identical).
- **Suggestions switch** = `reasoning.Controls.suggestions_on`, read **per line** in `run_ambient`:
  off skips trigger/gate/fire (no model, no GPU) while the transcript + meter still flow.
- **Answers switch** = `Controls.backend` (`local` ↔ `cloud`). SI1 kept: greyed out when
  `llm_client.cloud_ready()` is false, the "+ api" track is **red** not green, and the
  `announce_backend` egress banner is shown on flip. No egress happens until a suggestion fires.
- **Meter** shows an explicit **⏸ transcription off** state (JS `renderMeter`) so it never climbs to
  a false "overdue" while capture is intentionally off.
- **Gotcha:** `--mic auto` often resolves the **webcam** mic, not your external USB mic — set `COPILOT_MIC`
  explicitly for a real call (same trap as `live_transcribe.py --mic auto`).

## Shape

```
live_transcribe.py ──fsync'd lines──> live_transcript_*.txt
                                              │  (D19 seam — tail, no callback, no edit upstream)
                                              ▼
                          reasoning.run_ambient(..., sink=…)
                                              │  dicts: line · gate · skip · suggestion
                                              ▼
                       make_sink → DashboardState → EventBus → websocket → dashboard_ui.html
                                              ▲
                                    _tick_loop (METER_TICK_SECONDS)
```

- **`run_ambient` knows nothing about a UI.** The `sink` is optional and with `sink=None` the CLI
  path is byte-identical. `make_sink()` is the only place that knows both halves.
- **`DashboardState` is the whole view**, guarded by one lock. Every mutation *returns* the events
  to broadcast, so the lock is never held across a socket write.
- **`EventBus`** fans out from worker threads to websockets via `loop.call_soon_threadsafe`. Each
  subscriber has a bounded queue and the **oldest** message is dropped on overflow — a wedged
  browser must not stall the tail thread.
- Two threads write: the tail thread, and the suggestion's own streaming thread.

## The P5 meter — `meter_state()`

A pure function: `(last_line_at, now, ceiling) -> MeterState`. It takes the clock as an argument
and reads nothing, which is why it is fully tested. It is computed **here, in Python, and pushed**;
the browser derives nothing, so the number read under pressure has one implementation.

| state | when | what it says |
|---|---|---|
| `cold` | no line has arrived | "waiting for the first transcript line" — **no promise** |
| `waiting` | `elapsed <= ceiling` | "last line N s ago · next due within ≤M s" |
| `overdue` | `elapsed > ceiling` | "last line N s ago · overdue by K s (ceiling C s)" — **withdraws** the promise |

**Rounding is deliberately unflattering.** The age is floored and the remaining time is *ceiled*, so
the bound is one the arrival can only beat, never miss by a rounding artefact.

### The ceiling is NOT `SEGMENT_MAX_SECONDS`

A line cannot reach the screen until its segment closes **and** is decoded. Scoring the real call's
109 arrival gaps as `end + decode` (D25's own model, `0.248 + 0.0144·audio_s`):

| ceiling | holds |
|---|---|
| 30.0 s (`SEGMENT_MAX_SECONDS` alone) | **84/109** — 54% of segments end at the cap and land at 30.1–30.4 s |
| **31.0 s** (`+ METER_DECODE_ALLOWANCE_SECONDS`) | **108/109** ← shipped |

The single true overrun is **43.2 s** — a silence *between* segments, which no segment cap bounds
and the meter must not pretend to. That is why the `overdue` state exists and is demonstrated
rather than assumed.

### Time-compressed replays

`--speed N` divides every gap by N, so the meter is only honest if its ceiling is divided too.
`replay_transcript.py` prints the exact `--ceiling-seconds` to pass. Forgetting it makes the meter
promise something false — the one failure mode this panel exists to avoid.

## What the panels will and will not claim

- **Transcript** — two-sided from the `them:`/`you:` tags. An **untagged** line renders as
  "Untagged (assumed interviewer)", not silently as `them`. It also renders #326's known defect
  (the 30 s cap merges Q&A exchanges and can mis-tag) rather than papering over it.
- **Suggestions** — D23-gated only. A dropped question shows as a counter, never as a non-answer.
  A finished suggestion is badged **stale** after `SUGGESTION_STALE_LINES` new lines; a *streaming*
  one never is, because it is answering the current question by definition.
- **Counters** — `dropped` is the D23 gate and **only** the gate. A `SUGGESTION_COOLDOWN_SECONDS`
  skip is counted and worded separately ("held by cooldown"); folding it in made the ungated panel
  report "2 dropped" with the gate switched off.
- **Plan** — read-only. `mentioned` = a `done_signals` phrase was literally said. It is **not** a
  claim the step was covered, the legend says so, and the word "covered" does not appear. P3/G7 is
  open and no auto-covered state machine is invented here.
- **Connection** — if the ticks stop for >2 s the page greys out and says the content may already be
  wrong. A silent socket is otherwise indistinguishable from a calm interview.
- **Provenance** — `detect_mode()` reads the transcript's own header, so a replay is labelled
  "REPLAY — not a live call" whatever `--mode` claims.

## Gotchas

- **`from __future__ import annotations` + FastAPI.** The websocket handler's `socket: WebSocket`
  annotation is a *string* FastAPI resolves against **module** globals. Importing `fastapi` inside
  `create_app()` made the name a local, FastAPI took `socket` for an unknown query parameter, and
  **every** browser handshake was closed with a bare HTTP 403 — while all unit tests passed. Keep
  the import at module level. `TestServedEndToEnd` connects for real so this cannot regress.
- **`uvicorn` ships no websocket implementation.** Without `websockets` (pinned) `/ws` is accepted
  and immediately closed, which looks exactly like a dead dashboard.
- **`newest_transcript()` globs `live_transcript_2*.txt`.** Replays therefore write
  `replay_transcript_*.txt`, so a rehearsal left in `scripts/outputs/` can never be what `--watch`
  picks up as the newest run.
- **The GPU lease board charges an already-resident Ollama model as *unmanaged* VRAM.** A
  `--vram 11800` lease cannot be granted while the 14B is loaded — it double-counts and queues
  forever. Either declare the incremental claim while the model is resident, or `evict-ollama`
  first and let the lease load it cold.
- **`SEGMENT_MAX_SECONDS` is load-bearing twice** — the meter's ceiling and D21's accuracy trade.
  It is not a free knob here; #400 owns that lever with a specified experiment.

## Knobs (CFG — `config/settings.py`, CLI > env > config > default)

`DASHBOARD_HOST` (loopback only, enforced) · `DASHBOARD_PORT` · `DASHBOARD_MAX_LINES` ·
`METER_ENABLED` · `METER_TICK_SECONDS` · `METER_CEILING_SECONDS` (0 = derive) ·
`METER_DECODE_ALLOWANCE_SECONDS` · `SUGGESTION_STALE_LINES` · `PLAN_MENTION_TRACKING` ·
`DASHBOARD_SHOW_PARTIALS` (`--no-partials`).

## The D25 provisional line (#399)

The dashboard tails `live_transcript_<stamp>.partial` beside the `.txt` — same format, same
`follow_transcript` seam, path **derived** by `partial_path_for()` so the two cannot name different
runs. It lives in **one slot** (`ProvisionalView`), pinned below the last transcript line and
rendered dashed, dimmed and labelled *provisional*, with no animation at all.

- It is **never a transcript line**: it does not move `counters.lines`, the ring buffer, the
  plan-mention scan or the suggestion-staleness sequence — and it does not move the meter, because
  P5-A measures the wait for a line that has *arrived* and a provisional has not.
- **Cleared by any final at or past its start**: `==` superseded, `>` orphaned. The clear is
  published *after* the line event, so no frame shows neither.
- **Expired** by one owner thread past the meter ceiling. Not in `tick()` — `tick()` runs once per
  connected browser, so a mutation there clears the slot for one browser and strands the rest.
- `from_start` defaults to **False** for the `.partial` and True for the `.txt`: the transcript is
  the record and a late browser is owed all of it; a provisional is only ever a claim about *now*.
- `/healthz` carries the tally (`shown` / `superseded` / `orphaned` / `expired` / `showing`) so a
  headless run is auditable without a browser.
- A missing `.partial` (`PARTIAL_DECODE_ENABLED=0`) is a supported run, not an error.

## Security (SI1/SI2 — enforced, not documented)

- `_assert_loopback()` **refuses to start** on a non-loopback host, with a named cause. Tested.
- The page loads **no** font, script or stylesheet from a network. Tested.
- Transcript text enters the DOM via `textContent` only — never `innerHTML`. Tested.
- The disclosure banner is permanent; there is no hide/opacity/click-through/always-on-top
  affordance and there must never be one (D11). Tested.

## 2026-09-14 — history panel bound
`.sugg-history` is capped at 40% of the section and scrolls internally; `.sugg-body` keeps at least 45%. The head is a collapse button (`copilot.historyCollapsed` in localStorage). Before this a long history squeezed the live suggestion out of view.
