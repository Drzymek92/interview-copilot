# Security Invariants — the reconciliation checklist (applies D10)

**Purpose.** This is the project's canonical list of load-bearing **security decision data points**.
Any future decision that touches a security-sensitive surface — **data privacy/locality, transport,
credentials/keys, or a remote/command/agent surface** — **must be reconciled against this list before
implementation** (the gate is installed by **D10**; the routine step lives in
`ROUTINE_add_decision.md`). This doc **applies** the security-bearing decisions in `DECISIONS.md` and
never restates them; the canonical statements live there (D1).

**Starts (almost) empty — it grows with the project.** A brand-new project has few security
invariants. Add an `SI#` row the first time a decision *establishes* one (e.g. "data class X is
local-only", "surface Y is default-deny", "credential Z has a revocation path"), and cite the `D#`
that set it. If the project has **no** security-sensitive surface, leave this a stub and record the
reconciliation step as a one-time `N/A` in `agent/project.md`.

**How to use (at decision time).** For each invariant the new decision could touch, record one line
in the decision's rationale or its `design/NN` note:
**`SI# — Reconciled: <how>`** or **`SI# — N/A: <why>`**.
Mark an invariant **🔒 non-waivable** when it encodes a safety floor (privacy/security): per **D6** a
safety rule cannot be waived by a Policy Override, so a decision that cannot meet a 🔒 invariant does
**not** ship — it redesigns, or explicitly **supersedes** the cited decision in `DECISIONS.md` first.

## Invariants
<!-- Add one row per security invariant as decisions establish them; each cites its source D#.
     Replace the two example rows below with real ones (or delete them for a stub). -->

| SI | 🔒 | Invariant (confirm the new decision honors it) | Source | Confirm question |
|----|----|-----|--------|------------------|
| SI1 | 🔒 | **Interview audio, transcript, and the session context bundle are LOCAL-BY-DEFAULT.** They leave the machine only via a provider the user has *explicitly* selected (cloud LLM/STT); a **fully-local mode** (Ollama + local Whisper) must always exist and no egress may be silent. Resume/JD/company data and interviewer speech are the sensitive class. | D14 | Does the decision add or default-enable an egress of audio/transcript/context? Is it opt-in-visible? Does local-only still work? |
| SI2 | 🔒 | **No anti-detection / concealment surface.** The tool must not add any feature whose purpose is to hide its presence or output from the interviewer, screen-share, or proctoring. Disclosure is a safety floor, not a toggle. | D11 | Does the decision add a "stealth"/hide/undetectable capability? If so it does not ship. |

## Offline practice track (D29/D30/D31) — reconciled against SI1
The practice track adds new **local** data classes and reuses the disclosed surface, so it introduces
no new invariant — it is **reconciled against the existing SI1/SI2**:
- **SI1 — Reconciled:** JD text, generated context bundles (**D29/D30**), spoken practice answers,
  their transcripts, and the scores/report (**D31**) are the same sensitive class as interview data and
  are **local-by-default**. Generation and scoring default to local Ollama; any cloud model is opt-in +
  `announce_backend` banner; the full local-only path works. The `--training` server keeps the D18
  127.0.0.1 bind and the `POST /control` training switch adds no network surface.
- **SI2 — N/A:** the trainer is a disclosed practice tool; it adds no concealment/anti-detection surface.

## Live Q&A capture + post-interview review (D32/D33/D34) — reconciled against SI1
Capturing real-interview question↔answer pairs and reviewing them adds a new **local** artifact and
reuses the disclosed surface, so it mints no new invariant:
- **SI1 — Reconciled:** the captured pairs (`interview_qa_<stamp>.jsonl`, **D33**) are the same
  sensitive class as the transcript they derive from and are **local-by-default**, sharing the run
  stamp; the review scoring defaults to local Ollama (cloud opt-in + `announce_backend` banner) and the
  report is written locally (**D32**). The `POST /control "review"` action keeps the D18 127.0.0.1 bind
  and adds no network surface (**D34**). No egress is added on any default path.
- **SI2 — N/A:** the review is a disclosed, local feedback surface; it adds no concealment surface.

## Non-waivable floor (🔒)
List here the `SI#` that are safety-floor (privacy/security). Per **D6** these cannot be waived by a
Policy Override — a decision that cannot meet one redesigns or supersedes the cited decision first.

- **SI1** — data locality / privacy of interview audio, transcript, and context bundle (source D14).
- **SI2** — no concealment/anti-detection surface (source D11).

## Provenance
Distilled from the security-bearing rows of `DECISIONS.md` (and any security review the project runs).
This checklist **never** becomes the source of truth — `DECISIONS.md` is (D1). When a decision changes
the security posture, update the cited row here **and** adjust the underlying `D#`.

## Hub + saved settings (D35/D36) — reconciled against SI1/SI2
One server now hosts the menu and every mode view, and a settings file can change runtime behaviour, so both are
reconciled here:
- **SI1 — Reconciled (D35):** `app_hub.py` keeps the D18 bind (`_assert_loopback`, 127.0.0.1 only); every new
  endpoint (`/api/*`, `/ws`, `/static/*` whitelist) is same-origin loopback and adds no network surface. Report
  and capture files are served only by strict name patterns (no paths). Review/generate jobs default to the
  local backend; a cloud backend is opt-in, still gated by `cloud_ready()`, and the job shows the
  `announce_backend` egress banner before any result.
- **SI1 — Reconciled (D36):** `config/user_settings.json` is local + gitignored and holds no secrets — the
  registry has no host/port/credential knob, and `settings.load_saved_settings` refuses `DASHBOARD_HOST`/`PORT`
  and any credential-suffixed key even if hand-written into the file. Selecting cloud through a saved setting
  changes nothing until a mode starts, and a cloud setting without a configured key falls back to local.
- **SI2 — N/A (D35/D36):** the D11 disclosure is in the top bar of every hub page (test-asserted); no page adds a
  hide, opacity, click-through or always-on-top control.

## CV / PDF inputs + default bundle (D37) — reconciled against SI1/SI2
- **SI1 — Reconciled:** the CV is the most personal data class the tool now holds; it is parsed **in-process**
  (PyMuPDF via `scripts/doc_text.py`), posted only to the loopback `POST /api/extract` (stateless — nothing is
  written), and stored only in the local bundle under the gitignored `scripts/inputs/`. It leaves the machine only
  if the user picks a cloud backend for a mode, which stays opt-in and announced. The default bundle is generic and
  holds no personal data.
- **SI2 — N/A:** no concealment surface.
