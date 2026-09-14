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

## Non-waivable floor (🔒)
List here the `SI#` that are safety-floor (privacy/security). Per **D6** these cannot be waived by a
Policy Override — a decision that cannot meet one redesigns or supersedes the cited decision first.

- **SI1** — data locality / privacy of interview audio, transcript, and context bundle (source D14).
- **SI2** — no concealment/anti-detection surface (source D11).

## Provenance
Distilled from the security-bearing rows of `DECISIONS.md` (and any security review the project runs).
This checklist **never** becomes the source of truth — `DECISIONS.md` is (D1). When a decision changes
the security posture, update the cited row here **and** adjust the underlying `D#`.
