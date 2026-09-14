# Contributing

This is a personal / portfolio project, shared mainly as a reference. Issues and small PRs are
welcome, but there is no roadmap commitment.

## Development setup
See [SETUP.md](SETUP.md) for the full environment (GPU, Ollama, audio stack). For code-only changes
you can skip the audio/model setup and just run the deterministic test suite:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/          # model/live tests self-skip when Ollama is unreachable
ruff check .
```

## Conventions
- **Python 3.11+**, type hints on function signatures, functions over classes unless state is needed.
- **No secrets in code.** Everything sensitive comes from the environment; only `config/.env.example`
  is committed. Real interview data lives under `scripts/inputs/` and is gitignored — never commit it.
- **Determinism first:** if a task is decidable by code (parsing, counting, a rule), write a script
  rather than calling a model.
- **Design decisions** are recorded in [`design/DECISIONS.md`](design/DECISIONS.md); if you change
  behaviour that a `D#` describes, update the reasoning there too.
- Keep the UI dependency-free (no CDN, no webfont) — a page rendering an interview transcript must
  make no outbound requests (SI1).

## Tests
- New behaviour needs a test. Prefer a deterministic test (no GPU, no model) where possible; the
  suite already separates those from the `@live` tests that make a real model call.
- Run `ruff check .` and `pytest tests/` before opening a PR.

## Scope & safety
This tool is **disclosed-use only** (D11): please do not propose stealth, anti-detection, or
proctor-evasion features — they will not be accepted (SI2).
