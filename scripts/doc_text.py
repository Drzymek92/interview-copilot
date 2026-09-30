"""Plain text out of a job description or CV file — PDF, Markdown or text (D37).

    python scripts/doc_text.py path/to/cv.pdf          # print the extracted text

Used by `generate_context.py --jd/--cv` and by the hub's Generate view (`POST /api/extract`), so a
PDF is read the same way from the CLI and from the browser. Local only (SI1): PyMuPDF parses the
file in-process, nothing is uploaded anywhere. Scanned (image-only) PDFs have no text layer — the
caller gets a clear error instead of an empty string, since OCR is out of scope.
"""

from __future__ import annotations

import sys
from pathlib import Path

TEXT_SUFFIXES = (".txt", ".md", ".markdown")
SUPPORTED_SUFFIXES = (".pdf", *TEXT_SUFFIXES)
MAX_BYTES = 10 * 1024 * 1024           # a CV/JD over 10 MB is not a CV/JD
MAX_CHARS = 60_000                     # the same ceiling the Generate form enforces


class DocTextError(ValueError):
    """The file could not be turned into usable text (surfaced to the user verbatim)."""


def _pdf_text(data: bytes) -> tuple[str, int]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:        # pragma: no cover - pinned in config/requirements.txt
        raise DocTextError("PDF support needs PyMuPDF: pip install pymupdf") from exc
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocTextError(f"not a readable PDF ({exc.__class__.__name__})") from exc
    with doc:
        if doc.needs_pass:
            raise DocTextError("the PDF is password-protected — export an unprotected copy")
        pages = [page.get_text("text") for page in doc]
    return "\n\n".join(p.strip() for p in pages if p.strip()), len(pages)


def _clean(text: str) -> str:
    # Normalise line endings, drop trailing spaces and runs of 3+ blank lines (PDF layout noise).
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out: list[str] = []
    blank = 0
    for ln in lines:
        blank = blank + 1 if not ln else 0
        if blank <= 1:
            out.append(ln)
    return "\n".join(out).strip()


def extract_text(data: bytes, filename: str) -> dict:
    """{text, pages, chars, truncated, kind} for one uploaded file. Raises DocTextError."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocTextError(f"unsupported file type {suffix or '(none)'} — use PDF, .txt or .md")
    if len(data) > MAX_BYTES:
        raise DocTextError("the file is larger than 10 MB")
    if not data:
        raise DocTextError("the file is empty")
    if suffix == ".pdf":
        text, pages = _pdf_text(data)
        kind = "pdf"
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        pages, kind = 1, "text"
    text = _clean(text)
    if not text:
        raise DocTextError("no text found — a scanned PDF has no text layer (OCR is not supported)"
                           if kind == "pdf" else "the file has no text")
    truncated = len(text) > MAX_CHARS
    if truncated:
        text = text[:MAX_CHARS]
    return {"text": text, "pages": pages, "chars": len(text), "truncated": truncated, "kind": kind}


def read_path(path: Path) -> str:
    """The text of a file on disk (the CLI path)."""
    return extract_text(path.read_bytes(), path.name)["text"]


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2].strip())
        sys.exit(2)
    try:
        print(read_path(Path(sys.argv[1])))
    except (OSError, DocTextError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
