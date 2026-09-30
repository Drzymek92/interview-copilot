"""D37 — JD/CV files → text (PDF via PyMuPDF, .txt/.md), locally."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import doc_text  # noqa: E402

fitz = pytest.importorskip("fitz")


def make_pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def test_pdf_text_is_extracted_across_pages():
    out = doc_text.extract_text(make_pdf(["Jan Kowalski - AI Engineer", "Skills: Python, RAG"]), "cv.PDF")
    assert out["kind"] == "pdf" and out["pages"] == 2
    assert "Jan Kowalski" in out["text"] and "Skills: Python, RAG" in out["text"]


def test_image_only_pdf_says_why():
    with pytest.raises(doc_text.DocTextError, match="scanned"):
        doc_text.extract_text(make_pdf([""]), "scan.pdf")


def test_garbage_named_pdf_is_refused():
    with pytest.raises(doc_text.DocTextError, match="PDF"):
        doc_text.extract_text(b"not a pdf at all", "fake.pdf")


def test_text_and_markdown_decode_and_clean():
    out = doc_text.extract_text("Line 1  \r\n\r\n\r\n\r\nLine 2\n".encode("utf-8"), "jd.md")
    assert out["text"] == "Line 1\n\nLine 2" and out["kind"] == "text"
    assert doc_text.extract_text("zażółć".encode("latin-1", "replace"), "x.txt")["chars"] > 0


@pytest.mark.parametrize("name,data,match", [
    ("cv.docx", b"x", "unsupported"),
    ("cv.txt", b"", "empty"),
    ("cv.txt", b"x" * (doc_text.MAX_BYTES + 1), "10 MB"),
])
def test_refusals(name, data, match):
    with pytest.raises(doc_text.DocTextError, match=match):
        doc_text.extract_text(data, name)


def test_long_text_is_truncated_and_flagged():
    out = doc_text.extract_text(("word " * 20000).encode(), "long.txt")
    assert out["truncated"] is True and out["chars"] == doc_text.MAX_CHARS


def test_read_path(tmp_path):
    p = tmp_path / "jd.pdf"
    p.write_bytes(make_pdf(["Senior LLM Engineer"]))
    assert "Senior LLM Engineer" in doc_text.read_path(p)
