# services/extractor.py
"""
Text extraction utilities using pypdf (PdfReader) and python-docx (Document).
Plain .txt files are read with normal file handling.

Supported extensions: .pdf, .docx, .txt

Returns an empty string on extraction failure.
"""

import os
from typing import Optional

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # graceful fallback

try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None  # graceful fallback


def extract_text_from_file(path: str) -> Optional[str]:
    """
    Extract text from PDF, DOCX or TXT files.

    Args:
        path: filesystem path to the file

    Returns:
        extracted text as a single string (may be empty) or None on catastrophic error
    """
    _, ext = os.path.splitext(path.lower())
    try:
        if ext == ".pdf":
            if PdfReader is None:
                raise RuntimeError("pypdf (PdfReader) is not installed")
            return _extract_pdf(path)
        elif ext == ".docx":
            if DocxDocument is None:
                raise RuntimeError("python-docx (Document) is not installed")
            return _extract_docx(path)
        elif ext == ".txt":
            return _extract_txt(path)
        else:
            return ""
    except Exception as e:
        # keep behavior simple for the caller: log and return empty string
        print(f"[extractor] error extracting {path}: {e}")
        return ""


def _extract_pdf(path: str) -> str:
    text_parts = []
    reader = PdfReader(path)
    # PdfReader.pages is iterable
    for page in reader.pages:
        try:
            page_text = page.extract_text()
        except Exception as e:
            # continue on page-level failures
            print(f"[extractor] pdf page extract error ({path}): {e}")
            page_text = None
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def _extract_docx(path: str) -> str:
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)


def _extract_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()
