# app/nodes/extractor.py

from pathlib import Path
from typing import Dict, Any, List
import os

# PDF
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# DOCX
try:
    import docx
except Exception:
    docx = None


def extract_text_from_pdf(path: str) -> List[str]:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Install with `pip install pypdf`.")
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            # best-effort: append empty string on failure for that page
            pages.append("")
    return pages


def extract_text_from_docx(path: str) -> List[str]:
    if docx is None:
        raise RuntimeError("python-docx is not installed. Install with `pip install python-docx`.")
    doc = docx.Document(path)
    # python-docx doesn't give pages; we return single "page" with full text.
    paragraphs = [p.text for p in doc.paragraphs if p.text is not None]
    # return as one element list to keep consistent page-list shape
    return ["\n".join(paragraphs)]


def extract_text_from_txt(path: str, encoding: str = "utf-8") -> List[str]:
    # treat whole txt as single page
    with open(path, "r", encoding=encoding, errors="replace") as f:
        text = f.read()
    return [text]


def extractor(file_path: str, doc_id: str = None) -> Dict[str, Any]:
    """
    Framework-free extractor node.

    Args:
        file_path: path to the saved uploaded file (from file_loader)
        doc_id: optional, the document id (propagated from file_loader)

    Returns:
        {
            "doc_id": doc_id or None,
            "file_path": str(path),
            "file_type": "pdf"|"docx"|"txt",
            "page_texts": ["...page1...", "...page2...", ...],
            "page_count": int,
            "full_text": "concatenated text"
        }
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = p.suffix.lstrip(".").lower()

    if ext == "pdf":
        page_texts = extract_text_from_pdf(str(p))
    elif ext == "docx":
        page_texts = extract_text_from_docx(str(p))
    elif ext == "txt":
        page_texts = extract_text_from_txt(str(p))
    else:
        raise ValueError(f"Unsupported file extension for extractor: {ext}")

    # Normalize page texts: ensure strings and strip trailing whitespace
    page_texts = [ (t or "").strip() for t in page_texts ]
    full_text = "\n\n".join([t for t in page_texts if t])

    return {
        "doc_id": doc_id,
        "file_path": str(p),
        "file_type": ext,
        "page_texts": page_texts,
        "page_count": len(page_texts),
        "full_text": full_text,
    }
