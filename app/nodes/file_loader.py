# app/nodes/file_loader.py

import uuid
from pathlib import Path
from typing import Dict, Any

TMP_UPLOAD_DIR = Path("data/temp/uploads")
TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def file_loader_from_bytes(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Save uploaded file bytes to a temporary path and return metadata.

    Args:
        file_bytes: raw bytes of the uploaded file
        filename: original filename provided by the client

    Returns:
        dict with keys: doc_id, file_path, file_type, filename
    """
    if not file_bytes:
        raise ValueError("Uploaded file is empty")

    if not filename or "." not in filename:
        raise ValueError("Filename must include an extension")

    # Normalize extension
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in {"pdf", "docx", "txt"}:
        raise ValueError(f"Unsupported file type: {ext}")

    # Create deterministic unique id for the document
    doc_id = str(uuid.uuid4())

    # Save to tmp/uploads using the doc_id as the name
    tmp_name = f"{doc_id}.{ext}"
    tmp_path = TMP_UPLOAD_DIR / tmp_name

    # Write bytes to disk
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    return {
        "doc_id": doc_id,
        "file_path": str(tmp_path),
        "file_type": ext,
        "filename": filename,
    }
