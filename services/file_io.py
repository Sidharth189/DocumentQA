# services/file_io.py
import os
import shutil
import uuid
from typing import List
from fastapi import UploadFile

UPLOAD_DIR = "uploads"
ALLOWED_EXT = {".pdf", ".docx", ".txt"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT

def save_uploads(files: List[UploadFile]) -> List[str]:
    saved = []
    for f in files or []:
        if not allowed_file(f.filename):
            continue
        token = uuid.uuid4().hex
        safe_name = f"{token}__{os.path.basename(f.filename)}"
        dest = os.path.join(UPLOAD_DIR, safe_name)
        with open(dest, "wb") as out:
            f.file.seek(0)
            shutil.copyfileobj(f.file, out)
        saved.append(dest)
    return saved

def cleanup_files(paths: List[str]):
    for p in paths or []:
        try:
            os.remove(p)
        except Exception:
            pass
