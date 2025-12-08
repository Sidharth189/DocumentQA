# services/chunker.py
from typing import List, Dict
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

def chunk_documents(text_entries: List[Dict]) -> List[LCDocument]:
    """
    text_entries: List of {"source": filename, "text": full_text}
    returns: list of langchain.Document with metadata {"source":..., "chunk": idx}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    docs = []
    for i, entry in enumerate(text_entries):
        chunks = splitter.split_text(entry["text"])
        for j, c in enumerate(chunks):
            meta = {"source": entry.get("source", f"doc_{i}"), "chunk": j}
            docs.append(LCDocument(page_content=c, metadata=meta))
    return docs
