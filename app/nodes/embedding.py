# app/nodes/embedder.py

from typing import List, Optional

# Try import
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

_MODEL = None

def load_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load a local sentence-transformers model.
    Call this once, or let embed_texts load it automatically.
    """
    global _MODEL
    if SentenceTransformer is None:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers")
    _MODEL = SentenceTransformer(model_name)
    return _MODEL


def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Embed a list of texts using a local sentence-transformers model.
    Returns a list of vectors.
    """
    global _MODEL

    if _MODEL is None:     # lazy load
        load_model(model_name)

    vectors = _MODEL.encode(texts, convert_to_numpy=False, show_progress_bar=False)
    return [list(v) for v in vectors]


def embed_chunks(chunks: List[dict], model_name: str = "all-MiniLM-L6-v2") -> List[dict]:
    """
    Adds 'vector' to each chunk dict.
    """
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, model_name=model_name)

    out = []
    for chunk, vec in zip(chunks, vectors):
        new_chunk = dict(chunk)
        new_chunk["vector"] = vec
        out.append(new_chunk)
    return out
