# app/nodes/vector_search.py
from typing import List, Dict, Any, Optional
from app.nodes.embedder import embed_texts
import weaviate
import os

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

def _get_client():
    return weaviate.Client(WEAVIATE_URL)

def search_query(query: str, top_k: int = 5, model_name: str = "all-MiniLM-L6-v2") -> List[Dict[str, Any]]:
    """
    Embed the query and perform a near-vector search in Weaviate.
    Returns a list of dicts: { "text", "doc_id", "page", "score" } (score may be None if not present)
    """
    vec = embed_texts([query], model_name=model_name)[0]

    client = _get_client()
    res = client.query.get("DocumentChunk", ["text", "doc_id", "page"]).with_near_vector({"vector": vec}).with_limit(top_k).do()

    hits = []
    try:
        items = res["data"]["Get"]["DocumentChunk"]
    except Exception:
        return hits

    for it in items:
        text = it.get("text")
        doc_id = it.get("doc_id")
        page = it.get("page")
        # some responses include _additional with certainty/distance; handle both
        additional = it.get("_additional", {}) if isinstance(it, dict) else {}
        score = additional.get("certainty") if additional.get("certainty") is not None else additional.get("distance")
        hits.append({"text": text, "doc_id": doc_id, "page": page, "score": score})

    return hits
