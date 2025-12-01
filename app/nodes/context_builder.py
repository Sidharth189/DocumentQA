# app/nodes/context_builder.py
from typing import List, Dict, Any, Tuple

def build_context(chunks: List[Dict[str, Any]], max_chars: int = 2000) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build a context string from ordered chunks until max_chars reached.
    Returns (context_text, used_chunks) where used_chunks is list of chunk dicts included.
    Each chunk dict is expected to have keys: text, doc_id, page
    """
    context_parts: List[str] = []
    used: List[Dict[str, Any]] = []
    total = 0

    for c in chunks:
        txt = (c.get("text") or "").strip()
        if not txt:
            continue
        # prepare citation suffix
        citation = ""
        if c.get("doc_id") is not None:
            citation = f" [{c.get('doc_id')}:{c.get('page')}]"
        part = txt + citation
        part_len = len(part)
        if total + part_len > max_chars:
            # if nothing added yet, still include the first (long) chunk
            if not used:
                context_parts.append(part)
                used.append(c)
            break
        context_parts.append(part)
        used.append(c)
        total += part_len

    context = "\n\n".join(context_parts)
    return context, used
