import re
from typing import List, Dict

# sentence splitter
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_from_pages(
    page_texts: List[str],
    doc_id: str,
    max_chars: int = 1200,
    overlap_chars: int = 200
) -> List[Dict]:
    """
    page-aware, sentence-preserving chunker.
    - Splits each page into sentences
    - Combines sentences until max_chars reached
    - Overlap_chars preserves context between chunks
    """
    chunks = []
    global_index = 0

    for page_num, page in enumerate(page_texts, start=1):
        sentences = split_sentences(page)
        if not sentences:
            continue

        current = []
        current_len = 0
        sent_idx = 0

        while sent_idx < len(sentences):
            current = []
            current_len = 0

            # Build a chunk
            while sent_idx < len(sentences) and (current_len + len(sentences[sent_idx]) <= max_chars or not current):
                current.append(sentences[sent_idx])
                current_len += len(sentences[sent_idx])
                sent_idx += 1

            text = " ".join(current).strip()
            chunk_id = f"{doc_id}_p{page_num}_c{global_index}"

            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "page": page_num,
                "text": text,
                "chunk_index": global_index,
            })
            global_index += 1

            # Apply overlap: rewind sentences based on approximate chars
            if overlap_chars > 0 and sent_idx < len(sentences):
                overlap_text = text[-overlap_chars:]
                # find the sentence index whose start is near the overlap area
                for back_idx in range(len(sentences)):
                    if sentences[back_idx].startswith(overlap_text[:10]):
                        sent_idx = back_idx
                        break

    return chunks
