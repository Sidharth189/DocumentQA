from typing import Dict, Any, List, Tuple
from .langchain_config import get_qa_chain
from .nodes.file_loader import file_loader_from_bytes
from .nodes.extract import extractor
from .nodes.clean_data import data_cleaner
from .nodes.chunker import chunk_from_pages
from .nodes.embedding import embed_chunks
from .nodes.vector_upsert import vector_upsert, create_schema

# Simple helper: ingest a file end-to-end and upsert to Weaviate
def ingest_document(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Save uploaded file, extract, sanitize, chunk, embed, and upsert vectors.
    Returns metadata about ingestion.
    """
    loader = file_loader_from_bytes(file_bytes=file_bytes, filename=filename)
    doc_id = loader.get("doc_id")
    file_path = loader.get("file_path")

    extract_res = extractor(file_path, doc_id=doc_id)
    page_texts = extract_res.get("page_texts", [])

    san = data_cleaner(page_texts, doc_id=doc_id)
    cleaned_pages = san.get("cleaned_page_texts", [])

    chunks = chunk_from_pages(cleaned_pages, doc_id=doc_id)
    embedded = embed_chunks(chunks)

    objects = []
    for c in embedded:
        objects.append({
            "id": c["chunk_id"],
            "vector": c["vector"],
            "properties": {
                "text": c["text"],
                "doc_id": c["doc_id"],
                "page": c.get("page")
            }
        })

    # create_schema()  # uncomment if you want ingestion to create/reset schema automatically
    vector_upsert(objects)

    return {
        "doc_id": doc_id,
        "file_path": file_path,
        "page_count": extract_res.get("page_count"),
        "chunks": len(chunks)
    }


# Query function that uses the LangChain RetrievalQA chain
def answer_query(query: str, top_k: int = 5, model: str | None = None) -> Dict[str, Any]:
    """
    Runs RetrievalQA chain and returns answer + source documents metadata.
    """
    chain = get_qa_chain(k=top_k, model=model)
    res = chain({"query": query})

    answer = res.get("result") or res.get("answer") or res.get("output_text") or ""
    source_docs = res.get("source_documents") or []

    sources = []
    for d in source_docs:
        md = getattr(d, "metadata", {}) or {}
        sources.append({
            "doc_id": md.get("doc_id"),
            "page": md.get("page"),
            "text_snippet": (getattr(d, "page_content", "") or "")[:400]
        })

    return {"answer": answer, "sources": sources}
