# app/langchain_config.py
import os
import weaviate
from typing import Optional

# community integrations for embeddings & vectorstores
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Weaviate

# LLM wrapper for Groq
from langchain_groq import ChatGroq

# Attempt to import RetrievalQA (older API) and/or create_retrieval_chain (newer API)
RetrievalQA = None
create_retrieval_chain = None
try:
    # try the class (older / classic usage)
    from langchain.chains.retrieval_qa.base import RetrievalQA  # explicit path
except Exception:
    try:
        from langchain.chains import RetrievalQA  # sometimes exported here
    except Exception:
        RetrievalQA = None

if RetrievalQA is None:
    # try the helper used in newer docs
    try:
        from langchain.chains.retrieval import create_retrieval_chain
    except Exception:
        create_retrieval_chain = None

# config
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
WEAVIATE_INDEX = os.getenv("WEAVIATE_INDEX", "DocumentChunk")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-2b-instant")

def get_weaviate_client() -> weaviate.WeaviateClient:
    # v4 client: use .connect_to_local() with URL
    return weaviate.connect_to_local(WEAVIATE_URL)


def get_embeddings():
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore():
    client = get_weaviate_client()
    embeddings = get_embeddings()
    return Weaviate(
        client,
        index_name=WEAVIATE_INDEX,
        text_key="text",
        attributes=["doc_id", "page"],
        embedding=embeddings
    )


def get_retriever(k: int = 5):
    vs = get_vectorstore()
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})


def get_llm(model: Optional[str] = None):
    m = model or GROQ_DEFAULT_MODEL
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment")
    return ChatGroq(api_key=GROQ_API_KEY, model_name=m)


def get_qa_chain(k: int = 5, model: Optional[str] = None):
    """
    Returns a chain-like callable object that you can invoke with {"query": "..."}.
    Uses RetrievalQA class if available, otherwise falls back to create_retrieval_chain helper.
    """
    retriever = get_retriever(k=k)
    llm = get_llm(model)

    if RetrievalQA is not None:
        # classic pattern
        return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    if create_retrieval_chain is not None:
        # newer helper: create_retrieval_chain(llm, retriever, chain_type="stuff", return_source_documents=True)
        # signature may vary across versions; this tries the common signature.
        try:
            chain = create_retrieval_chain(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True)
            return chain
        except TypeError:
            # older variants might take positional args
            return create_retrieval_chain(llm, retriever, chain_type="stuff", return_source_documents=True)

    # If neither is available, raise a clear error
    raise ImportError(
        "Could not find RetrievalQA or create_retrieval_chain in your langchain installation. "
        "Install a compatible langchain (and langchain-community) version. "
        "See README or set appropriate packages."
    )
