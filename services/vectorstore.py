# services/vectorstore.py
import logging
from typing import Optional, List
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LCDocument

CHROMA_DIR = "chroma_db"


# Optional imports for handling quota errors and local fallback
try:
    from langchain_google_genai._common import GoogleGenerativeAIError
except Exception:
    GoogleGenerativeAIError = None

try:
    from google.api_core.exceptions import ResourceExhausted
except Exception:
    ResourceExhausted = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
 
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None


class LocalSentenceTransformerEmbeddings:
    """Simple local embeddings wrapper using sentence-transformers.

    Implements `embed_documents` and `embed_query` used by Chroma.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]):
        embs = self.model.encode(texts, convert_to_numpy=True)
        # Convert numpy arrays (possibly float32) to Python floats list
        return [e.astype(float).tolist() for e in embs]

    def embed_query(self, text: str):
        e = self.model.encode([text], convert_to_numpy=True)[0]
        return e.astype(float).tolist()


class LocalTFIDFEmbeddings:
    """Lightweight local embedding emulation using TF-IDF.

    Produces fixed-length dense vectors by converting TF-IDF sparse vectors
    to dense lists. This is not a semantic embedding like sentence-transformers,
    but it provides a usable local fallback when remote embeddings are unavailable.
    """

    def __init__(self, max_features: int = 4096):
        if TfidfVectorizer is None:
            raise RuntimeError("scikit-learn is not installed for TF-IDF fallback")
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

    def _ensure_fitted(self, texts):
        if not self._fitted:
            self.vectorizer.fit(texts)
            self._fitted = True

    def embed_documents(self, texts: List[str]):
        # Fit vectorizer on provided texts (local-only) and return dense lists
        self._ensure_fitted(texts)
        X = self.vectorizer.transform(texts)
        arr = X.toarray()
        return [a.tolist() for a in arr]

    def embed_query(self, text: str):
        if not self._fitted:
            # If vectorizer hasn't been fitted yet, fit on the query only
            self.vectorizer.fit([text])
            self._fitted = True
        v = self.vectorizer.transform([text]).toarray()[0]
        return v.tolist()


def build_chroma_collection(embedding: Embeddings, docs: List[LCDocument], collection_name: Optional[str] = None) -> Chroma:
    """
    Create or replace a Chroma collection with the provided documents.
    Tries the provided `embedding` (e.g., Gemini). On quota / rate-limit errors,
    will attempt to fall back to a local `sentence-transformers` model if available.
    Returns the Chroma vectorstore instance.
    """
    collection_name = collection_name or "doc_collection"

    try:
        chroma = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=CHROMA_DIR,
            collection_name=collection_name,
        )
        chroma.persist()
        return chroma

    except Exception as e:
        # Detect quota / rate limit errors from Google GenAI adapter
        is_quota_error = False
        if GoogleGenerativeAIError is not None and isinstance(e, GoogleGenerativeAIError):
            is_quota_error = True
        if ResourceExhausted is not None and isinstance(e, ResourceExhausted):
            is_quota_error = True

        logging.exception("Embedding/vectorstore creation failed: %s", e)

        if is_quota_error:
            logging.warning("Detected quota/rate-limit error from remote embeddings provider.")
            # Prefer sentence-transformers (semantic embeddings) if available
            if SentenceTransformer is not None:
                logging.info("Falling back to local sentence-transformers embeddings (semantic).")
                local_emb = LocalSentenceTransformerEmbeddings()
                chroma = Chroma.from_documents(
                    documents=docs,
                    embedding=local_emb,
                    persist_directory=CHROMA_DIR,
                    collection_name=collection_name,
                )
                chroma.persist()
                return chroma

            # Otherwise try a lightweight TF-IDF based fallback (less semantic)
            if TfidfVectorizer is not None:
                logging.info("Falling back to local TF-IDF embeddings (lightweight).")
                local_emb = LocalTFIDFEmbeddings()
                chroma = Chroma.from_documents(
                    documents=docs,
                    embedding=local_emb,
                    persist_directory=CHROMA_DIR,
                    collection_name=collection_name,
                )
                chroma.persist()
                return chroma

            # No local fallback available; raise a clear error
            raise RuntimeError(
                "Remote embedding provider quota exceeded and no local fallback available. "
                "Install `sentence-transformers` or `scikit-learn` to enable local fallbacks, or increase your Google Generative AI quota/billing. "
                "See: https://ai.google.dev/gemini-api/docs/rate-limits"
            ) from e

        # If not a quota error, re-raise the original exception
        raise


def load_chroma_retriever(chroma: Chroma, k: int = 5):
    """
    Return a retriever object from an existing Chroma vectorstore.
    """
    return chroma.as_retriever(search_kwargs={"k": k})
