from typing import List

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class SentenceTransformersEmbeddings:
    """Embeddings adapter using the `sentence-transformers` library.

    Provides `embed_documents` and `embed_query` compatible with Chroma and
    LangChain's expectations (lists of Python floats).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Install with `pip install sentence-transformers`."
            )
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]):
        embs = self.model.encode(texts, convert_to_numpy=True)
        # convert numpy arrays to lists of native Python floats
        return [e.astype(float).tolist() for e in embs]

    def embed_query(self, text: str):
        e = self.model.encode([text], convert_to_numpy=True)[0]
        return e.astype(float).tolist()