# services/workflow.py
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from fastapi import UploadFile, HTTPException

from .file_io import save_uploads, cleanup_files
from .extractor import extract_text_from_file
from .chunker import chunk_documents
from .embedding import SentenceTransformersEmbeddings
from .vectorstore import build_chroma_collection, load_chroma_retriever
from .llm import get_llm

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables from .env
load_dotenv()

# Use a global variable or environment for the key (support common names)
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set. RAG may fail.")

async def run_workflow(query: str, files: List[UploadFile]) -> Dict[str, Any]:

    # 1) Save uploads
    saved_paths = save_uploads(files)
    if not saved_paths:
        raise HTTPException(400, "No valid .pdf/.txt/.docx files uploaded")

    # 2) Extract text
    texts = []
    for p in saved_paths:
        t = extract_text_from_file(p)
        if t.strip():
            texts.append({"source": os.path.basename(p), "text": t})

    if not texts:
        cleanup_files(saved_paths)
        raise HTTPException(400, "Could not extract any text")

    # 3) Chunk documents
    docs = chunk_documents(texts)

    # 4) Embeddings adapter (local sentence-transformers)
    # Uses a local model; no external API key required for embeddings.
    model_name = os.getenv("SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2")
    try:
        embedding_adapter = SentenceTransformersEmbeddings(model_name=model_name)
    except Exception as e:
        # If sentence-transformers is not installed, return a helpful error
        cleanup_files(saved_paths)
        raise HTTPException(500, f"Local sentence-transformers embeddings not available: {e}")

    # 5) Build / load vectorstore
    chroma = build_chroma_collection(embedding_adapter, docs)
    retriever = load_chroma_retriever(chroma, k=5)

    # 6) Choose LLM - use Gemini for consistency
    llm = get_llm()
    # 7) Retrieve relevant documents
    # Retriever API varies across LangChain versions. Prefer calling the
    # retriever if it exposes a retrieval method, otherwise fall back to
    # using the Chroma vectorstore's `similarity_search` implementation.
    try:
        if hasattr(retriever, "get_relevant_documents") and callable(getattr(retriever, "get_relevant_documents")):
            raw_chunks = retriever.get_relevant_documents(query)
        else:
            # fallback to Chroma similarity search
            raw_chunks = chroma.similarity_search(query, k=5)
    except Exception as e:
        raise HTTPException(500, f"Retrieval failed: {e}")

    # 8) Build context from retrieved documents
    context = "\n\n".join([
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in raw_chunks
    ])

    # 9) Create prompt and get answer
    prompt_text = (
        f"You are a helpful assistant that answers questions based on provided documents.\n\n"
        f"INSTRUCTIONS:\n"
        f"- Answer the question concisely and directly based ONLY on the provided context.\n"
        f"- If the answer is not in the context, respond with 'I don't have information about this in the provided documents.'\n"
        f"- Use clear, structured formatting with short paragraphs.\n"
        f"- Use bullet points or numbered lists when listing multiple items.\n"
        f"- Use bold text for key terms (e.g., **MemStore**, **HFiles**).\n"
        f"- Avoid repeating the same information multiple times.\n"
        f"- Keep the answer focused and avoid unnecessary explanations.\n\n"
        f"CONTEXT FROM DOCUMENTS:\n"
        f"{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER:"
    )
    
    message = HumanMessage(content=prompt_text)
    response = llm.invoke([message])
    
    # Ensure answer is a string (handle both AIMessage.content and other formats)
    if hasattr(response, 'content'):
        answer = str(response.content).strip()
    else:
        answer = str(response).strip()
    
    # Ensure answer is not empty
    if not answer:
        answer = "No response generated from LLM."
    
    retrieved_chunks = [
        {
            "text": doc.page_content[:2000],
            "metadata": doc.metadata
        }
        for doc in raw_chunks
    ]

    # 10) Cleanup
    cleanup_files(saved_paths)

    # 11) Return structured result
    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks
    }