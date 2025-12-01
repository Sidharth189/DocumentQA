# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional

from .langchain_integration import ingest_document, answer_query
from .nodes.vector_upsert import create_schema

app = FastAPI(title="DocumentQA - LangChain RetrievalQA")

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/create_schema")
def create_schema_route():
    """
    Optional: create/reset Weaviate schema. Call once when starting up if needed.
    """
    try:
        create_schema()
        return {"message": "schema created/reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload file and run full ingestion pipeline (save -> extract -> sanitize -> chunk -> embed -> upsert)
    """
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="empty file")
        result = ingest_document(file_bytes, file.filename)
        return {"message": "ingest complete", "result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AskRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    model: Optional[str] = None

@app.post("/ask")
def ask(req: AskRequest):
    try:
        res = answer_query(req.query, top_k=req.top_k, model=req.model)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
