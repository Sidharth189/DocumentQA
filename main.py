# main.py
import os
from typing import List
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from services.workflow import run_workflow

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.post("/api/query")
async def api_query(query: str = Form(...), files: List[UploadFile] = File(None)):
    result = await run_workflow(query, files)
    return result
