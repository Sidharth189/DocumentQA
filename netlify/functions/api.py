# netlify/functions/api.py
import os
from typing import List
from fastapi import FastAPI, Form, File, UploadFile
from netlify_lambda_converter import NetlifyLambda # <-- CRITICAL IMPORT

# Import your existing service logic
from services.workflow import run_workflow 

app = FastAPI()


@app.post("/api/query")
async def api_query(query: str = Form(...), files: List[UploadFile] = File(None)):
    # Your logic remains the same
    result = await run_workflow(query, files)
    return result

# CRITICAL: Define the handler for Netlify
handler = NetlifyLambda(app)