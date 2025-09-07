from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
import shutil, os
from typing import List
from dotenv import load_dotenv

from config import TO_MANUALS_DIR, LLM_MODEL_NAME
from rag_index import build_or_load_index, reload_index_for_files
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"]

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1) Serve React/Vue/etc index.html at root
@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse("static/index.html")

# 2) Serve all other assets (JS/CSS/images) under /static/*
app.mount("/static", StaticFiles(directory="static"), name="static")

# ensure the directory exists
TO_MANUALS_DIR.mkdir(exist_ok=True)

# we'll populate these on startup
index = None
query_engine = None

@app.on_event("startup")
async def startup_event():
    global index, query_engine
    index = await run_in_threadpool(build_or_load_index)
    llm = HuggingFaceInferenceAPI(
        model_name=LLM_MODEL_NAME,
        token=HF_TOKEN,
        max_new_tokens=256,
        temperature=0.2
    )
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

@app.post("/upload")
async def upload_manuals(uploaded_files: List[UploadFile] = File(...)):
    saved_paths = []
    for f in uploaded_files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"Invalid file type: {f.filename}")
        dest = TO_MANUALS_DIR / f.filename
        with open(dest, "wb") as buf:
            shutil.copyfileobj(f.file, buf)
        saved_paths.append(dest)

    # offload the blocking, embedding-heavy reindex to a thread
    await run_in_threadpool(reload_index_for_files, saved_paths)

    return {
        "message": "Upload successful",
        "files": [p.name for p in saved_paths]
    }

from fastapi.concurrency import run_in_threadpool

@app.post("/query")
async def ask_question(question: str = Form(...)):
    """
    Accept a question and return an answer with source citations.
    Offload to a thread so that any internal asyncio.run calls
    don’t collide with Uvicorn’s loop.
    """
    if query_engine is None:
        raise HTTPException(status_code=500, detail="Query engine not initialized")

    # offload the blocking query (which itself may call asyncio.run)
    response = await run_in_threadpool(query_engine.query, question)

    return {
        "answer": response.response,
        "sources": response.get_citations() if hasattr(response, 'get_citations') else []
    }

