"""
api.py — Phase 4a: FastAPI Backend

Endpoints:
  POST /upload         ← ingest a PDF or DOCX
  POST /query          ← ask a question, get answer + sources
  GET  /documents      ← list all ingested files
  DELETE /documents/{filename}  ← remove a file from the store
  GET  /health         ← liveness check

Run locally:
  uvicorn src.api:app --reload --port 8000

Then open: http://localhost:8000/docs  (auto-generated Swagger UI)

Interview talking point:
  - FastAPI auto-generates OpenAPI docs — shows you know REST API design
  - Separation of concerns: API layer is thin, logic is in ingestion.py / retrieval.py
  - Async endpoints for scalability
"""

import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from src.config import UPLOADS_PATH, validate_config
from src.ingestion import ingest_document, list_ingested_files, delete_file_from_store
from src.retrieval import query_documents

# ── App setup ─────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Document Assistant API",
    description="Upload documents and ask questions. Powered by LLaMA 3 + ChromaDB.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

    model_config = {"json_schema_extra": {"example": {"question": "What is the main topic of the document?"}}}


class Source(BaseModel):
    index: int
    filename: str
    page: int | str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    num_chunks_retrieved: int
    num_chunks_after_rerank: int


class UploadResponse(BaseModel):
    filename: str
    chunks_stored: int
    message: str


class DocumentListResponse(BaseModel):
    documents: List[str]
    count: int


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Liveness check — use this to verify the server is running."""
    return {"status": "ok", "message": "RAG API is running"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or DOCX file to the knowledge base.
    The file is chunked, embedded, and stored in ChromaDB.
    Re-uploading the same file replaces its old chunks.
    """
    # Validate file type
    allowed = {".pdf", ".docx", ".doc"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}"
        )

    # Save uploaded file to disk
    save_path = UPLOADS_PATH / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"File saved to: {save_path}")

    # Run ingestion pipeline
    try:
        result = ingest_document(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return UploadResponse(
        filename=result["filename"],
        chunks_stored=result["chunks_stored"],
        message=f"Successfully ingested '{file.filename}' into {result['chunks_stored']} chunks.",
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question against the uploaded documents.
    Returns a grounded answer with source citations.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = query_documents(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return QueryResponse(**result)


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all documents currently in the knowledge base."""
    files = list_ingested_files()
    return DocumentListResponse(documents=files, count=len(files))


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Remove a document and all its chunks from the knowledge base."""
    success = delete_file_from_store(filename)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")

    # Also remove the raw file if it exists
    raw_path = UPLOADS_PATH / filename
    if raw_path.exists():
        raw_path.unlink()

    return {"message": f"'{filename}' removed from knowledge base."}


# ── Startup event ─────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    try:
        validate_config()
    except ValueError as e:
        logger.error(str(e))
        # Don't crash — let the /health endpoint still work so user sees the error


# ── Run directly (python src/api.py) ─────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
