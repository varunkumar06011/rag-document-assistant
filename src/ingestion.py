"""
ingestion.py — Phase 2: Document Ingestion Pipeline

Flow:
  PDF/DOCX file
      ↓
  load_document()      ← reads raw text + metadata
      ↓
  split_into_chunks()  ← smart chunking with overlap
      ↓
  embed_and_store()    ← HuggingFace embeddings → ChromaDB

Interview talking points:
  - Chunk size 500 / overlap 50 chosen to balance context vs noise
  - Metadata stored alongside vectors (source, page, chunk_id)
  - Idempotent: re-uploading same file replaces old chunks
"""

import hashlib
from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from loguru import logger

from src.config import (
    VECTORSTORE_PATH, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME
)


# ── Singleton embedding model (loaded once, reused everywhere) ────
_embeddings = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns the embedding model.
    Lazy-loaded so the app starts fast; downloaded once then cached locally.
    """
    global _embeddings
    if _embeddings is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},      # change to "cuda" if you have GPU
            encode_kwargs={"normalize_embeddings": True},  # cosine similarity works better
        )
        logger.info("Embedding model loaded ✅")
    return _embeddings


# ── Step 1: Load document ─────────────────────────────────────────
def load_document(file_path: str) -> List[Document]:
    """
    Loads a PDF or DOCX file and returns a list of LangChain Document objects.
    Each Document has:
        .page_content  = raw text
        .metadata      = {"source": filename, "page": N}
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        logger.info(f"Loaded PDF: {path.name} — {len(docs)} pages")

    elif suffix in (".docx", ".doc"):
        loader = Docx2txtLoader(str(path))
        docs = loader.load()
        logger.info(f"Loaded DOCX: {path.name} — {len(docs)} sections")

    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use PDF or DOCX.")

    # Add filename to metadata for source citations later
    for doc in docs:
        doc.metadata["filename"] = path.name

    return docs


# ── Step 2: Split into chunks ─────────────────────────────────────
def split_into_chunks(documents: List[Document]) -> List[Document]:
    """
    Splits documents into smaller chunks for embedding.

    Why RecursiveCharacterTextSplitter?
    - Tries to split on paragraphs first, then sentences, then words
    - Respects natural language boundaries (better than fixed-size splits)
    - Overlap ensures context isn't lost at chunk boundaries

    Chunk size 500 chars ≈ 100-120 words — enough context for one idea.
    Overlap 50 chars ≈ 1-2 sentences — bridges adjacent chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # priority order
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_total"] = len(chunks)

    logger.info(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ── Step 3: Embed and store ───────────────────────────────────────
def get_vectorstore() -> Chroma:
    """Returns the ChromaDB vectorstore (creates it if it doesn't exist)."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTORSTORE_PATH),
    )


def embed_and_store(chunks: List[Document], filename: str) -> Dict[str, Any]:
    """
    Converts chunks to vectors and stores them in ChromaDB.

    Steps:
    1. Delete any existing chunks from this file (idempotent re-upload)
    2. Generate embeddings using HuggingFace model (runs locally, free)
    3. Store vectors + metadata in ChromaDB (persisted to disk)

    Returns a summary dict for logging and UI display.
    """
    vectorstore = get_vectorstore()

    # Delete old chunks from same file before re-ingesting
    try:
        existing = vectorstore.get(where={"filename": filename})
        if existing["ids"]:
            vectorstore.delete(ids=existing["ids"])
            logger.info(f"Removed {len(existing['ids'])} old chunks for: {filename}")
    except Exception:
        pass  # Collection may be empty on first run

    # Generate unique IDs for each chunk (hash of content for deduplication)
    ids = [
        hashlib.md5(f"{filename}_{chunk.metadata['chunk_id']}".encode()).hexdigest()
        for chunk in chunks
    ]

    # Store in ChromaDB — this is where the actual embedding happens
    vectorstore.add_documents(documents=chunks, ids=ids)

    logger.info(f"Stored {len(chunks)} chunks from '{filename}' in ChromaDB ✅")

    return {
        "filename": filename,
        "chunks_stored": len(chunks),
        "vectorstore_path": str(VECTORSTORE_PATH),
        "embedding_model": EMBEDDING_MODEL,
    }


# ── Main pipeline function ────────────────────────────────────────
def ingest_document(file_path: str) -> Dict[str, Any]:
    """
    Full ingestion pipeline in one call.
    Used by both the API and the Streamlit UI.

    Returns a result dict summarising what was done.
    """
    logger.info(f"Starting ingestion: {file_path}")

    # Step 1: Load
    documents = load_document(file_path)

    # Step 2: Chunk
    chunks = split_into_chunks(documents)

    # Step 3: Embed + store
    filename = Path(file_path).name
    result = embed_and_store(chunks, filename)

    return result


def list_ingested_files() -> List[str]:
    """Returns list of filenames currently stored in the vectorstore."""
    try:
        vectorstore = get_vectorstore()
        results = vectorstore.get()
        filenames = list({
            meta["filename"]
            for meta in results["metadatas"]
            if "filename" in meta
        })
        return sorted(filenames)
    except Exception:
        return []


def delete_file_from_store(filename: str) -> bool:
    """Removes all chunks for a specific file from the vectorstore."""
    try:
        vectorstore = get_vectorstore()
        existing = vectorstore.get(where={"filename": filename})
        if existing["ids"]:
            vectorstore.delete(ids=existing["ids"])
            logger.info(f"Deleted {len(existing['ids'])} chunks for: {filename}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting {filename}: {e}")
        return False
