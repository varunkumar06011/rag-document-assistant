"""
config.py — Central configuration loaded from .env
All other modules import from here. Never hardcode values elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
VECTORSTORE_PATH = BASE_DIR / os.getenv("VECTORSTORE_PATH", "data/vectorstore")
UPLOADS_PATH     = BASE_DIR / os.getenv("UPLOADS_PATH",    "data/uploads")

# Create directories if they don't exist
VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
UPLOADS_PATH.mkdir(parents=True, exist_ok=True)

# ── LLM ───────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama3-8b-8192")

# ── Embeddings ────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Chunking ──────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP",  50))

# ── Retrieval ─────────────────────────────────────────────────────
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 6))
TOP_K_RERANK    = int(os.getenv("TOP_K_RERANK",    3))

# ── Chroma collection name ────────────────────────────────────────
COLLECTION_NAME = "rag_documents"


def validate_config():
    """Call this at startup to catch missing keys early."""
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        raise ValueError(
            "\n\n❌  GROQ_API_KEY is not set!\n"
            "   1. Go to https://console.groq.com and sign up (free)\n"
            "   2. Create an API key\n"
            "   3. Copy .env.example → .env  and paste your key\n"
        )
    print(f"✅  Config loaded | LLM: {LLM_MODEL} | Embed: {EMBEDDING_MODEL}")
