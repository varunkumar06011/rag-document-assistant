"""
hf_app.py — HuggingFace Spaces deployment entry point

HuggingFace Spaces runs Streamlit apps directly.
This file wraps app.py so it works on Spaces where
the API is embedded (not a separate server).

On Spaces: set GROQ_API_KEY in the Space "Secrets" settings.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from src.config import GROQ_API_KEY

# ── On HuggingFace Spaces, call RAG functions directly ────────────
# (No separate FastAPI server — everything in one process)

from src.ingestion import ingest_document, list_ingested_files, delete_file_from_store
from src.retrieval import query_documents
import tempfile
import shutil

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 RAG Assistant")
    st.markdown("---")

    # Check API key
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        st.error("⚠️ GROQ_API_KEY not set!\nAdd it in Space Settings → Secrets")
    else:
        st.success("🟢 Groq API connected")

    st.markdown("### 📁 Upload Documents")
    uploaded_file = st.file_uploader("PDF or DOCX", type=["pdf", "docx"])

    if uploaded_file and st.button("⬆️ Ingest", use_container_width=True, type="primary"):
        with st.spinner("Processing..."):
            # Save to temp file
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(uploaded_file, tmp)
                tmp_path = tmp.name

            # Run original filename through ingestion
            try:
                # Rename temp file to original name for metadata
                final_path = Path(tmp_path).parent / uploaded_file.name
                shutil.move(tmp_path, str(final_path))
                result = ingest_document(str(final_path))
                st.success(f"✅ {result['chunks_stored']} chunks stored!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")
    docs = list_ingested_files()
    if docs:
        for doc in docs:
            st.markdown(f"📄 `{doc}`")
    else:
        st.info("No documents uploaded yet")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Main chat ─────────────────────────────────────────────────────
st.title("💬 Chat with Your Documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])} chunks)"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['filename']}** · Page {src['page']}\n> {src['snippet']}")

if question := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = query_documents(question)
                st.markdown(result["answer"])
                if result.get("sources"):
                    with st.expander(f"📎 Sources ({len(result['sources'])} chunks)", expanded=True):
                        for src in result["sources"]:
                            st.markdown(f"**{src['filename']}** · Page {src['page']}\n> {src['snippet']}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                })
            except Exception as e:
                st.error(f"Error: {e}")
