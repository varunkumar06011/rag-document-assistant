"""
app.py — Phase 4b: Streamlit Chat UI

Run locally:
  streamlit run app.py

Features:
  - Drag-and-drop document upload
  - Chat interface with message history
  - Source citations expandable panel
  - System stats sidebar
  - Chunk count and model info display
"""

import streamlit as st

import json
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API base URL (FastAPI backend) ────────────────────────────────
API_URL = "http://localhost:8000"


# ── Helper: call the API ──────────────────────────────────────────
def api_upload(file_bytes: bytes, filename: str) -> dict:
    response = requests.post(
        f"{API_URL}/upload",
        files={"file": (filename, file_bytes, "application/octet-stream")},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def api_query(question: str) -> dict:
    response = requests.post(
        f"{API_URL}/query",
        json={"question": question},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def api_list_documents() -> list:
    response = requests.get(f"{API_URL}/documents", timeout=10)
    response.raise_for_status()
    return response.json().get("documents", [])


def api_delete_document(filename: str) -> bool:
    response = requests.delete(f"{API_URL}/documents/{filename}", timeout=10)
    return response.status_code == 200


def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Session state init ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 RAG Assistant")
    st.markdown("---")

    # API health indicator
    if check_api_health():
        st.success("🟢 API connected")
    else:
        st.error("🔴 API offline — run: `uvicorn src.api:app --reload`")

    st.markdown("### 📁 Upload Documents")
    st.markdown("Supports PDF and DOCX files")

    uploaded_file = st.file_uploader(
        "Drop your document here",
        type=["pdf", "docx"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        if st.button("⬆️ Ingest Document", use_container_width=True, type="primary"):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    result = api_upload(uploaded_file.read(), uploaded_file.name)
                    st.success(f"✅ {result['chunks_stored']} chunks stored!")
                    st.json(result)
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    st.markdown("---")

    # List and manage documents
    st.markdown("### 📚 Knowledge Base")
    try:
        documents = api_list_documents()
        if documents:
            for doc in documents:
                col1, col2 = st.columns([3, 1])
                col1.markdown(f"📄 `{doc}`")
                if col2.button("🗑️", key=f"del_{doc}", help=f"Delete {doc}"):
                    if api_delete_document(doc):
                        st.success(f"Deleted {doc}")
                        st.rerun()
        else:
            st.info("No documents uploaded yet")
    except Exception:
        st.warning("Could not fetch document list")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.markdown("""
    | Setting | Value |
    |---------|-------|
    | LLM | LLaMA 3 8B (Groq) |
    | Embeddings | MiniLM-L6-v2 |
    | Chunk size | 500 chars |
    | Re-rank top-k | 3 |
    """)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────
st.title("💬 Chat with Your Documents")
st.markdown("Upload a document in the sidebar, then ask questions below.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])} chunks used)", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"""
**Source {src['index']}** — `{src['filename']}` · Page {src['page']}
> {src['snippet']}
""")


# Chat input
if question := st.chat_input("Ask a question about your documents..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = api_query(question)
                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)

                # Show retrieval stats
                col1, col2 = st.columns(2)
                col1.metric("Chunks retrieved", result.get("num_chunks_retrieved", 0))
                col2.metric("After re-ranking", result.get("num_chunks_after_rerank", 0))

                # Show sources
                if sources:
                    with st.expander(f"📎 Sources ({len(sources)} chunks used)", expanded=True):
                        for src in sources:
                            st.markdown(f"""
**Source {src['index']}** — `{src['filename']}` · Page {src['page']}
> {src['snippet']}
""")

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except requests.exceptions.ConnectionError:
                err = "❌ Cannot connect to API. Make sure the FastAPI server is running:\n```\nuvicorn src.api:app --reload\n```"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

            except Exception as e:
                err = f"❌ Error: {str(e)}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
