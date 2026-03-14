"""
app.py — Streamlit-only RAG Document Assistant

Deployable on Streamlit Cloud (no FastAPI required)

Features:
- Drag and drop document upload
- Chat interface
- Message history
- Source citations
- FAISS vector search
- Groq LLaMA3 model
"""

import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide"
)

# ── Session State ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None


# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:

    st.title("📄 RAG Assistant")
    st.markdown("---")

    st.success("🟢 System Ready")

    st.markdown("### 📁 Upload Document")
    st.markdown("Supports PDF files")

    uploaded_file = st.file_uploader(
        "Drop your document here",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:

        with st.spinner("Processing document..."):

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                path = tmp.name

            loader = PyPDFLoader(path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.from_documents(chunks, embeddings)

            st.session_state.retriever = vectorstore.as_retriever()

            st.success(f"Stored {len(chunks)} chunks!")

    st.markdown("---")

    st.markdown("### ⚙️ System Settings")

    st.markdown("""
| Setting | Value |
|--------|------|
| LLM | LLaMA3 8B (Groq) |
| Embeddings | MiniLM-L6-v2 |
| Chunk Size | 500 |
| Retrieval | FAISS |
""")

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Chat UI ─────────────────────────────────────────────
st.title("💬 Chat with Your Documents")

st.markdown("Upload a document in the sidebar, then ask questions below.")


# Show chat history
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"> {s}")


# ── Chat Input ───────────────────────────────────────────────
if question := st.chat_input("Ask a question about your document..."):

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        if not st.session_state.retriever:

            err = "⚠️ Please upload a document first."
            st.error(err)

            st.session_state.messages.append({
                "role": "assistant",
                "content": err
            })

        else:

            with st.spinner("Searching document..."):

                llm = ChatGroq(
                    groq_api_key=st.secrets["GROQ_API_KEY"],
                    model_name="llama3-8b-8192"
                )

                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=st.session_state.retriever,
                    return_source_documents=True
                )

                result = qa({"query": question})

                answer = result["result"]

                sources = []
                for doc in result["source_documents"]:
                    sources.append(doc.page_content[:200] + "...")

                st.markdown(answer)

                if sources:
                    with st.expander(f"📎 Sources ({len(sources)})"):
                        for s in sources:
                            st.markdown(f"> {s}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
```
