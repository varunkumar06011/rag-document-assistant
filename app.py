"""
Streamlit RAG Document Assistant
Works on Streamlit Cloud without FastAPI
"""

import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


# ─────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide"
)


# ─────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.title("📄 RAG Assistant")
    st.markdown("---")

    st.success("🟢 System Ready")

    st.markdown("### Upload PDF Document")

    uploaded_file = st.file_uploader(
        "Drop your PDF here",
        type=["pdf"]
    )

    if uploaded_file:

        with st.spinner("Processing document..."):

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name

            loader = PyPDFLoader(file_path)
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

            st.session_state.vectorstore = vectorstore

            st.success(f"{len(chunks)} chunks created")

    st.markdown("---")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────
st.title("💬 Chat with Your Documents")

st.write("Upload a document in the sidebar and ask questions.")


# Display previous chat
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Sources"):
                for src in message["sources"]:
                    st.markdown(src)


# Chat input
question = st.chat_input("Ask a question about your document")


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        if st.session_state.vectorstore is None:

            st.error("Please upload a document first.")

        else:

            with st.spinner("Generating answer..."):

                llm = ChatGroq(
                    groq_api_key=st.secrets["GROQ_API_KEY"],
                    model_name="llama3-8b-8192"
                )

                retriever = st.session_state.vectorstore.as_retriever()

                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True
                )

                result = qa.invoke({"query": question})

                answer = result["result"]

                sources = []
                for doc in result["source_documents"]:
                    sources.append(doc.page_content[:200] + "...")

                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(s)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })