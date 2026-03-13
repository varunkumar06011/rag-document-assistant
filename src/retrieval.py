"""
retrieval.py — Phase 3: Retrieval Chain with Re-ranking
"""

from typing import List, Dict, Any, Tuple

from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from loguru import logger

from src.config import (
    GROQ_API_KEY, LLM_MODEL,
    VECTORSTORE_PATH, COLLECTION_NAME,
    TOP_K_RETRIEVAL, TOP_K_RERANK
)

from src.ingestion import get_embeddings


# ─────────────────────────────────────────────────────────────
# Singleton LLM
# ─────────────────────────────────────────────────────────────

_llm = None

def get_llm() -> ChatGroq:
    """Initialise Groq LLM once and reuse."""
    global _llm

    if _llm is None:
        try:
            _llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=LLM_MODEL,   # IMPORTANT: use 'model' not model_name
                temperature=0.1,
                max_tokens=1024,
            )

            logger.info(f"LLM ready: {LLM_MODEL} via Groq ✅")

        except Exception as e:
            logger.error(f"Failed to initialise Groq LLM: {e}")
            raise e

    return _llm


# ─────────────────────────────────────────────────────────────
# Cross Encoder (Re-ranker)
# ─────────────────────────────────────────────────────────────

_cross_encoder = None

def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder

    if _cross_encoder is None:
        logger.info("Loading cross-encoder re-ranker...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder ready ✅")

    return _cross_encoder


# ─────────────────────────────────────────────────────────────
# Prompt Template
# ─────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a precise document assistant.

Answer the user's question using ONLY the provided context.

If the answer is not in the context say:
"I couldn't find this in the uploaded documents."

Rules:
- Be concise
- Be factual
- Mention which document and page the answer came from
- Never invent information

Context:
{context}

Question:
{question}

Answer:
""")


# ─────────────────────────────────────────────────────────────
# Vector Search
# ─────────────────────────────────────────────────────────────

def vector_search(query: str, k: int = TOP_K_RETRIEVAL) -> List[Document]:

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTORSTORE_PATH),
    )

    results = vectorstore.similarity_search_with_score(query, k=k)

    docs = [doc for doc, _score in results]

    if not docs:
        logger.warning("Vector search returned no documents")

    logger.info(f"Vector search returned {len(docs)} chunks")

    return docs


# ─────────────────────────────────────────────────────────────
# Re-ranking
# ─────────────────────────────────────────────────────────────

def rerank(query: str, docs: List[Document], top_k: int = TOP_K_RERANK) -> List[Document]:

    if not docs:
        return []

    cross_encoder = get_cross_encoder()

    pairs = [(query, doc.page_content) for doc in docs]

    scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    top_docs = [doc for _score, doc in ranked[:top_k]]

    logger.info(f"Re-ranked to top {top_k} chunks")

    return top_docs


# ─────────────────────────────────────────────────────────────
# Build Context
# ─────────────────────────────────────────────────────────────

def build_context(docs: List[Document]) -> Tuple[str, List[Dict]]:

    context_parts = []
    sources = []

    for i, doc in enumerate(docs):

        meta = doc.metadata
        filename = meta.get("filename", "Unknown")
        page = meta.get("page", "?")

        context_parts.append(
            f"[Source {i+1}: {filename}, Page {page}]\n{doc.page_content}"
        )

        sources.append({
            "index": i + 1,
            "filename": filename,
            "page": page,
            "snippet": doc.page_content[:200]
        })

    context = "\n\n---\n\n".join(context_parts)

    return context, sources


# ─────────────────────────────────────────────────────────────
# LLM Generation
# ─────────────────────────────────────────────────────────────

def generate_answer(query: str, context: str) -> str:

    llm = get_llm()

    prompt = RAG_PROMPT.format_messages(
        context=context,
        question=query
    )

    try:

        response = llm.invoke(prompt)

        # handle different response types
        if hasattr(response, "content"):
            return response.content

        if isinstance(response, str):
            return response

        if isinstance(response, dict):
            return response.get("content", str(response))

        return str(response)

    except Exception as e:

        logger.error(f"LLM generation failed: {e}")

        return "Model failed to generate answer."


# ─────────────────────────────────────────────────────────────
# Main Query Function
# ─────────────────────────────────────────────────────────────

def query_documents(question: str) -> Dict[str, Any]:

    if not question.strip():
        return {"answer": "Please ask a question.", "sources": []}

    logger.info(f"Query: {question}")

    candidate_docs = vector_search(question)

    if not candidate_docs:

        return {
            "answer": "No documents found. Please upload a document first.",
            "sources": [],
            "num_chunks_retrieved": 0,
            "num_chunks_after_rerank": 0
        }

    top_docs = rerank(question, candidate_docs)

    context, sources = build_context(top_docs)

    answer = generate_answer(question, context)

    logger.info("Answer generated ✅")

    return {
        "answer": answer,
        "sources": sources,
        "num_chunks_retrieved": len(candidate_docs),
        "num_chunks_after_rerank": len(top_docs)
    }