# 📄 RAG Document Assistant

A **Retrieval-Augmented Generation (RAG)** system that allows users to upload **PDF or DOCX documents** and ask natural language questions. The assistant retrieves relevant document chunks and generates **grounded answers with source citations** using **LLaMA 3 via Groq**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![FastAPI](https://img.shields.io/badge/FastAPI-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# ⭐ Key Features

* Upload **PDF or DOCX documents**
* Semantic document search using **vector embeddings**
* **Two-stage retrieval pipeline**
* Answer generation using **LLaMA 3 via Groq API**
* **Source citations for every answer**
* **ChromaDB vector database**
* **FastAPI backend**
* **Streamlit user interface**
* **Docker deployment support**

---

# 🎯 What This Does

Upload any document and ask questions about it.
The system retrieves relevant context and generates answers grounded in the document.

### Example

User: What does the document say about the finance industry?

Assistant:
The finance industry offers opportunities across corporate, investment, and personal finance sectors.
[Source: Finance.pdf, Page 3]

---

# 🏗️ Architecture

PDF/DOCX Upload
↓
Text Splitter (chunk_size=500, overlap=50)
↓
HuggingFace Embeddings (all-MiniLM-L6-v2)
↓
ChromaDB Vector Store
↓
User Question
↓
Vector Search (Top-K Retrieval)
↓
Cross-Encoder Re-ranking
↓
LLaMA 3 (Groq API)
↓
Answer + Source Citations

### Two-Stage Retrieval

1. **Vector similarity search** retrieves candidate chunks quickly.
2. **Cross-encoder re-ranking** evaluates `(query, chunk)` pairs and selects the most relevant context.

---

# 🚀 Quick Start

## 1 Clone the repository

```
git clone https://github.com/varunkumar06011/rag-document-assistant
cd rag-document-assistant
pip install -r requirements.txt
```

---

## 2 Get a Groq API Key

1. Visit https://console.groq.com
2. Create a free API key
3. Copy `.env.example` to `.env`

```
cp .env.example .env
```

Edit `.env`

```
GROQ_API_KEY=your_key_here
```

---

## 3 Run the Application

### Terminal 1 — Start FastAPI

```
uvicorn src.api:app --reload --port 8000
```

### Terminal 2 — Start Streamlit UI

```
streamlit run app.py
```

Open in browser

```
http://localhost:8501
```

---

# 🐳 Run with Docker

Build the image

```
docker build -t rag-assistant .
```

Run container

```
docker run -p 8000:8000 -p 8501:8501 -e GROQ_API_KEY=your_key_here rag-assistant
```

---

# 🌐 Deployment

This project can be deployed using:

* HuggingFace Spaces
* Docker
* Render
* Railway

For HuggingFace Spaces:

1. Create a **Streamlit Space**
2. Add `GROQ_API_KEY` in **Secrets**
3. Push this repository

---

# 📡 API Endpoints

| Method | Endpoint              | Description                   |
| ------ | --------------------- | ----------------------------- |
| POST   | /upload               | Upload and ingest document    |
| POST   | /query                | Ask questions about documents |
| GET    | /documents            | List uploaded files           |
| DELETE | /documents/{filename} | Remove a document             |
| GET    | /health               | Health check                  |

Swagger documentation:

```
http://localhost:8000/docs
```

Example API call

```
curl -X POST http://localhost:8000/query \
-H "Content-Type: application/json" \
-d '{"question":"What is the main topic?"}'
```

---

# 📁 Project Structure

```
rag-document-assistant

src/
  config.py
  ingestion.py
  retrieval.py
  api.py

app.py
hf_app.py

evaluation/
  evaluate.py

notebooks/
  rag_learning.ipynb

data/
  uploads/
  vectorstore/

Dockerfile
requirements.txt
.env.example
README.md
```

---

# 🛠 Tech Stack

Language: Python 3.11
RAG Framework: LangChain
LLM: LLaMA 3 via Groq API
Embeddings: HuggingFace MiniLM
Vector Database: ChromaDB
Re-Ranking: Cross-Encoder
Backend: FastAPI
Frontend: Streamlit
Evaluation: RAGAS
Deployment: Docker / HuggingFace Spaces

---

# 📊 Evaluation

Evaluation can be performed using **RAGAS metrics**:

* Faithfulness
* Answer relevancy
* Context precision
* Context recall

Run evaluation:

```
python evaluation/evaluate.py
```

---

# 📜 License

MIT License

---

# 👨‍💻 Author

Varun Kumar

GitHub
https://github.com/varunkumar06011
