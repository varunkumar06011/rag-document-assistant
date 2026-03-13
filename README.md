---
title: Rag Document Assistant
emoji: 🚀
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: RAG assistant for querying PDF and DOCX documents
---

# 📄 RAG Document Assistant

A **Retrieval-Augmented Generation (RAG)** system that allows users to upload **PDF documents** and ask natural language questions.
The assistant retrieves relevant document chunks and generates **grounded answers with source citations** using **LLaMA 3 via Groq**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# ⭐ Key Features

* Upload **PDF documents**
* Semantic document search using **vector embeddings**
* **Retrieval-Augmented Generation (RAG)**
* Answer generation using **LLaMA 3 via Groq API**
* **Source citations for every answer**
* **FAISS vector database**
* **Streamlit chat interface**
* Deployable on **Streamlit Cloud**

---

# 🎯 What This Does

Upload any document and ask questions about it.

The system retrieves relevant context and generates answers grounded in the document.

### Example

User: What does the document say about the finance industry?

Assistant:
The finance industry offers opportunities across corporate, investment, and personal finance sectors.
**Source: Finance.pdf – Page 3**

---

# 🏗️ Architecture

```
PDF Upload
↓
Text Splitter (chunk_size=500)
↓
HuggingFace Embeddings (MiniLM)
↓
FAISS Vector Store
↓
User Question
↓
Vector Similarity Search
↓
LLaMA 3 (Groq API)
↓
Answer + Source Citations
```

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

1. Visit
   https://console.groq.com

2. Create a free API key

3. Add the key to Streamlit secrets:

```
GROQ_API_KEY=your_key_here
```

---

## 3 Run the Application

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

# 🌐 Deployment

This project can be deployed using:

* Streamlit Cloud
* HuggingFace Spaces
* Docker
* Render
* Railway

For **Streamlit Cloud**:

1. Connect the GitHub repository
2. Add `GROQ_API_KEY` in **Secrets**
3. Deploy the app

---

# 📁 Project Structure

```
rag-document-assistant

app.py
requirements.txt
README.md
```

---

# 🛠 Tech Stack

Language: Python 3.11
Framework: LangChain
LLM: LLaMA 3 via Groq API
Embeddings: HuggingFace MiniLM
Vector Database: FAISS
Frontend: Streamlit

---

# 📜 License

MIT License

---

# 👨‍💻 Author

Varun Kumar

GitHub
https://github.com/varunkumar06011
