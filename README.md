# 🧠 Personal Knowledge Assistant (RAG)

This project is a **Personal Knowledge Assistant** built using **Retrieval-Augmented Generation (RAG)**.
It allows users to ask questions and receive accurate, grounded answers based on their own notes instead of relying only on a language model’s memory.

---

## 🚀 Features
- Ingest personal notes and documents
- Store embeddings in a persistent vector database (ChromaDB)
- Retrieve relevant context using semantic search
- Generate grounded answers using a local LLM (Ollama)
- Simple Streamlit UI for interaction

---

## 🧩 Tech Stack
- Python
- ChromaDB (Vector Database)
- Sentence Transformers (Embeddings)
- Ollama (LLM)
- Streamlit (UI)

---

## ⚙️ How It Works
1. Notes are chunked and converted into embeddings
2. Embeddings are stored in ChromaDB
3. User queries are embedded and matched semantically
4. Relevant chunks are retrieved
5. The LLM generates answers using retrieved context

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python ingest.py
streamlit run app.py
