# 🧠 Personal Knowledge Assistant (RAG)

This project is a **Personal Knowledge Assistant** built using **Retrieval-Augmented Generation (RAG)**.
It allows users to ask questions and receive accurate, grounded answers based on their own notes or uploaded PDFs instead of relying only on a language model’s memory.

---

## 🚀 Features
- Ingest personal notes and documents
- Upload PDFs and query them dynamically
- Store embeddings in a persistent vector database (ChromaDB)
- Retrieve relevant context using semantic (vector) search
- Generate grounded answers using a local LLM (Ollama)
- Simple Streamlit UI for interaction

---

## 🧩 Tech Stack
- Python
- ChromaDB (Vector Database)
- Sentence Transformers (Embeddings)
- Ollama (LLM)
- Streamlit (UI)
- PyPDF (PDF text extraction)

---

## ⚙️ How It Works
1. Notes or PDFs are chunked into smaller pieces
2. Each chunk is converted into an embedding
3. Embeddings are stored in ChromaDB
4. User queries are embedded and matched semantically
5. Relevant chunks are retrieved
6. The LLM generates answers using retrieved context

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python ingest.py        # (optional: for notes.txt)
streamlit run app.py
