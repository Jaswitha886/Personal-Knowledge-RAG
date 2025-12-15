# app.py
import streamlit as st
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import ollama
from pdf_ingest import ingest_pdf
import tempfile
import os

st.set_page_config(page_title="PDF Knowledge Assistant", layout="centered")
st.title(" PDF Knowledge Assistant (RAG)")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

client = PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="personal_knowledge_base")

# ---- PDF Upload ----
st.subheader("Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if st.button("Ingest PDF"):
        with st.spinner("Ingesting PDF..."):
            ingest_pdf(tmp_path)
        st.success("PDF ingested successfully!")
        os.remove(tmp_path)

st.divider()

# ---- Question Answering ----
question = st.text_input("Ask a question from the uploaded PDF:")

if st.button("Get Answer") and question.strip():
    query_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_docs = results["documents"][0]

    if not retrieved_docs:
        st.warning("No relevant content found.")
    else:
        context = "\n\n".join(retrieved_docs)

        prompt = f"""
Answer using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""

        with st.spinner("Thinking..."):
            response = ollama.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}]
            )

        st.subheader("Answer")
        st.write(response["message"]["content"])
