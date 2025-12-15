# app.py
import streamlit as st
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import ollama

# Page config
st.set_page_config(page_title="Personal Knowledge Assistant", layout="centered")

st.title("🧠 Personal Knowledge Assistant")
st.write("Ask questions based on your personal notes using RAG.")

# Load embedding model (once)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# Connect to persistent ChromaDB
client = PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(
    name="personal_knowledge_base"
)

# User input
question = st.text_input("Ask a question:")

if st.button("Get Answer") and question.strip():
    # Embed question
    query_embedding = embedding_model.encode(question).tolist()

    # Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_docs = results["documents"][0]

    if not retrieved_docs:
        st.warning("No relevant information found.")
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
