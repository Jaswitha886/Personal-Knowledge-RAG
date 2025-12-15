# query.py
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import ollama

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to SAME persistent DB
client = PersistentClient(path="chroma_db")

# Always safe
collection = client.get_or_create_collection(
    name="personal_knowledge_base"
)

# Debug check (you can remove later)
print(client.list_collections())

# Ask question
question = input("Ask a question: ")

# Embed query
query_embedding = embedding_model.encode(question).tolist()

# Retrieve context
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

context = "\n\n".join(results["documents"][0])

prompt = f"""
Answer using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""

response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": prompt}]
)

print("\n--- Answer ---\n")
print(response["message"]["content"])
