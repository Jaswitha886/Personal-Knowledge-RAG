# ingest.py
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Read notes
with open("data/notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Chunk text
def chunk_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = chunk_text(text)
print(f"Chunks created: {len(chunks)}")

# Create persistent ChromaDB client
client = PersistentClient(path="chroma_db")

# Create or get collection
collection = client.get_or_create_collection(
    name="personal_knowledge_base"
)

# Add documents
for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        embeddings=[embedding_model.encode(chunk).tolist()],
        ids=[str(i)]
    )

print(" Ingestion completed successfully.")
print(" Data is persisted automatically.")
