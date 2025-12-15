# pdf_ingest.py
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Chunk function
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def ingest_pdf(pdf_path, collection_name="personal_knowledge_base"):
    # Read PDF
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        if page.extract_text():
            full_text += page.extract_text()

    # Chunk text
    chunks = chunk_text(full_text)
    print(f"PDF Chunks created: {len(chunks)}")

    # Connect to persistent DB
    client = PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name=collection_name)

    # Store chunks
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embedding_model.encode(chunk).tolist()],
            ids=[f"pdf_{i}"]
        )

    print(" PDF ingestion completed.")
