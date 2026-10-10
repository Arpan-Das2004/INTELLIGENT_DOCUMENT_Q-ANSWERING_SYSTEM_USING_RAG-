"""
Steps 1-4: Load documents -> chunk -> embed -> store in vector DB.

Run this once (or whenever your source documents change):
    python src/ingest.py
"""

import os
import shutil
import chromadb

from chunking import load_documents_from_folder, build_chunk_records
from embeddings import get_embedder

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")
DB_FOLDER = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "rag_docs"
EMBEDDING_BACKEND = "sentence_transformer"  # change to "sentence_transformer" or "openai" for better quality
#EMBEDDING_BACKEND = "tfidf"

def ingest(data_folder: str = DATA_FOLDER, db_folder: str = DB_FOLDER, reset: bool = True):
    print(f"[1/4] Loading documents from: {data_folder}")
    docs = load_documents_from_folder(data_folder)
    if not docs:
        raise FileNotFoundError(f"No .txt or .pdf files found in {data_folder}")
    print(f"      Loaded {len(docs)} document(s): {[d['source'] for d in docs]}")

    print("[2/4] Chunking documents...")
    chunk_records = build_chunk_records(docs, chunk_size=700, overlap=100)
    print(f"      Created {len(chunk_records)} chunks")

    print(f"[3/4] Generating embeddings using backend: '{EMBEDDING_BACKEND}'...")
    embedder = get_embedder(EMBEDDING_BACKEND)
    texts = [r["text"] for r in chunk_records]
    embedder.fit(texts)  # TF-IDF needs to fit on the corpus; other backends no-op
    vectors = embedder.embed(texts)
    print(f"      Generated {len(vectors)} embedding vectors (dim={len(vectors[0])})")

    print("[4/4] Storing in Chroma vector database...")
    if reset and os.path.exists(db_folder):
        shutil.rmtree(db_folder)
    client = chromadb.PersistentClient(path=db_folder)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    collection.add(
        ids=[r["id"] for r in chunk_records],
        embeddings=vectors,
        documents=[r["text"] for r in chunk_records],
        metadatas=[{"source": r["source"], "chunk_index": r["chunk_index"]} for r in chunk_records],
    )
    print(f"      Stored {collection.count()} chunks in collection '{COLLECTION_NAME}'")
    print("\nIngestion complete. Vector DB saved to:", os.path.abspath(db_folder))

    # Save the fitted embedder so retrieve.py uses the EXACT same vector space.
    # (Only matters for TF-IDF, which must be fit once and reused — real
    # embedding models like sentence-transformers/OpenAI don't need this.)
    if EMBEDDING_BACKEND == "tfidf":
        import pickle
        with open(os.path.join(db_folder, "tfidf_embedder.pkl"), "wb") as f:
            pickle.dump(embedder, f)


if __name__ == "__main__":
    ingest()
