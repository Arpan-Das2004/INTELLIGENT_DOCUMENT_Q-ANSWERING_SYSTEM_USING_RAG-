"""
Step 5: Given a user question, retrieve the most relevant chunks.
"""

import os
import pickle
import chromadb

from embeddings import get_embedder

DB_FOLDER = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "rag_docs"


def load_retriever(db_folder: str = DB_FOLDER):
    client = chromadb.PersistentClient(path=db_folder)
    collection = client.get_collection(COLLECTION_NAME)

    # Reuse the exact embedder fit during ingestion (needed for TF-IDF so
    # queries land in the same vector space as the stored chunks).
    tfidf_path = os.path.join(db_folder, "tfidf_embedder.pkl")
    if os.path.exists(tfidf_path):
        with open(tfidf_path, "rb") as f:
            embedder = pickle.load(f)
    else:
        embedder = get_embedder("sentence_transformer")  # or "openai"

    return collection, embedder


def retrieve(question: str, top_k: int = 3, db_folder: str = DB_FOLDER) -> list[dict]:
    collection, embedder = load_retriever(db_folder)
    query_vector = embedder.embed([question])[0]

    results = collection.query(query_embeddings=[query_vector], n_results=top_k)

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": results["distances"][0][i],  # lower = more similar
        })
    return retrieved


if __name__ == "__main__":
    # Quick manual test
    'test_question = "How many days of annual leave do employees get?"'
    test_question = "How many days of annual leave do employees get?"
    print(f"Question: {test_question}\n")
    chunks = retrieve(test_question, top_k=3)
    for i, c in enumerate(chunks):
        print(f"--- Result {i+1} (source: {c['source']}, distance: {c['distance']:.4f}) ---")
        print(c["text"][:300], "...\n")
