"""
Step 1 & 2: Load documents and split them into overlapping chunks.
"""

import os
from pypdf import PdfReader


def load_document_text(filepath: str) -> str:
    """Load raw text from a .txt or .pdf file."""
    if filepath.lower().endswith(".pdf"):
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def load_documents_from_folder(folder: str) -> list[dict]:
    """Returns a list of {"source": filename, "text": full_text} for every
    .txt / .pdf file in the folder."""
    docs = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".txt", ".pdf")):
            path = os.path.join(folder, fname)
            text = load_document_text(path)
            docs.append({"source": fname, "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks by character count.

    chunk_size=700, overlap=100 is a reasonable starting point for
    English prose. Tune these based on your documents:
      - Shorter, fact-dense docs (policies, specs) -> smaller chunks (400-600)
      - Long narrative docs (reports, articles)      -> larger chunks (800-1200)
    """
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap  # step forward, re-including the overlap window
    return chunks


def build_chunk_records(docs: list[dict], chunk_size: int = 700, overlap: int = 100) -> list[dict]:
    """Turn loaded documents into a flat list of chunk records, each with
    a unique id, the chunk text, and metadata about which document it came from."""
    records = []
    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            records.append({
                "id": f"{doc['source']}::chunk_{i}",
                "text": chunk,
                "source": doc["source"],
                "chunk_index": i,
            })
    return records
