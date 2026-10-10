"""
Embedding backends for the RAG pipeline.

This file is intentionally written so you can SWAP the embedding method
without touching any other file (ingest.py, retrieve.py). Just change
which class get_embedder() returns.

- TfidfEmbedder: pure Python, no downloads, no API key. Good for getting
  the pipeline running immediately and understanding the mechanics.
  NOT as semantically smart as a real embedding model.

- SentenceTransformerEmbedder: a real local embedding model (MiniLM).
  Much better quality. Requires `pip install sentence-transformers`
  and an internet connection to download the model the first time.

- OpenAIEmbedder: uses OpenAI's embedding API. Best quality, needs an
  API key and per-call cost.

For your internship project: start with TfidfEmbedder to prove the
pipeline works end-to-end, then switch to SentenceTransformerEmbedder
or OpenAIEmbedder for the "real" version you demo to your mentor.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    """Lightweight, dependency-free embedder based on TF-IDF + SVD.

    Not a true semantic embedding model, but implements the exact same
    interface, so swapping it out later for a real model requires zero
    changes anywhere else in the code.
    """

    def __init__(self, max_features: int = 2048):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._fitted = False

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            # Fit on the fly if someone calls embed() before fit() (e.g. single query)
            raise RuntimeError(
                "TfidfEmbedder must be fit on the document corpus first. "
                "Call embedder.fit(all_chunk_texts) once during ingestion, "
                "then reuse the same embedder instance for queries."
            )
        vectors = self.vectorizer.transform(texts)
        return vectors.toarray().tolist()


class SentenceTransformerEmbedder:
    """Real local embedding model. Requires: pip install sentence-transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]):
        pass  # no fitting needed, model is pre-trained

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()


class OpenAIEmbedder:
    """OpenAI embedding API. Requires: pip install openai, and OPENAI_API_KEY env var."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name

    def fit(self, texts: list[str]):
        pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [item.embedding for item in response.data]


def get_embedder(backend: str = "tfidf"):
    """Factory function — change `backend` to switch embedding methods."""
    if backend == "tfidf":
        return TfidfEmbedder()
    elif backend == "sentence_transformer":
        return SentenceTransformerEmbedder()
    elif backend == "openai":
        return OpenAIEmbedder()
    else:
        raise ValueError(f"Unknown backend: {backend}")
