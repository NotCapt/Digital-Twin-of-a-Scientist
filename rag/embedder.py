"""
RAG Pipeline — Embedding Module

Embeds text chunks using Sentence-Transformers and stores them
in ChromaDB for semantic retrieval.
"""

import os
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer


class Embedder:
    """Embeds text chunks and manages the ChromaDB vector store."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_dir: str = "storage/chroma_db",
        collection_name: str = "turing_knowledge",
    ):
        """
        Args:
            model_name: Sentence-Transformer model to use for embeddings.
            persist_dir: Directory for ChromaDB persistence.
            collection_name: Name of the ChromaDB collection for RAG docs.
        """
        self.model_name = model_name
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name

        # Ensure storage directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the embedding model
        print(f"[Embedder] Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"[Embedder] Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Alan Turing's knowledge base for RAG retrieval"},
        )

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()

    def add_chunks(self, chunks: list[dict], batch_size: int = 100):
        """
        Embed and add chunks to ChromaDB.

        Args:
            chunks: List of chunk dicts with 'text', 'chunk_id', 'metadata'.
            batch_size: Number of chunks to process per batch.
        """
        total = len(chunks)
        print(f"[Embedder] Embedding and storing {total} chunks...")

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]

            ids = [chunk["chunk_id"] for chunk in batch]
            texts = [chunk["text"] for chunk in batch]

            # Sanitize metadata — ChromaDB only accepts str, int, float, bool
            metadatas = [self._sanitize_metadata(chunk["metadata"]) for chunk in batch]

            # Generate embeddings
            embeddings = self.embed_texts(texts)

            # Upsert into ChromaDB (handles duplicates gracefully)
            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            print(f"[Embedder] Stored batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")

        print(f"[Embedder] All {total} chunks stored in collection '{self.collection_name}'")

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """
        Sanitize metadata for ChromaDB compatibility.
        ChromaDB only accepts str, int, float, bool values (no None).
        """
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = ""  # Replace None with empty string
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized

    def get_collection_stats(self) -> dict:
        """Get statistics about the current collection."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "embedding_model": self.model_name,
            "embedding_dim": self.model.get_sentence_embedding_dimension(),
            "persist_dir": str(self.persist_dir),
        }

    def clear_collection(self):
        """Delete and recreate the collection (use with caution)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Alan Turing's knowledge base for RAG retrieval"},
        )
        print(f"[Embedder] Collection '{self.collection_name}' cleared and recreated.")
