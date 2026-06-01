"""
RAG Pipeline — Retriever Module

Performs semantic search over the ChromaDB knowledge base
to retrieve relevant chunks for grounding Turing's responses.
"""

from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer


class Retriever:
    """Retrieves relevant document chunks from ChromaDB using semantic search."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_dir: str = "storage/chroma_db",
        collection_name: str = "turing_knowledge",
        top_k: int = 5,
        boost_primary_source: float = 1.3,
    ):
        """
        Args:
            model_name: Sentence-Transformer model (must match embedder).
            persist_dir: ChromaDB persistence directory.
            collection_name: Collection to search in.
            top_k: Number of top results to return.
            boost_primary_source: Score multiplier for Turing's own words.
        """
        self.model_name = model_name
        self.top_k = top_k
        self.boost_primary_source = boost_primary_source

        # Load the same embedding model used for indexing
        self.model = SentenceTransformer(model_name)

        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=persist_dir)
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            # Collection doesn't exist yet — create it
            self.collection = self.client.get_or_create_collection(name=collection_name)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        year_range: Optional[tuple[int, int]] = None,
        source_types: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a given query.

        Args:
            query: The user's question or input.
            top_k: Override default top_k.
            year_range: Optional (start_year, end_year) to filter by.
            source_types: Optional list of source types to filter by.

        Returns:
            List of result dicts with 'text', 'metadata', 'score', 'chunk_id'.
        """
        k = top_k or self.top_k

        # Check if collection has any documents
        if self.collection.count() == 0:
            return []

        # Embed the query
        query_embedding = self.model.encode(query, convert_to_numpy=True).tolist()

        # Build metadata filter
        where_filter = self._build_where_filter(year_range, source_types)

        # Query ChromaDB
        try:
            if where_filter:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(k * 2, self.collection.count()),  # Fetch extra for re-ranking
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                )
            else:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(k * 2, self.collection.count()),
                    include=["documents", "metadatas", "distances"],
                )
        except Exception as e:
            print(f"[Retriever] Query error: {e}")
            return []

        # Process and re-rank results
        processed = self._process_results(results)

        # Apply primary source boosting
        processed = self._apply_source_boosting(processed)

        # Sort by boosted score and return top_k
        processed.sort(key=lambda x: x["score"], reverse=True)
        return processed[:k]

    def _build_where_filter(
        self,
        year_range: Optional[tuple[int, int]] = None,
        source_types: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """Build a ChromaDB where filter from the given constraints."""
        conditions = []

        if year_range:
            start_year, end_year = year_range
            conditions.append({"year": {"$gte": start_year}})
            conditions.append({"year": {"$lte": end_year}})

        if source_types:
            if len(source_types) == 1:
                conditions.append({"source_type": {"$eq": source_types[0]}})
            else:
                conditions.append({"source_type": {"$in": source_types}})

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def _process_results(self, results: dict) -> list[dict]:
        """Process raw ChromaDB results into a clean format."""
        processed = []

        if not results or not results.get("ids") or not results["ids"][0]:
            return processed

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):
            # ChromaDB returns L2 distances; convert to similarity score
            # Lower distance = more similar; we invert for a 0-1 score
            distance = distances[i]
            similarity = 1.0 / (1.0 + distance)

            processed.append({
                "chunk_id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i],
                "score": similarity,
                "distance": distance,
            })

        return processed

    def _apply_source_boosting(self, results: list[dict]) -> list[dict]:
        """Boost scores for chunks that are Turing's own words."""
        for result in results:
            is_turing_voice = result["metadata"].get("is_turing_voice", False)
            if is_turing_voice:
                result["score"] *= self.boost_primary_source
        return results

    def format_context(self, results: list[dict]) -> str:
        """
        Format retrieved results into a context string for the LLM prompt.

        Returns a formatted string with source attributions.
        """
        if not results:
            return "No relevant sources found in my works."

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source_title", "Unknown")
            year = result["metadata"].get("year", "")
            source_type = result["metadata"].get("source_type", "")
            section = result["metadata"].get("section", "")

            header = f"[Source {i}: {source}"
            if year:
                header += f" ({year})"
            if section and section not in ("Full Document", "Introduction"):
                header += f" — {section}"
            header += "]"

            context_parts.append(f"{header}\n{result['text']}")

        return "\n\n---\n\n".join(context_parts)

    def get_stats(self) -> dict:
        """Get retriever statistics."""
        return {
            "collection_count": self.collection.count(),
            "model": self.model_name,
            "top_k": self.top_k,
            "boost_primary": self.boost_primary_source,
        }
