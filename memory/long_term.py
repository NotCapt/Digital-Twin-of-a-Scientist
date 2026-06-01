"""
Memory System — Long-Term Memory Module

Manages persistent memory across sessions using ChromaDB.
Three memory types:
  - Episodic: Summaries of past conversations
  - Semantic: Facts learned about the user
  - Important Moments: Key exchanges flagged as significant
"""

import json
from datetime import datetime
from typing import Optional
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


class LongTermMemory:
    """
    Manages persistent cross-session memory stored in ChromaDB.

    Provides three separate collections for different memory types:
    episodic, semantic, and important moments.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_dir: str = "storage/chroma_db",
        episodic_collection: str = "episodic_memory",
        semantic_collection: str = "semantic_memory",
        important_collection: str = "important_moments",
    ):
        self.model_name = model_name
        self.persist_dir = persist_dir

        # Load embedding model (shared with RAG)
        self.model = SentenceTransformer(model_name)

        # Initialize ChromaDB client
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Create/get collections
        self.episodic = self.client.get_or_create_collection(
            name=episodic_collection,
            metadata={"description": "Summaries of past conversation sessions"},
        )
        self.semantic = self.client.get_or_create_collection(
            name=semantic_collection,
            metadata={"description": "Facts learned about the user"},
        )
        self.important = self.client.get_or_create_collection(
            name=important_collection,
            metadata={"description": "Key exchanges flagged as significant"},
        )

    # ─── Episodic Memory ──────────────────────────────────────────────

    def store_episodic(self, session_id: str, summary: str, topics: list[str],
                       turn_count: int, user_sentiment: str = ""):
        """
        Store a conversation session summary as an episodic memory.

        Args:
            session_id: Unique session identifier.
            summary: LLM-generated summary of the conversation.
            topics: List of topics discussed.
            turn_count: Number of turns in the session.
            user_sentiment: Overall user sentiment during the session.
        """
        embedding = self.model.encode(summary, convert_to_numpy=True).tolist()

        self.episodic.upsert(
            ids=[session_id],
            documents=[summary],
            embeddings=[embedding],
            metadatas=[{
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "topics": json.dumps(topics),
                "turn_count": turn_count,
                "user_sentiment": user_sentiment or "",
            }],
        )

    def retrieve_episodic(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve relevant past session summaries."""
        if self.episodic.count() == 0:
            return []

        query_embedding = self.model.encode(query, convert_to_numpy=True).tolist()

        results = self.episodic.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.episodic.count()),
            include=["documents", "metadatas", "distances"],
        )

        return self._format_results(results)

    def get_all_episodic(self) -> list[dict]:
        """Get all episodic memories (for dashboard display)."""
        if self.episodic.count() == 0:
            return []

        results = self.episodic.get(include=["documents", "metadatas"])
        memories = []
        for i in range(len(results["ids"])):
            memories.append({
                "id": results["ids"][i],
                "summary": results["documents"][i],
                "metadata": results["metadatas"][i],
            })

        # Sort by timestamp (most recent first)
        memories.sort(
            key=lambda x: x["metadata"].get("timestamp", ""),
            reverse=True,
        )
        return memories

    # ─── Semantic Memory ──────────────────────────────────────────────

    def store_semantic(self, fact_id: str, fact: str, category: str,
                       confidence: float = 0.8):
        """
        Store a fact learned about the user.

        Args:
            fact_id: Unique identifier for this fact.
            fact: The fact text (e.g., "User's name is Faaiz").
            category: Category (personal_info, interests, expertise, preferences).
            confidence: Confidence level (0-1).
        """
        embedding = self.model.encode(fact, convert_to_numpy=True).tolist()

        now = datetime.now().isoformat()

        # Check if fact already exists — update timestamp if so
        try:
            existing = self.semantic.get(ids=[fact_id])
            if existing and existing["ids"]:
                first_seen = existing["metadatas"][0].get("first_seen", now)
            else:
                first_seen = now
        except Exception:
            first_seen = now

        self.semantic.upsert(
            ids=[fact_id],
            documents=[fact],
            embeddings=[embedding],
            metadatas=[{
                "fact_id": fact_id,
                "category": category,
                "confidence": confidence,
                "first_seen": first_seen,
                "last_updated": now,
            }],
        )

    def retrieve_semantic(self, query: str = "", top_k: int = 5) -> list[dict]:
        """Retrieve relevant facts about the user."""
        if self.semantic.count() == 0:
            return []

        if query:
            query_embedding = self.model.encode(query, convert_to_numpy=True).tolist()
            results = self.semantic.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.semantic.count()),
                include=["documents", "metadatas", "distances"],
            )
            return self._format_results(results)
        else:
            # Return all semantic facts
            return self.get_all_semantic()

    def get_all_semantic(self) -> list[dict]:
        """Get all semantic memories (facts about the user)."""
        if self.semantic.count() == 0:
            return []

        results = self.semantic.get(include=["documents", "metadatas"])
        memories = []
        for i in range(len(results["ids"])):
            memories.append({
                "id": results["ids"][i],
                "fact": results["documents"][i],
                "metadata": results["metadatas"][i],
            })
        return memories

    # ─── Important Moments ────────────────────────────────────────────

    def store_important(self, moment_id: str, exchange: str,
                        importance_reason: str, session_id: str):
        """
        Store a key exchange flagged as significant.

        Args:
            moment_id: Unique identifier for this moment.
            exchange: The exchange text.
            importance_reason: Why this was flagged as important.
            session_id: Which session this occurred in.
        """
        embedding = self.model.encode(exchange, convert_to_numpy=True).tolist()

        self.important.upsert(
            ids=[moment_id],
            documents=[exchange],
            embeddings=[embedding],
            metadatas=[{
                "moment_id": moment_id,
                "importance_reason": importance_reason,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
            }],
        )

    def retrieve_important(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve relevant important moments."""
        if self.important.count() == 0:
            return []

        query_embedding = self.model.encode(query, convert_to_numpy=True).tolist()

        results = self.important.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.important.count()),
            include=["documents", "metadatas", "distances"],
        )

        return self._format_results(results)

    def get_all_important(self) -> list[dict]:
        """Get all important moments (for dashboard)."""
        if self.important.count() == 0:
            return []

        results = self.important.get(include=["documents", "metadatas"])
        memories = []
        for i in range(len(results["ids"])):
            memories.append({
                "id": results["ids"][i],
                "exchange": results["documents"][i],
                "metadata": results["metadatas"][i],
            })

        memories.sort(
            key=lambda x: x["metadata"].get("timestamp", ""),
            reverse=True,
        )
        return memories

    # ─── Utilities ────────────────────────────────────────────────────

    def _format_results(self, results: dict) -> list[dict]:
        """Format ChromaDB query results into a clean list."""
        formatted = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return formatted

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):
            similarity = 1.0 / (1.0 + distances[i])
            formatted.append({
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i],
                "score": similarity,
            })

        return formatted

    def get_memory_stats(self) -> dict:
        """Get statistics about all memory collections."""
        return {
            "episodic_count": self.episodic.count(),
            "semantic_count": self.semantic.count(),
            "important_count": self.important.count(),
        }

    def delete_semantic_fact(self, fact_id: str):
        """Delete a specific semantic fact."""
        try:
            self.semantic.delete(ids=[fact_id])
        except Exception as e:
            print(f"[LTM] Error deleting fact {fact_id}: {e}")

    def clear_all(self):
        """Clear all long-term memories (use with caution)."""
        for collection_name in ["episodic_memory", "semantic_memory", "important_moments"]:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass

        # Recreate collections
        self.episodic = self.client.get_or_create_collection(name="episodic_memory")
        self.semantic = self.client.get_or_create_collection(name="semantic_memory")
        self.important = self.client.get_or_create_collection(name="important_moments")
        print("[LTM] All long-term memories cleared.")
