"""
Memory System — Memory Manager

Central coordinator for all memory operations.
Manages session lifecycle, coordinates between short-term and long-term memory,
and handles memory consolidation.
"""

import uuid
import json
from datetime import datetime
from typing import Optional

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory


class MemoryManager:
    """
    Coordinates all memory operations across short-term and long-term storage.

    Responsibilities:
    - Session lifecycle (create, resume, end)
    - Memory consolidation (STM → LTM at session end)
    - Unified retrieval interface
    - Fact extraction and storage
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_dir: str = "storage/chroma_db",
        sliding_window_size: int = 10,
        max_long_term_results: int = 3,
    ):
        self.sliding_window_size = sliding_window_size
        self.max_long_term_results = max_long_term_results

        # Initialize memory systems
        self.stm = ShortTermMemory(window_size=sliding_window_size)
        self.ltm = LongTermMemory(
            model_name=model_name,
            persist_dir=persist_dir,
        )

        # Session tracking
        self.current_session_id: str = self._generate_session_id()
        self.session_active: bool = True

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"sess_{timestamp}_{short_uuid}"

    # ─── Session Lifecycle ────────────────────────────────────────────

    def start_new_session(self) -> str:
        """Start a new conversation session."""
        # End current session if active
        if self.session_active and self.stm.get_turn_count() > 0:
            self.end_session()

        self.current_session_id = self._generate_session_id()
        self.stm = ShortTermMemory(window_size=self.sliding_window_size)
        self.session_active = True
        return self.current_session_id

    def end_session(self, session_summary: str = "", topics: list[str] = None,
                    user_sentiment: str = ""):
        """
        End the current session and consolidate memories.

        Args:
            session_summary: LLM-generated summary (if empty, will use raw data).
            topics: Topics discussed in this session.
            user_sentiment: Overall sentiment detected.
        """
        if not self.session_active:
            return

        turn_count = self.stm.get_turn_count()
        if turn_count == 0:
            self.session_active = False
            return

        # Use provided summary or create a basic one
        if not session_summary:
            session_data = self.stm.get_session_summary_data()
            session_summary = self._create_basic_summary(session_data)

        # Store as episodic memory
        self.ltm.store_episodic(
            session_id=self.current_session_id,
            summary=session_summary,
            topics=topics or [],
            turn_count=turn_count,
            user_sentiment=user_sentiment,
        )

        self.session_active = False
        print(f"[MemoryManager] Session {self.current_session_id} ended. "
              f"{turn_count} turns consolidated to episodic memory.")

    def _create_basic_summary(self, session_data: dict) -> str:
        """Create a basic summary from session data when LLM summary isn't available."""
        turns = session_data.get("all_turns", [])
        if not turns:
            return "Empty conversation session."

        # Extract user messages for a basic summary
        user_messages = [t["content"] for t in turns if t["role"] == "user"]
        topics_discussed = ", ".join(user_messages[:5])  # First 5 user messages as topic hints

        return (
            f"Conversation session with {len(turns)} turns. "
            f"User discussed: {topics_discussed}"
        )

    # ─── Turn Management ─────────────────────────────────────────────

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.stm.add_turn("user", content)

    def add_assistant_message(self, content: str):
        """Add an assistant (Turing) message to the conversation."""
        self.stm.add_turn("assistant", content)

    # ─── Memory Retrieval ─────────────────────────────────────────────

    def get_conversation_context(self) -> dict:
        """
        Get full conversation context for prompt assembly.

        Returns:
            Dict with:
            - recent_turns: list of recent conversation turns
            - summary: compressed summary of older turns
            - total_turns: total turn count
        """
        return self.stm.get_context_for_prompt()

    def get_relevant_memories(self, query: str) -> dict:
        """
        Retrieve all relevant memories for a given query.

        Args:
            query: The user's current message.

        Returns:
            Dict with:
            - episodic: relevant past session summaries
            - semantic: relevant user facts
            - important: relevant important moments
        """
        episodic = self.ltm.retrieve_episodic(query, top_k=self.max_long_term_results)
        semantic = self.ltm.retrieve_semantic(query, top_k=5)
        important = self.ltm.retrieve_important(query, top_k=2)

        return {
            "episodic": episodic,
            "semantic": semantic,
            "important": important,
        }

    def get_all_user_facts(self) -> list[dict]:
        """Get all semantic facts about the user (for prompt context)."""
        return self.ltm.get_all_semantic()

    # ─── Fact Management ──────────────────────────────────────────────

    def store_user_fact(self, fact: str, category: str, confidence: float = 0.8):
        """
        Store a fact about the user in semantic memory.

        Args:
            fact: The fact text (e.g., "User's name is Faaiz").
            category: Category (personal_info, interests, expertise, preferences).
            confidence: Confidence level.
        """
        # Generate a deterministic ID based on the fact content
        import hashlib
        fact_id = f"fact_{hashlib.md5(fact.lower().encode()).hexdigest()[:12]}"

        self.ltm.store_semantic(
            fact_id=fact_id,
            fact=fact,
            category=category,
            confidence=confidence,
        )

    def store_important_moment(self, exchange: str, importance_reason: str):
        """
        Store an important moment from the current conversation.

        Args:
            exchange: The exchange text.
            importance_reason: Why this was important.
        """
        moment_id = f"moment_{self.current_session_id}_{uuid.uuid4().hex[:8]}"

        self.ltm.store_important(
            moment_id=moment_id,
            exchange=exchange,
            importance_reason=importance_reason,
            session_id=self.current_session_id,
        )

    # ─── Compression ──────────────────────────────────────────────────

    def needs_compression(self) -> bool:
        """Check if the conversation needs compression."""
        return self.stm.needs_compression()

    def get_overflow_text(self) -> str:
        """Get the overflow text that needs to be compressed."""
        return self.stm.format_overflow_for_compression()

    def set_compression_result(self, summary: str):
        """Set the compressed summary after LLM generates it."""
        self.stm.set_compressed_summary(summary)

    # ─── Dashboard Data ───────────────────────────────────────────────

    def get_dashboard_data(self) -> dict:
        """Get all data needed for the memory visualization dashboard."""
        stats = self.ltm.get_memory_stats()

        return {
            "stats": {
                "total_sessions": stats["episodic_count"],
                "total_facts": stats["semantic_count"],
                "total_moments": stats["important_count"],
                "current_session_turns": self.stm.get_turn_count(),
                "current_session_id": self.current_session_id,
            },
            "sessions": self.ltm.get_all_episodic(),
            "facts": self.ltm.get_all_semantic(),
            "moments": self.ltm.get_all_important(),
        }

    def delete_user_fact(self, fact_id: str):
        """Delete a specific user fact from semantic memory."""
        self.ltm.delete_semantic_fact(fact_id)

    def clear_all_memories(self):
        """Clear all memories (STM + LTM). Use with caution."""
        self.stm.clear()
        self.ltm.clear_all()
        print("[MemoryManager] All memories cleared.")

    def get_formatted_memory_context(self, query: str) -> str:
        """
        Get a formatted string of relevant memories for inclusion in the prompt.

        Args:
            query: The user's current message.

        Returns:
            Formatted string with all relevant memory context.
        """
        memories = self.get_relevant_memories(query)
        parts = []

        # Semantic facts (always include if available)
        all_facts = self.get_all_user_facts()
        if all_facts:
            facts_text = "\n".join([f"- {f['fact']}" for f in all_facts])
            parts.append(f"[Known facts about the user]\n{facts_text}")

        # Episodic memories (relevant past sessions)
        if memories["episodic"]:
            episodic_text = "\n".join([
                f"- Previous session: {m['text']}"
                for m in memories["episodic"]
            ])
            parts.append(f"[Relevant past conversations]\n{episodic_text}")

        # Important moments
        if memories["important"]:
            moments_text = "\n".join([
                f"- Significant exchange: {m['text']}"
                for m in memories["important"]
            ])
            parts.append(f"[Important moments from past interactions]\n{moments_text}")

        return "\n\n".join(parts) if parts else ""
