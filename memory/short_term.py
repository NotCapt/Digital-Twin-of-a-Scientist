"""
Memory System — Short-Term Memory Module

Manages within-session conversation memory using a sliding window
with summary compression for older turns.
"""

from typing import Optional
from datetime import datetime


class ShortTermMemory:
    """
    Manages conversation history within a single session.

    Uses a sliding window of recent turns kept verbatim,
    with older turns compressed into summaries.
    """

    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size: Number of recent turns to keep in full.
        """
        self.window_size = window_size
        self.conversation_history: list[dict] = []
        self.compressed_summary: str = ""
        self.session_start_time: str = datetime.now().isoformat()

    def add_turn(self, role: str, content: str):
        """
        Add a conversation turn.

        Args:
            role: 'user' or 'assistant'
            content: The message content.
        """
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "turn_index": len(self.conversation_history),
        }
        self.conversation_history.append(turn)

    def get_recent_turns(self) -> list[dict]:
        """Get the most recent turns within the sliding window."""
        return self.conversation_history[-self.window_size * 2 :]  # *2 because each turn has user+assistant

    def get_all_turns(self) -> list[dict]:
        """Get all turns in the conversation."""
        return self.conversation_history.copy()

    def get_turn_count(self) -> int:
        """Get the total number of turns."""
        return len(self.conversation_history)

    def needs_compression(self) -> bool:
        """Check if the conversation is long enough to need compression."""
        return len(self.conversation_history) > self.window_size * 2

    def get_overflow_turns(self) -> list[dict]:
        """Get turns that have fallen outside the sliding window (need compression)."""
        if not self.needs_compression():
            return []
        return self.conversation_history[: -self.window_size * 2]

    def set_compressed_summary(self, summary: str):
        """Set the compressed summary of older turns."""
        self.compressed_summary = summary

    def get_compressed_summary(self) -> str:
        """Get the compressed summary of older turns."""
        return self.compressed_summary

    def get_context_for_prompt(self) -> dict:
        """
        Get the full conversation context for prompt assembly.

        Returns:
            Dict with 'summary' (str) and 'recent_turns' (list[dict]).
        """
        recent = self.get_recent_turns()

        return {
            "summary": self.compressed_summary,
            "recent_turns": recent,
            "total_turns": len(self.conversation_history),
        }

    def format_turns_for_prompt(self) -> str:
        """Format recent turns as a string for inclusion in the LLM prompt."""
        recent = self.get_recent_turns()
        if not recent:
            return ""

        formatted = []
        for turn in recent:
            role_label = "User" if turn["role"] == "user" else "Turing"
            formatted.append(f"{role_label}: {turn['content']}")

        return "\n\n".join(formatted)

    def format_overflow_for_compression(self) -> str:
        """Format overflow turns as a string for the LLM to compress."""
        overflow = self.get_overflow_turns()
        if not overflow:
            return ""

        formatted = []
        for turn in overflow:
            role_label = "User" if turn["role"] == "user" else "Turing"
            formatted.append(f"{role_label}: {turn['content']}")

        return "\n\n".join(formatted)

    def get_session_summary_data(self) -> dict:
        """Get data needed for creating an episodic memory at session end."""
        return {
            "session_start": self.session_start_time,
            "session_end": datetime.now().isoformat(),
            "total_turns": len(self.conversation_history),
            "all_turns": self.conversation_history.copy(),
        }

    def clear(self):
        """Clear all conversation history."""
        self.conversation_history = []
        self.compressed_summary = ""
        self.session_start_time = datetime.now().isoformat()
