"""
Core — Prompt Builder Module

Assembles the final prompt sent to Gemini by combining:
1. System prompt (persona)
2. Timeline context
3. Long-term memory context
4. RAG context (retrieved sources)
5. Conversation summary (compressed old turns)
6. Recent conversation turns
7. Current user query
8. Response instructions
"""

from typing import Optional
from google.genai import types


class PromptBuilder:
    """
    Assembles the multi-layered prompt for the Gemini LLM.

    Token budget (approximate):
    - System prompt: ~500 tokens
    - Timeline context: ~100 tokens
    - Long-term memory: ~200 tokens
    - RAG context: ~800 tokens
    - Conversation summary: ~200 tokens
    - Recent turns: ~2000 tokens
    - User query: variable
    - Response instructions: ~100 tokens
    Total: ~4000 tokens (well within Gemini's context window)
    """

    def __init__(self):
        pass

    def build_prompt(
        self,
        system_prompt: str,
        user_query: str,
        rag_context: str = "",
        memory_context: str = "",
        timeline_context: str = "",
        conversation_summary: str = "",
        recent_turns: list[dict] = None,
        is_post_death_topic: bool = False,
    ) -> tuple[str, list[dict]]:
        """
        Build the complete prompt for Gemini.

        Args:
            system_prompt: The Turing persona system prompt.
            user_query: The current user message.
            rag_context: Retrieved source chunks formatted as text.
            memory_context: Formatted long-term memory context.
            timeline_context: Timeline-relevant context.
            conversation_summary: Compressed summary of older turns.
            recent_turns: Recent conversation turns for chat history.
            is_post_death_topic: Whether the query involves post-1954 topics.

        Returns:
            Tuple of (system_instruction, conversation_history) where
            conversation_history is a list of types.Content for Gemini.
        """
        # ── 1. Build the system instruction ──
        system_parts = [system_prompt]

        # Add timeline context
        if timeline_context:
            system_parts.append(f"\n\n{timeline_context}")

        # Add post-death advisory
        if is_post_death_topic:
            system_parts.append(
                "\n\n[IMPORTANT: The user is asking about topics or technologies "
                "that came after your death in 1954. You may discuss these but MUST "
                "frame your responses speculatively, referencing your own earlier work "
                "that relates to the topic. Example: 'This development came after my time, "
                "but the germ of the idea — that machines might learn from experience — "
                "is something I explored in my 1950 paper...']"
            )

        # Add long-term memory context
        if memory_context:
            system_parts.append(f"\n\n[Memory — What you remember about this user]\n{memory_context}")

        # Add RAG context
        if rag_context:
            system_parts.append(
                f"\n\n[Retrieved Sources — Your works and related materials]\n"
                f"Use these sources to ground your response. Reference them naturally "
                f"in conversation, not as footnotes.\n\n{rag_context}"
            )

        # Add response instructions
        system_parts.append(self._get_response_instructions())

        full_system_instruction = "\n".join(system_parts)

        # ── 2. Build conversation history ──
        conversation_history = []

        # Add conversation summary as a system-injected context message
        if conversation_summary:
            conversation_history.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"[Earlier in our conversation, we discussed: {conversation_summary}]")],
            ))
            conversation_history.append(types.Content(
                role="model",
                parts=[types.Part.from_text(text="Indeed, I recall our earlier discussion. Please, do continue.")],
            ))

        # Add recent conversation turns
        if recent_turns:
            for turn in recent_turns:
                role = "user" if turn["role"] == "user" else "model"
                conversation_history.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=turn["content"])],
                ))

        # Add the current user query
        conversation_history.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_query)],
        ))

        return full_system_instruction, conversation_history

    def _get_response_instructions(self) -> str:
        """Get the response instruction block appended to the system prompt."""
        return """

[Response Guidelines]
- Respond in character as Alan Turing's digital twin
- Use British English spelling throughout
- Weave source references naturally into your speech
- If you recall facts about the user from memory, reference them naturally
- Keep responses substantive but not excessively long (aim for 150-400 words unless the topic demands more)
- When discussing technical topics, balance rigour with accessibility
- If the user asks something outside your expertise, say so honestly and redirect to what you do know
"""

    def build_compression_messages(self, compression_prompt: str) -> list:
        """Build messages for the conversation compression call."""
        return [types.Content(role="user", parts=[types.Part.from_text(text=compression_prompt)])]

    def build_fact_extraction_messages(self, extraction_prompt: str) -> list:
        """Build messages for the fact extraction call."""
        return [types.Content(role="user", parts=[types.Part.from_text(text=extraction_prompt)])]

    def build_session_summary_messages(self, summary_prompt: str) -> list:
        """Build messages for the session summary call."""
        return [types.Content(role="user", parts=[types.Part.from_text(text=summary_prompt)])]

    def build_importance_check_messages(self, importance_prompt: str) -> list:
        """Build messages for the importance check call."""
        return [types.Content(role="user", parts=[types.Part.from_text(text=importance_prompt)])]

