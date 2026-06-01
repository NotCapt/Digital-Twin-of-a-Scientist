"""
Core — Orchestrator Module

Central coordinator that processes each user message through the full pipeline:
1. Parse & classify the query
2. Retrieve RAG context
3. Fetch memories (STM + LTM)
4. Check timeline relevance
5. Build the prompt
6. Call Gemini
7. Post-process (persona validation)
8. Update memory
"""

import json
import os
import yaml
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from core.prompt_builder import PromptBuilder
from rag.retriever import Retriever
from memory.manager import MemoryManager
from persona.system_prompt import (
    get_system_prompt,
    get_compression_prompt,
    get_fact_extraction_prompt,
    get_session_summary_prompt,
    get_importance_check_prompt,
)
from persona.validator import PersonaValidator
from timeline.engine import TimelineEngine


class Orchestrator:
    """
    Central coordinator for the Digital Twin of Alan Turing.

    Manages the full pipeline from user input to response generation,
    coordinating RAG retrieval, memory, timeline, and persona systems.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize all subsystems.

        Args:
            config_path: Path to the configuration YAML file.
        """
        self.config = self._load_config(config_path)
        self._init_gemini()
        self._init_subsystems()

        print("[Orchestrator] All systems initialised and ready.")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            print(f"[Orchestrator] Config file not found at {config_path}, using defaults.")
            return self._default_config()

    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            "llm": {
                "model": "gemini-2.5-flash",
                "temperature": 0.7,
                "max_output_tokens": 2048,
                "top_p": 0.95,
            },
            "rag": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "top_k": 5,
                "embedding_model": "all-MiniLM-L6-v2",
                "boost_primary_source": 1.3,
                "collection_name": "turing_knowledge",
            },
            "memory": {
                "sliding_window_size": 10,
                "max_long_term_results": 3,
            },
            "storage": {
                "chroma_persist_dir": "storage/chroma_db",
            },
        }

    def _init_gemini(self):
        """Initialize the Gemini client."""
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in your .env file or environment variables."
            )

        self.gemini_client = genai.Client(api_key=api_key)
        self.model_name = self.config["llm"]["model"]
        print(f"[Orchestrator] Gemini client initialised with model: {self.model_name}")

    def _init_subsystems(self):
        """Initialize all subsystems."""
        rag_config = self.config.get("rag", {})
        mem_config = self.config.get("memory", {})
        storage_config = self.config.get("storage", {})

        persist_dir = storage_config.get("chroma_persist_dir", "storage/chroma_db")

        # RAG Retriever
        self.retriever = Retriever(
            model_name=rag_config.get("embedding_model", "all-MiniLM-L6-v2"),
            persist_dir=persist_dir,
            collection_name=rag_config.get("collection_name", "turing_knowledge"),
            top_k=rag_config.get("top_k", 5),
            boost_primary_source=rag_config.get("boost_primary_source", 1.3),
        )

        # Memory Manager
        self.memory = MemoryManager(
            model_name=rag_config.get("embedding_model", "all-MiniLM-L6-v2"),
            persist_dir=persist_dir,
            sliding_window_size=mem_config.get("sliding_window_size", 10),
            max_long_term_results=mem_config.get("max_long_term_results", 3),
        )

        # Prompt Builder
        self.prompt_builder = PromptBuilder()

        # Persona
        self.system_prompt = get_system_prompt()
        self.validator = PersonaValidator()

        # Timeline Engine
        self.timeline = TimelineEngine()

        print("[Orchestrator] Subsystems initialised: RAG, Memory, Prompt, Persona, Timeline")

    # ─── Main Response Pipeline ───────────────────────────────────────

    def process_message(self, user_message: str) -> dict:
        """
        Process a user message through the full pipeline and generate a response.

        Args:
            user_message: The user's input message.

        Returns:
            Dict with:
            - response: The generated response text
            - sources: List of source references used
            - timeline_period: The detected time period (if any)
            - memory_used: Summary of memory context used
        """
        # Step 1: Add user message to STM
        self.memory.add_user_message(user_message)

        # Step 2: Detect timeline context
        timeline_context = self.timeline.get_timeline_context(user_message)
        is_post_death = self.timeline.is_post_death_topic(user_message)
        year_range = self.timeline.get_year_range_for_retrieval(user_message)

        # Step 3: Retrieve RAG context
        rag_results = self.retriever.retrieve(
            query=user_message,
            year_range=year_range,
        )
        rag_context = self.retriever.format_context(rag_results)

        # Step 4: Get memory context
        memory_context = self.memory.get_formatted_memory_context(user_message)

        # Step 5: Handle conversation compression if needed
        conversation_summary = self.memory.stm.get_compressed_summary()
        if self.memory.needs_compression():
            compression_summary = self._compress_conversation()
            if compression_summary:
                conversation_summary = compression_summary

        # Step 6: Get recent conversation turns (excluding the current message)
        recent_context = self.memory.get_conversation_context()
        recent_turns = recent_context["recent_turns"][:-1]  # Exclude the just-added user message

        # Step 7: Build the prompt
        system_instruction, conversation_history = self.prompt_builder.build_prompt(
            system_prompt=self.system_prompt,
            user_query=user_message,
            rag_context=rag_context,
            memory_context=memory_context,
            timeline_context=timeline_context,
            conversation_summary=conversation_summary,
            recent_turns=recent_turns,
            is_post_death_topic=is_post_death,
        )

        # Step 8: Call Gemini
        response_text = self._call_gemini(system_instruction, conversation_history)

        # Step 9: Post-process — validate persona
        validation = self.validator.validate(response_text)
        if validation["severity"] == "minor":
            response_text = self.validator.auto_fix_minor(response_text)

        # Step 10: Add assistant response to STM
        self.memory.add_assistant_message(response_text)

        # Step 11: Background memory operations (async-style, non-blocking)
        self._update_memories(user_message, response_text)

        # Step 12: Prepare result
        sources = [
            {
                "title": r["metadata"].get("source_title", "Unknown"),
                "year": r["metadata"].get("year", ""),
                "type": r["metadata"].get("source_type", ""),
                "relevance": round(r["score"], 3),
            }
            for r in rag_results
        ]

        time_ref = self.timeline.detect_time_reference(user_message)

        return {
            "response": response_text,
            "sources": sources,
            "timeline_period": time_ref["period"]["label"] if time_ref and time_ref.get("period") else None,
            "memory_used": bool(memory_context),
            "validation": validation,
        }

    def _call_gemini(self, system_instruction: str, conversation_history: list[dict]) -> str:
        """
        Call the Gemini API to generate a response.

        Args:
            system_instruction: The full system prompt.
            conversation_history: The conversation history as Gemini Content list.

        Returns:
            The generated response text.
        """
        try:
            llm_config = self.config.get("llm", {})

            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=conversation_history,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=llm_config.get("temperature", 0.7),
                    max_output_tokens=llm_config.get("max_output_tokens", 2048),
                    top_p=llm_config.get("top_p", 0.95),
                ),
            )

            return response.text or "I find myself at a loss for words — most unusual. Could you rephrase your question?"

        except Exception as e:
            print(f"[Orchestrator] Gemini API error: {e}")
            return (
                "I must apologise — it appears there has been some sort of "
                "technical difficulty in transmitting my thoughts. Shall we try again?"
            )

    def _call_gemini_simple(self, prompt: str) -> str:
        """Make a simple single-turn Gemini call (for memory operations)."""
        try:
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=512,
                ),
            )
            return response.text or ""
        except Exception as e:
            print(f"[Orchestrator] Gemini simple call error: {e}")
            return ""

    # ─── Memory Update Operations ─────────────────────────────────────

    def _update_memories(self, user_message: str, assistant_response: str):
        """
        Run background memory update operations after each turn.

        - Extract facts about the user
        - Check if the exchange is important
        """
        # Fact extraction (every 3 turns to save API calls)
        if self.memory.stm.get_turn_count() % 3 == 0:
            self._extract_and_store_facts(user_message, assistant_response)

        # Importance check (every 5 turns)
        if self.memory.stm.get_turn_count() % 5 == 0:
            self._check_importance(user_message, assistant_response)

    def _extract_and_store_facts(self, user_message: str, assistant_response: str):
        """Extract facts about the user and store in semantic memory."""
        try:
            prompt = get_fact_extraction_prompt(user_message, assistant_response)
            result = self._call_gemini_simple(prompt)

            if not result:
                return

            # Parse JSON response
            # Clean up potential markdown formatting
            result = result.strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[1] if "\n" in result else result
                result = result.rsplit("```", 1)[0]
            result = result.strip()

            facts = json.loads(result)
            for fact in facts:
                if isinstance(fact, dict) and "fact" in fact:
                    self.memory.store_user_fact(
                        fact=fact["fact"],
                        category=fact.get("category", "general"),
                        confidence=fact.get("confidence", 0.7),
                    )
                    print(f"[Orchestrator] Stored fact: {fact['fact']}")

        except (json.JSONDecodeError, Exception) as e:
            # Non-critical — just skip if parsing fails
            pass

    def _check_importance(self, user_message: str, assistant_response: str):
        """Check if a conversation exchange is important enough to flag."""
        try:
            prompt = get_importance_check_prompt(user_message, assistant_response)
            result = self._call_gemini_simple(prompt)

            if not result:
                return

            # Clean up potential markdown formatting
            result = result.strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[1] if "\n" in result else result
                result = result.rsplit("```", 1)[0]
            result = result.strip()

            evaluation = json.loads(result)
            if evaluation.get("is_important"):
                exchange = f"User: {user_message}\nTuring: {assistant_response[:200]}..."
                self.memory.store_important_moment(
                    exchange=exchange,
                    importance_reason=evaluation.get("reason", "Flagged as significant"),
                )
                print(f"[Orchestrator] Important moment stored: {evaluation.get('reason')}")

        except (json.JSONDecodeError, Exception):
            pass

    def _compress_conversation(self) -> Optional[str]:
        """Compress older conversation turns into a summary."""
        try:
            overflow_text = self.memory.get_overflow_text()
            if not overflow_text:
                return None

            prompt = get_compression_prompt(overflow_text)
            summary = self._call_gemini_simple(prompt)

            if summary:
                self.memory.set_compression_result(summary)
                return summary

        except Exception as e:
            print(f"[Orchestrator] Compression error: {e}")

        return None

    # ─── Session Management ───────────────────────────────────────────

    def start_new_session(self) -> str:
        """Start a new conversation session, ending the current one."""
        # End current session with summary
        if self.memory.stm.get_turn_count() > 0:
            self._end_session_with_summary()

        session_id = self.memory.start_new_session()
        print(f"[Orchestrator] New session started: {session_id}")
        return session_id

    def _end_session_with_summary(self):
        """End the current session and create an episodic memory."""
        try:
            # Generate session summary
            all_turns = self.memory.stm.get_all_turns()
            conversation_text = "\n".join([
                f"{'User' if t['role'] == 'user' else 'Turing'}: {t['content']}"
                for t in all_turns
            ])

            prompt = get_session_summary_prompt(conversation_text)
            result = self._call_gemini_simple(prompt)

            if result:
                # Clean up potential markdown formatting
                result = result.strip()
                if result.startswith("```"):
                    result = result.split("\n", 1)[1] if "\n" in result else result
                    result = result.rsplit("```", 1)[0]
                result = result.strip()

                try:
                    summary_data = json.loads(result)
                    self.memory.end_session(
                        session_summary=summary_data.get("summary", ""),
                        topics=summary_data.get("topics", []),
                        user_sentiment=summary_data.get("user_sentiment", ""),
                    )
                except json.JSONDecodeError:
                    self.memory.end_session(session_summary=result)
            else:
                self.memory.end_session()

        except Exception as e:
            print(f"[Orchestrator] Session end error: {e}")
            self.memory.end_session()

    def end_current_session(self):
        """Public method to end the current session."""
        self._end_session_with_summary()

    # ─── Dashboard & Info ─────────────────────────────────────────────

    def get_dashboard_data(self) -> dict:
        """Get data for the memory visualization dashboard."""
        return self.memory.get_dashboard_data()

    def get_timeline_data(self) -> dict:
        """Get data for the timeline visualization tab."""
        return {
            "periods": self.timeline.get_all_periods(),
            "events": self.timeline.get_all_events(),
            "contemporaries": self.timeline.get_contemporaries(),
        }

    def get_system_info(self) -> dict:
        """Get system information for display."""
        return {
            "model": self.model_name,
            "rag_stats": self.retriever.get_stats(),
            "memory_stats": self.memory.ltm.get_memory_stats(),
            "session_id": self.memory.current_session_id,
        }

    def clear_all_data(self):
        """Clear all memories and start fresh."""
        self.memory.clear_all_memories()
        print("[Orchestrator] All data cleared.")
