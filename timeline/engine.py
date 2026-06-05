"""
Timeline Engine

Makes the agent aware of Turing's historical timeline.
Supports timeline-aware retrieval filtering and era-appropriate responses.
"""

import json
import re
from pathlib import Path
from typing import Optional


class TimelineEngine:
    """
    Provides timeline awareness for the Turing Digital Twin.

    Capabilities:
    - Determine which period a query refers to
    - Filter RAG retrieval by relevant time period
    - Provide era-appropriate context for the persona
    - Power the timeline visualization tab
    """

    def __init__(self, timeline_path: str = "data/timeline.json"):
        self.timeline_path = Path(timeline_path)
        self.timeline_data = self._load_timeline()
        self.periods = self.timeline_data.get("periods", [])
        self.contemporaries = self.timeline_data.get("contemporaries", {})

        # Build a flat list of all events for quick lookup
        self.all_events = []
        for period in self.periods:
            for event in period.get("key_events", []):
                self.all_events.append({
                    "year": event["year"],
                    "event": event["event"],
                    "period_label": period["label"],
                })

        # Sort events chronologically
        self.all_events.sort(key=lambda x: x["year"])

    def _load_timeline(self) -> dict:
        """Load timeline data from JSON file."""
        if not self.timeline_path.exists():
            print(f"[Timeline] Warning: Timeline file not found at {self.timeline_path}")
            return {"periods": [], "contemporaries": {}}

        try:
            with open(self.timeline_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Timeline] Error loading timeline: {e}")
            return {"periods": [], "contemporaries": {}}

    def detect_time_reference(self, query: str) -> Optional[dict]:
        """
        Detect if a query references a specific time period.

        Args:
            query: The user's message.

        Returns:
            Dict with 'year', 'year_range', 'period' if time reference found,
            or None if no time reference detected.
        """
        # Pattern 1: Explicit year mention (e.g., "in 1950", "around 1936")
        year_match = re.search(r"\b(1[89]\d{2}|20[0-2]\d)\b", query)
        if year_match:
            year = int(year_match.group(1))
            period = self.get_period_for_year(year)
            return {
                "year": year,
                "year_range": (period["years"][0], period["years"][1]) if period else None,
                "period": period,
                "type": "explicit_year",
            }

        # Pattern 2: Period references
        period_keywords = {
            "bletchley": (1939, 1945),
            "war": (1939, 1945),
            "enigma": (1939, 1945),
            "bombe": (1939, 1945),
            "princeton": (1936, 1938),
            "cambridge": (1931, 1935),
            "childhood": (1912, 1930),
            "school": (1912, 1930),
            "sherborne": (1912, 1930),
            "npl": (1945, 1948),
            "ace": (1945, 1948),
            "manchester": (1948, 1954),
            "morphogenesis": (1948, 1954),
            "turing test": (1948, 1954),
            "imitation game": (1948, 1954),
        }

        query_lower = query.lower()
        for keyword, year_range in period_keywords.items():
            if keyword in query_lower:
                period = self.get_period_for_year(year_range[0])
                return {
                    "year": None,
                    "year_range": year_range,
                    "period": period,
                    "type": "keyword_match",
                    "keyword": keyword,
                }

        # Pattern 3: Relative time references
        relative_patterns = {
            r"early (?:life|years|career)": (1912, 1935),
            r"(?:during|in) the war": (1939, 1945),
            r"after the war": (1945, 1954),
            r"later (?:work|years|life|career)": (1948, 1954),
            r"final (?:years|days|work)": (1948, 1954),
        }

        for pattern, year_range in relative_patterns.items():
            if re.search(pattern, query_lower):
                period = self.get_period_for_year(year_range[0])
                return {
                    "year": None,
                    "year_range": year_range,
                    "period": period,
                    "type": "relative_reference",
                }

        return None

    def get_period_for_year(self, year: int) -> Optional[dict]:
        """Get the period that contains a given year."""
        for period in self.periods:
            start, end = period["years"]
            if start <= year <= end:
                return period
        return None

    def get_timeline_context(self, query: str) -> str:
        """
        Get timeline context for inclusion in the prompt.

        Args:
            query: The user's message.

        Returns:
            A formatted string with relevant timeline context.
        """
        time_ref = self.detect_time_reference(query)

        if not time_ref or not time_ref.get("period"):
            return ""

        period = time_ref["period"]
        year = time_ref.get("year")

        context_parts = [
            f"[Timeline Context: {period['label']} ({period['years'][0]}-{period['years'][1]})]",
            period["description"],
        ]

        # Add specific events near the mentioned year
        if year:
            nearby_events = [
                e for e in self.all_events
                if abs(e["year"] - year) <= 2
            ]
            if nearby_events:
                events_text = "\n".join([f"  - {e['year']}: {e['event']}" for e in nearby_events])
                context_parts.append(f"Nearby events:\n{events_text}")

        # Add what knowledge was available at that time
        if "knowledge_available" in period:
            knowledge = ", ".join(period["knowledge_available"])
            context_parts.append(f"Knowledge domains active at this time: {knowledge}")

        return "\n".join(context_parts)

    def get_year_range_for_retrieval(self, query: str) -> Optional[tuple[int, int]]:
        """
        Get a year range for filtering RAG retrieval based on query context.

        Args:
            query: The user's message.

        Returns:
            (start_year, end_year) tuple or None if no filtering needed.
        """
        time_ref = self.detect_time_reference(query)
        if time_ref and time_ref.get("year_range"):
            return time_ref["year_range"]
        return None

    def is_post_death_topic(self, query: str) -> bool:
        """Check if a query asks about topics after Turing's death (1954)."""
        # Check for modern technology terms
        modern_terms = [
            "internet", "deep learning", "neural network", "transformer",
            "GPT", "ChatGPT", "modern AI", "smartphone", "social media",
            "quantum computing", "blockchain", "cloud computing",
            "machine learning today", "current", "nowadays", "today",
        ]

        query_lower = query.lower()
        return any(term in query_lower for term in modern_terms)

    # ─── Dashboard Data ───────────────────────────────────────────────

    def get_all_periods(self) -> list[dict]:
        """Get all timeline periods for dashboard display."""
        return self.periods

    def get_all_events(self) -> list[dict]:
        """Get all events in chronological order for dashboard display."""
        return self.all_events

    def get_contemporaries(self) -> dict:
        """Get the contemporaries reference map."""
        return self.contemporaries

    def get_period_description(self, period_label: str) -> str:
        """Get a detailed description for a specific period."""
        for period in self.periods:
            if period["label"] == period_label:
                events_text = "\n".join([
                    f"  • {e['year']}: {e['event']}"
                    for e in period.get("key_events", [])
                ])
                return f"{period['description']}\n\nKey events:\n{events_text}"
        return "Period not found."
