"""
Dashboard — Memory Visualization Module

Provides Gradio components for visualising the agent's memory:
- Session history (episodic memory)
- Learned facts about the user (semantic memory)
- Important moments flagged during conversations
- Memory statistics
"""

import json
from typing import Optional


def format_sessions_display(sessions: list[dict]) -> str:
    """Format episodic memories for display in the dashboard."""
    if not sessions:
        return "📭 No past sessions recorded yet. Start a conversation to build memory!"

    lines = ["## 📋 Past Sessions\n"]
    for i, session in enumerate(sessions, 1):
        meta = session.get("metadata", {})
        summary = session.get("summary", "No summary available")
        timestamp = meta.get("timestamp", "Unknown time")
        turn_count = meta.get("turn_count", "?")
        sentiment = meta.get("user_sentiment", "")

        # Parse topics if stored as JSON string
        topics_raw = meta.get("topics", "[]")
        try:
            topics = json.loads(topics_raw) if isinstance(topics_raw, str) else topics_raw
        except json.JSONDecodeError:
            topics = []

        topics_text = ", ".join(topics) if topics else "General discussion"

        lines.append(f"### Session {i}")
        lines.append(f"🕐 **Time:** {timestamp[:19].replace('T', ' ')}")
        lines.append(f"💬 **Turns:** {turn_count}")
        if sentiment:
            lines.append(f"😊 **Sentiment:** {sentiment}")
        lines.append(f"🏷️ **Topics:** {topics_text}")
        lines.append(f"📝 **Summary:** {summary}")
        lines.append("---")

    return "\n".join(lines)


def format_facts_display(facts: list[dict]) -> str:
    """Format semantic memories (user facts) for display."""
    if not facts:
        return "🔍 No facts learned about you yet. The more we converse, the better I shall know you!"

    lines = ["## 🧠 What I Remember About You\n"]

    # Group by category
    categories = {}
    for fact in facts:
        meta = fact.get("metadata", {})
        category = meta.get("category", "general")
        if category not in categories:
            categories[category] = []
        categories[category].append(fact)

    category_icons = {
        "personal_info": "👤",
        "interests": "⭐",
        "expertise": "🎓",
        "preferences": "💡",
        "background": "📚",
        "general": "📌",
    }

    for category, category_facts in categories.items():
        icon = category_icons.get(category, "📌")
        lines.append(f"### {icon} {category.replace('_', ' ').title()}")
        for fact in category_facts:
            fact_text = fact.get("fact", "Unknown")
            confidence = fact.get("metadata", {}).get("confidence", "?")
            lines.append(f"- {fact_text} *(confidence: {confidence})*")
        lines.append("")

    return "\n".join(lines)


def format_moments_display(moments: list[dict]) -> str:
    """Format important moments for display."""
    if not moments:
        return "⭐ No particularly significant moments recorded yet. Engage me with challenging questions!"

    lines = ["## ⭐ Important Moments\n"]

    for i, moment in enumerate(moments, 1):
        meta = moment.get("metadata", {})
        exchange = moment.get("exchange", "")
        reason = meta.get("importance_reason", "Flagged as significant")
        timestamp = meta.get("timestamp", "Unknown time")

        lines.append(f"### Moment {i}")
        lines.append(f"🕐 **When:** {timestamp[:19].replace('T', ' ')}")
        lines.append(f"💡 **Why important:** {reason}")
        lines.append(f"```\n{exchange[:300]}{'...' if len(exchange) > 300 else ''}\n```")
        lines.append("---")

    return "\n".join(lines)


def format_stats_display(stats: dict) -> str:
    """Format memory statistics for display."""
    lines = [
        "## 📊 Memory Statistics\n",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| 📋 Total Sessions | {stats.get('total_sessions', 0)} |",
        f"| 🧠 Facts Remembered | {stats.get('total_facts', 0)} |",
        f"| ⭐ Important Moments | {stats.get('total_moments', 0)} |",
        f"| 💬 Current Session Turns | {stats.get('current_session_turns', 0)} |",
        "",
        f"🔑 **Current Session ID:** `{stats.get('current_session_id', 'N/A')}`",
    ]
    return "\n".join(lines)


def format_full_dashboard(dashboard_data: dict) -> str:
    """Format the complete dashboard view."""
    stats = format_stats_display(dashboard_data.get("stats", {}))
    facts = format_facts_display(dashboard_data.get("facts", []))
    sessions = format_sessions_display(dashboard_data.get("sessions", []))
    moments = format_moments_display(dashboard_data.get("moments", []))

    return f"{stats}\n\n---\n\n{facts}\n\n---\n\n{sessions}\n\n---\n\n{moments}"


def format_timeline_display(timeline_data: dict) -> str:
    """Format the timeline data for the timeline tab."""
    periods = timeline_data.get("periods", [])
    events = timeline_data.get("events", [])
    contemporaries = timeline_data.get("contemporaries", {})

    lines = ["## 🕰️ Timeline of Alan Turing's Life\n"]

    # Visual timeline
    lines.append("```")
    lines.append("1912 ────── 1936 ────── 1939 ────── 1945 ────── 1948 ────── 1954")
    lines.append("  │          │          │           │           │          │")
    lines.append("Birth    Computable  Bletchley    War ends     ACE      Death")
    lines.append("         Numbers     Park                    Manchester")
    lines.append("```\n")

    # Period details
    for period in periods:
        start, end = period["years"]
        lines.append(f"### 📅 {period['label']} ({start}–{end})")
        lines.append(f"_{period['description']}_\n")

        key_events = period.get("key_events", [])
        if key_events:
            for event in key_events:
                lines.append(f"- **{event['year']}**: {event['event']}")
            lines.append("")

    # Contemporaries
    lines.append("---\n### 👥 Contemporaries & Colleagues\n")
    for name, description in contemporaries.items():
        lines.append(f"- **{name}**: {description}")

    return "\n".join(lines)
