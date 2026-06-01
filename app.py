"""
Digital Twin of Alan Turing — Gradio Application

Main entry point for the interactive demo.
Provides three tabs:
1. Chat — Conversation with Turing's Digital Twin
2. Memory Dashboard — Visualise what the agent remembers
3. Timeline — Interactive timeline of Turing's life
"""

import os
import sys
import gradio as gr
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.orchestrator import Orchestrator
from dashboard.memory_viz import (
    format_full_dashboard,
    format_timeline_display,
    format_sessions_display,
    format_facts_display,
    format_moments_display,
    format_stats_display,
)


# ─── Global State ─────────────────────────────────────────────────

orchestrator = None


def get_orchestrator():
    """Lazy-initialise the orchestrator."""
    global orchestrator
    if orchestrator is None:
        orchestrator = Orchestrator(config_path="config.yaml")
    return orchestrator


# ─── Chat Functions ───────────────────────────────────────────────

def chat_response(message: str, history: list) -> tuple:
    """
    Process a user message and generate a response.

    Args:
        message: The user's input.
        history: Gradio chat history (list of message dicts).

    Returns:
        Updated history and source info.
    """
    if not message.strip():
        return history, ""

    orch = get_orchestrator()

    # Process through the full pipeline
    result = orch.process_message(message)

    response = result["response"]

    # Format source citations
    sources_text = ""
    if result.get("sources"):
        source_items = []
        for s in result["sources"]:
            item = f"📄 {s['title']}"
            if s.get("year"):
                item += f" ({s['year']})"
            item += f" — relevance: {s['relevance']}"
            source_items.append(item)
        sources_text = "**Sources referenced:**\n" + "\n".join(source_items)

    if result.get("timeline_period"):
        sources_text += f"\n\n🕐 **Timeline period:** {result['timeline_period']}"

    # Update history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    return history, sources_text


def new_session():
    """Start a new conversation session."""
    orch = get_orchestrator()
    session_id = orch.start_new_session()
    return get_greeting(), f"✅ New session started: `{session_id}`"


def get_greeting():
    """Get Turing's greeting message."""
    return [{"role": "assistant", "content": (
        "Good day. I am a digital reconstruction of Alan Turing — mathematician, "
        "logician, cryptanalyst, and, if I may say, a rather keen long-distance runner. "
        "I find it rather fitting, perhaps even a touch ironic, that a digital emulation "
        "of me should exist, given that I spent a good portion of my life pondering whether "
        "machines could truly think.\n\n"
        "What shall we discuss? I am happy to talk about computability, cryptography, "
        "the question of machine intelligence, mathematical biology, or indeed anything "
        "that catches your intellectual curiosity."
    )}]


# ─── Dashboard Functions ──────────────────────────────────────────

def refresh_dashboard():
    """Refresh the memory dashboard data."""
    orch = get_orchestrator()
    data = orch.get_dashboard_data()

    stats = format_stats_display(data.get("stats", {}))
    facts = format_facts_display(data.get("facts", []))
    sessions = format_sessions_display(data.get("sessions", []))
    moments = format_moments_display(data.get("moments", []))

    return stats, facts, sessions, moments


def clear_memories():
    """Clear all stored memories."""
    orch = get_orchestrator()
    orch.clear_all_data()
    return refresh_dashboard()


# ─── Timeline Functions ───────────────────────────────────────────

def get_timeline():
    """Get the timeline display."""
    orch = get_orchestrator()
    data = orch.get_timeline_data()
    return format_timeline_display(data)


def get_period_detail(period_label: str):
    """Get detailed info about a specific period."""
    orch = get_orchestrator()
    description = orch.timeline.get_period_description(period_label)
    return description


# ─── Build the Gradio Interface ───────────────────────────────────

def create_app():
    """Create and configure the Gradio application."""

    # Custom CSS for a scholarly, premium feel
    custom_css = """
    /* Overall theme */
    .gradio-container {
        font-family: 'Georgia', 'Times New Roman', serif !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }

    .main-header h1 {
        font-size: 2em;
        margin: 0;
        color: #e8d5b7;
    }

    .main-header p {
        color: #a0aec0;
        margin-top: 5px;
        font-style: italic;
    }

    /* Chat styling */
    .chatbot {
        font-family: 'Georgia', serif !important;
        font-size: 15px !important;
    }

    /* Source panel */
    .source-panel {
        background: #f8f9fa;
        border-left: 3px solid #0f3460;
        padding: 10px;
        border-radius: 4px;
        font-size: 13px;
    }

    /* Dashboard cards */
    .dashboard-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        margin: 5px 0;
    }

    /* Timeline styling */
    .timeline-display {
        font-family: 'Courier New', monospace;
    }

    /* Button styling */
    .primary-btn {
        background: linear-gradient(135deg, #0f3460, #16213e) !important;
        color: white !important;
        border: none !important;
    }

    .danger-btn {
        background: #e74c3c !important;
        color: white !important;
    }
    """

    with gr.Blocks(
        title="Digital Twin of Alan Turing",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=["Georgia", "serif"],
        ),
        css=custom_css,
    ) as app:

        # ── Header ──
        gr.HTML("""
        <div class="main-header">
            <h1>🧠 Digital Twin of Alan Turing</h1>
            <p>"We can only see a short distance ahead, but we can see plenty there that needs to be done." — Alan Turing, 1950</p>
        </div>
        """)

        with gr.Tabs() as tabs:

            # ════════════════════════════════════════════
            # TAB 1: CHAT
            # ════════════════════════════════════════════
            with gr.TabItem("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            value=get_greeting(),
                            height=500,
                            show_label=False,
                            avatar_images=(
                                None,  # User avatar
                                None,  # Turing avatar
                            ),
                        )

                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Ask Turing anything — computability, cryptography, artificial intelligence, morphogenesis...",
                                show_label=False,
                                scale=5,
                                lines=1,
                                max_lines=3,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)

                        with gr.Row():
                            new_session_btn = gr.Button("🔄 New Session", size="sm")
                            session_status = gr.Textbox(
                                show_label=False,
                                interactive=False,
                                scale=3,
                                max_lines=1,
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("### 📚 Sources & Context")
                        sources_display = gr.Markdown(
                            value="*Sources will appear here after each response.*",
                            elem_classes=["source-panel"],
                        )

                        gr.Markdown("### ℹ️ About")
                        gr.Markdown(
                            "This is a **Digital Twin** of Alan Turing — "
                            "an AI agent that emulates his knowledge, reasoning style, "
                            "and communication patterns.\n\n"
                            "Built with:\n"
                            "- 🤖 Gemini 2.5 Flash\n"
                            "- 📚 RAG over Turing's works\n"
                            "- 🧠 Dual memory system\n"
                            "- 🕐 Timeline awareness"
                        )

                # Chat event handlers
                def on_submit(message, history):
                    return chat_response(message, history)

                msg_input.submit(
                    fn=on_submit,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, sources_display],
                ).then(
                    fn=lambda: "",
                    outputs=[msg_input],
                )

                send_btn.click(
                    fn=on_submit,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, sources_display],
                ).then(
                    fn=lambda: "",
                    outputs=[msg_input],
                )

                new_session_btn.click(
                    fn=new_session,
                    outputs=[chatbot, session_status],
                )

            # ════════════════════════════════════════════
            # TAB 2: MEMORY DASHBOARD
            # ════════════════════════════════════════════
            with gr.TabItem("Memory Dashboard"):
                gr.Markdown("## Memory Visualization Dashboard\n_This dashboard shows what the Digital Twin remembers._")

                refresh_btn = gr.Button("Refresh Dashboard", variant="primary")
                clear_btn = gr.Button("Clear All Memories", variant="secondary")

                memory_display = gr.Textbox(
                    value="Click 'Refresh Dashboard' to load.",
                    label="Memory Data",
                    interactive=False,
                    lines=20
                )

                def refresh_dashboard_combined():
                    stats, facts, sessions, moments = refresh_dashboard()
                    return f"{stats}\n\n{'='*40}\n\n{facts}\n\n{'='*40}\n\n{sessions}\n\n{'='*40}\n\n{moments}"

                def clear_dashboard_combined():
                    stats, facts, sessions, moments = clear_memories()
                    return f"{stats}\n\n{'='*40}\n\n{facts}\n\n{'='*40}\n\n{sessions}\n\n{'='*40}\n\n{moments}"

                refresh_btn.click(
                    fn=refresh_dashboard_combined,
                    outputs=[memory_display],
                )

                clear_btn.click(
                    fn=clear_dashboard_combined,
                    outputs=[memory_display],
                )

            # ════════════════════════════════════════════
            # TAB 3: TIMELINE
            # ════════════════════════════════════════════
            with gr.TabItem("🕰️ Timeline"):
                gr.Markdown("## 🕰️ Alan Turing's Life Timeline")
                gr.Markdown(
                    "_Explore the key periods and events of Turing's life. "
                    "The timeline engine uses this data to provide historically "
                    "accurate, era-appropriate responses._"
                )

                timeline_display = gr.Markdown(value="*Loading timeline...*")

                with gr.Row():
                    period_selector = gr.Dropdown(
                        choices=[
                            "Early Life & Education",
                            "Cambridge & King's College",
                            "Computability & Princeton",
                            "Bletchley Park & War Work",
                            "ACE & National Physical Laboratory",
                            "Manchester & Final Years",
                        ],
                        label="Select a period for details",
                        interactive=True,
                    )
                    period_detail = gr.Markdown(value="")

                # Load timeline on tab visit
                app.load(
                    fn=get_timeline,
                    outputs=[timeline_display],
                )

                period_selector.change(
                    fn=get_period_detail,
                    inputs=[period_selector],
                    outputs=[period_detail],
                )

    return app


# ─── Entry Point ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Digital Twin of Alan Turing")
    print("  Starting Gradio application...")
    print("=" * 60)

    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
