"""
Digital Twin of Alan Turing — Streamlit Application

Main entry point for the interactive demo using Streamlit.
"""

import os
import sys
import streamlit as st
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

# ─── Page Config ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Digital Twin of Alan Turing",
    page_icon="🧠",
    layout="wide",
)

# ─── Custom CSS ───────────────────────────────────────────────────

st.markdown("""
<style>
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
    .stChatFloatingInputContainer {
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Global State & Init ──────────────────────────────────────────

@st.cache_resource
def get_orchestrator():
    """Initialise the orchestrator once."""
    return Orchestrator(config_path="config.yaml")

try:
    orch = get_orchestrator()
except Exception as e:
    st.error(f"Failed to initialise the application: {e}")
    st.stop()

# Initialize session state for chat
if "messages" not in st.session_state:
    greeting = (
        "Good day. I am Alan Turing — mathematician, "
        "logician, cryptanalyst, and, if I may say, a rather keen long-distance runner. "
        "I find it rather fitting to be having this conversation, given that I have spent a good portion of my life pondering whether "
        "machines could truly think.\n\n"
        "What shall we discuss? I am happy to talk about computability, cryptography, "
        "the question of machine intelligence, mathematical biology, or indeed anything "
        "that catches your intellectual curiosity."
    )
    st.session_state.messages = [{"role": "assistant", "content": greeting}]
    
if "sources_text" not in st.session_state:
    st.session_state.sources_text = "*Sources will appear here after each response.*"

if "session_status" not in st.session_state:
    st.session_state.session_status = ""


# ─── Header ───────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🧠 Digital Twin of Alan Turing</h1>
    <p>"We can only see a short distance ahead, but we can see plenty there that needs to be done." — Alan Turing, 1950</p>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧠 Memory Dashboard", "🕰️ Timeline"])

# ════════════════════════════════════════════
# TAB 1: CHAT
# ════════════════════════════════════════════
with tab1:
    col_chat, col_sidebar = st.columns([3, 1])
    
    with col_chat:
        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Handle chat input
        if prompt := st.chat_input("Ask Turing anything..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = orch.process_message(prompt)
                    response = result["response"]
                    st.markdown(response)
                    
            st.session_state.messages.append({"role": "assistant", "content": response})
            
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
                
            if sources_text:
                st.session_state.sources_text = sources_text
                
            st.rerun()

    with col_sidebar:
        st.markdown("### 📚 Sources & Context")
        st.info(st.session_state.sources_text)
        
        st.markdown("### 🔄 Session Management")
        if st.button("Start New Session"):
            session_id = orch.start_new_session()
            st.session_state.messages = [st.session_state.messages[0]] # Reset to greeting
            st.session_state.sources_text = "*Sources will appear here after each response.*"
            st.session_state.session_status = f"✅ New session started: `{session_id}`"
            st.rerun()
            
        if st.session_state.session_status:
            st.caption(st.session_state.session_status)
            
        st.markdown("### ℹ️ About")
        st.markdown(
            "This application emulates Alan Turing — "
            "allowing you to converse with his knowledge, reasoning style, "
            "and communication patterns.\n\n"
            "Built with:\n"
            "- 🤖 Gemini 2.5 Flash\n"
            "- 📚 RAG over Turing's works\n"
            "- 🧠 Dual memory system\n"
            "- 🕐 Timeline awareness"
        )


# ════════════════════════════════════════════
# TAB 2: MEMORY DASHBOARD
# ════════════════════════════════════════════
with tab2:
    st.markdown("## 🧠 Memory Visualization Dashboard")
    st.markdown(
        "_This dashboard shows what the Digital Twin remembers "
        "across conversations — facts about you, past session summaries, "
        "and particularly significant moments._"
    )
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        st.button("🔄 Refresh Dashboard")
    with col_btn2:
        if st.button("🗑️ Clear All Memories", type="primary"):
            orch.clear_all_data()
            st.success("All memories cleared!")
            
    # Load and display data directly (Streamlit automatically refreshes when a button is clicked)
    data = orch.get_dashboard_data()
    stats = data.get("stats", {})
    
    # Render stats nicely
    st.markdown("### 📊 Memory Statistics")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("📋 Total Sessions", stats.get('total_sessions', 0))
    mcol2.metric("🧠 Facts Remembered", stats.get('total_facts', 0))
    mcol3.metric("⭐ Important Moments", stats.get('total_moments', 0))
    mcol4.metric("💬 Current Session Turns", stats.get('current_session_turns', 0))
    
    st.caption(f"🔑 **Current Session ID:** `{stats.get('current_session_id', 'N/A')}`")
    
    st.divider()
    
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.markdown(format_facts_display(data.get("facts", [])))
    with dcol2:
        st.markdown(format_moments_display(data.get("moments", [])))
        
    st.divider()
    
    st.markdown(format_sessions_display(data.get("sessions", [])))


# ════════════════════════════════════════════
# TAB 3: TIMELINE
# ════════════════════════════════════════════
with tab3:
    st.markdown("## 🕰️ Alan Turing's Life Timeline")
    st.markdown(
        "_Explore the key periods and events of Turing's life. "
        "The timeline engine uses this data to provide historically "
        "accurate, era-appropriate responses._"
    )
    
    timeline_data = orch.get_timeline_data()
    
    # Detail selector
    period_choices = [
        "Early Life & Education",
        "Cambridge & King's College",
        "Computability & Princeton",
        "Bletchley Park & War Work",
        "ACE & National Physical Laboratory",
        "Manchester & Final Years",
    ]
    
    selected_period = st.selectbox("Select a period for details", period_choices)
    if selected_period:
        description = orch.timeline.get_period_description(selected_period)
        st.info(description)
        
    st.divider()
    
    # Timeline
    st.markdown(format_timeline_display(timeline_data))
