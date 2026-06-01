import pytest
from memory.short_term import EpisodicMemory

def test_episodic_memory_sliding_window():
    memory = EpisodicMemory(window_size=3)
    memory.add_interaction("User1", "Bot1")
    memory.add_interaction("User2", "Bot2")
    memory.add_interaction("User3", "Bot3")
    memory.add_interaction("User4", "Bot4")
    
    history = memory.get_context()
    # Should only contain the last 3 interactions
    assert len(history) == 3
    assert history[0]["user"] == "User2"
    assert history[-1]["user"] == "User4"
