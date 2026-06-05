import pytest
from memory.short_term import ShortTermMemory

def test_short_term_memory_sliding_window():
    memory = ShortTermMemory(window_size=3)
    memory.add_turn("user", "User1")
    memory.add_turn("assistant", "Bot1")
    memory.add_turn("user", "User2")
    memory.add_turn("assistant", "Bot2")
    memory.add_turn("user", "User3")
    memory.add_turn("assistant", "Bot3")
    memory.add_turn("user", "User4")
    memory.add_turn("assistant", "Bot4")
    
    recent = memory.get_recent_turns()
    # Should only contain the last 3 turns (each turn has user+assistant, total 6 items)
    assert len(recent) == 6
    assert recent[0]["content"] == "User2"
    assert recent[-1]["content"] == "Bot4"

